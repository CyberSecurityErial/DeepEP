# Standalone 8-rail buffer balance

This directory contains a CUDA Runtime-only rail balancer.  It does not use
DeepEP, NCCL device APIs, NVSHMEM, CUTLASS, or a persistent kernel.

Measured latency, NVLink utilization, and comparison with the previous path
are recorded in [RESULTS.md](RESULTS.md).

For every logical partition, the planner computes a minimum-move distribution
over exactly eight rails.  Each rail reserves `ceil(global_count / 8)` records,
so the final RDMA payload size is identical on all rails.  When the count is
not divisible by eight, `local_valid_count` distinguishes the real records
from the single possible padding record.  Remainders are rotated and assigned
to rails that can retain a record first, minimizing peer traffic.

The hot path is two asynchronous launches on the caller's stream:

1. one warp-parallel GPU planner builds the public `PartitionInfo` and `Span`
   metadata (a serial fallback handles more than 32 partitions);
2. a finite CTA grid gathers only retained/incoming spans into the local rail
   buffer.

Small records use aligned 16-byte vector loads/stores.  On SM90+, records of at
least 4096 bytes use 16 KiB `cp.async.bulk` global-to-shared and
shared-to-global transfers (the Hopper TMA bulk engine).  The 14,368-byte
BF16/7168 case therefore selects TMA.  The threshold and engine are explicit
options.

`launch_unbalance()` consumes the same spans in reverse, which is useful for
return traffic after the network operation.  Padding is never returned.

The caller owns synchronization and pointer exchange.  `peer_input` and
`peer_output` must already be peer-accessible UVA mappings, obtained for
example through CUDA IPC, VMM, or a communicator window.  `all_counts` must be
the same rank-major `[8, num_partitions]` device snapshot on every rail.

Build and run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/rail_balance_test
./build/rail_balance_bench 14368 1024 200 auto 16 graph latency
```

The benchmark also accepts two optional trailing arguments:

```text
[skew|uniform] [balance|roundtrip]
```

`uniform` gives every Rail the same count for one partition and therefore
measures zero cross-Rail movement. `roundtrip` measures balance followed by
unbalance, which is the fair comparison with a dispatch+combine RailBalance
overhead. For example:

```bash
./build/rail_balance_bench \
  14368 1024 200 vector 16 graph latency skew roundtrip
```

## DeepEP integration boundary

This implementation is now kept inside the paper repository as
`tests/elastic/rail_balance_v2`. It consumes a partition-major packed record
buffer. DeepEP's public dispatch input is token-major and may contain several
remote destinations per token, so replacing the production kernel still
requires a packing adapter or moving this operation after an existing pack
stage. The standalone latency must not silently omit that adapter cost.

Graph capture is optional. It captures a fixed-capacity two-kernel skeleton,
not a fixed balance result: `all_counts` is reread and all spans are rebuilt on
every replay, so token counts and cross-rail traffic may change. Captured device
addresses, `record_bytes`, partition count, capacity, and launch options must
remain fixed. Use `eager` instead of `graph` when those structural parameters
change, and use `throughput` instead of `latency` to measure queued steady-state
launch intervals.

The default fat binary contains SM89 for local fallback testing and SM90 for
the TMA path.  A production Hopper-only build can set
`-DBALANCE_CUDA_ARCHITECTURES=90`.
