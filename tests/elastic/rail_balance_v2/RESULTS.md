# Rail-balance benchmark results

Measured 2026-09-01 on one 8-GPU NVIDIA L20X node. CUDA Runtime reports SM90,
and `nvidia-smi` reports 18 active NVLinks per GPU. Latency is the maximum
CUDA-event duration over eight persistent host workers launching concurrently;
P50/P95 are calculated from 200 single-operation samples. Every sample includes
the GPU planner and the copy kernel. Pointer exchange, count all-gather, RDMA,
and consumer synchronization are outside this operator.

The workload has eight partitions and a rotating hot partition per rail. At
1024 records/rail, 4312 of 8192 records (52.64%) cross a rail. TMA is selected
at 4096 bytes. The graph results below capture the fixed two-kernel execution
shape, while the planner still reads the current `all_counts` and regenerates
all spans on every replay.

## Payload sweep at 1024 records/rail

NVLink utilization uses the Hopper payload ceiling of 450 GB/s per GPU per
direction, or 3.6 TB/s over eight GPUs. `Logical NVLink` counts each remote
record once rather than once as Tx and again as Rx.

| Payload | Engine | P50 | P95 | Cross-rail data | Logical NVLink | Ceiling utilization |
|---:|:---:|---:|---:|---:|---:|---:|
| 16 B | vector | 21.57 us | 24.93 us | 0.069 MB | 3.20 GB/s | 0.09% |
| 32 B | vector | 20.58 us | 24.10 us | 0.138 MB | 6.71 GB/s | 0.19% |
| 64 B | vector | 21.82 us | 25.50 us | 0.276 MB | 12.65 GB/s | 0.35% |
| 128 B | vector | 22.72 us | 25.66 us | 0.552 MB | 24.29 GB/s | 0.67% |
| 256 B | vector | 23.78 us | 27.49 us | 1.104 MB | 46.43 GB/s | 1.29% |
| 512 B | vector | 23.81 us | 28.42 us | 2.208 MB | 92.73 GB/s | 2.58% |
| 1,024 B | vector | 24.00 us | 27.71 us | 4.415 MB | 183.98 GB/s | 5.11% |
| 2,048 B | vector | 24.80 us | 28.35 us | 8.831 MB | 356.09 GB/s | 9.89% |
| 4,096 B | TMA | 31.94 us | 34.98 us | 17.662 MB | 553.04 GB/s | 15.36% |
| 8,192 B | TMA | 38.94 us | 44.61 us | 35.324 MB | 907.04 GB/s | 25.20% |
| 14,368 B | TMA | 52.70 us | 55.65 us | 61.955 MB | 1,175.52 GB/s | 32.65% |
| 16,384 B | TMA | 53.12 us | 61.95 us | 70.648 MB | 1,329.97 GB/s | 36.94% |
| 32,768 B | TMA | 82.50 us | 93.82 us | 141.296 MB | 1,712.76 GB/s | 47.58% |
| 65,536 B | TMA | 136.64 us | 140.77 us | 282.591 MB | 2,068.14 GB/s | 57.45% |

The 4096-byte step is the configured vector-to-TMA crossover. It is a policy
choice favoring low SM occupancy for large records, not the latency-optimal
crossover in isolation.

NVLink byte counters were checked independently. For a 65,536-byte payload,
1,000 measured iterations plus 100 warmups predict 310.850 GB of cross-rail
traffic. `nvidia-smi nvlink --getthroughput d` increased by exactly 310.850 GB
in Tx and 310.850 GB in Rx.

## Comparison with the previous DeepEP-based path

The supplied `Size/node` table maps approximately to the token counts below.
The new measurement fixes a record at 14,368 bytes and rounds tokens/GPU to the
nearest integer. `Projected total reduction` assumes Native is unchanged and
replaces only the old RailBalance overhead.

| Size/node | Tokens/GPU | Old overhead | New graph P50 | New eager P50 | Graph reduction | Old/new | Projected total reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 2.7 -> 3 | 141.280 us | 21.09 us | 23.78 us | 85.1% | 6.70x | 19.12% |
| 256 | 5.3 -> 5 | 83.408 us | 22.43 us | 24.00 us | 73.1% | 3.72x | 11.33% |
| 512 | 10.7 -> 11 | 79.536 us | 22.59 us | 24.96 us | 71.6% | 3.52x | 10.54% |
| 1,024 | 21.3 -> 21 | 92.368 us | 23.14 us | 25.18 us | 74.9% | 3.99x | 12.33% |
| 2,048 | 42.7 -> 43 | 84.352 us | 23.26 us | 25.66 us | 72.4% | 3.63x | 8.78% |
| 4,096 | 85.3 -> 85 | 92.944 us | 24.77 us | 26.66 us | 73.3% | 3.75x | 6.55% |
| 8,192 | 170.7 -> 171 | 102.672 us | 28.19 us | 29.31 us | 72.5% | 3.64x | 4.43% |
| 16,384 | 341.3 -> 341 | 107.184 us | 34.18 us | 36.16 us | 68.1% | 3.14x | 2.50% |
| 49,152 | 1,024 | 143.456 us | 52.48 us | 54.46 us | 63.4% | 2.73x | 1.15% |

Old overhead median/mean is 92.944/103.022 us. New graph P50 median/mean
is 23.260/28.014 us, a 75.0% median reduction and 72.8% mean reduction. Even
without graph capture, eager median/mean is 26.660/30.019 us, a 71.3%/70.9%
reduction. Graph saves about 2.0 us on average in this eight-rank single-shot
measurement; it is useful but is not responsible for most of the speedup.

The new overhead is 4.3-4.9% of Native at the four smallest points and falls
to 0.67% at the largest point. For the isolated 14,368-byte benchmark, a
forced vector path can be 1-3 us faster once the record count is large. Auto
intentionally retains TMA because it uses one warp per CTA instead of the
vector kernel's eight warps; co-run SM pressure should be evaluated with the
real compute operator before changing this policy.

## CUDA Graph applicability

Changing the balance decision does not invalidate the graph. CUDA Graph fixes
the two kernel nodes, their argument values, and their grid dimensions; it does
not freeze device memory contents. On every replay the planner rereads the
current rank-major count matrix at the captured `all_counts` address and
rewrites `PartitionInfo` and `Span`. The copy grid is provisioned for the fixed
capacity, but each CTA reads the newly produced `span.record_count`; empty spans
exit and non-empty spans copy only the current payload.

Graph replay is therefore valid when these remain fixed:

- `record_bytes`, `num_partitions`, capacity, and launch options;
- input, output, count, and workspace device addresses.

Counts, quotas, source rails, cross-rail fraction, and actual copied bytes may
change every iteration. If addresses or shapes change, use eager launch, update
the graph node parameters, or cache one graph executable per shape class.

## Distance from the hardware limit

The local H200 reference reports 4.736 us for one eager empty kernel and
5.152 us for a two-node CUDA Graph. The two eager launches therefore have a
9.472-us empty-launch floor; graph reduces that floor by 4.320 us. In the
actual eight-rank test, graph is only about 2 us faster because GPU work and
rank-tail effects dominate.

For the target 14,368-byte, 1024-record case, 61.955 MB crosses NVLink. At the
physical 3.6-TB/s ceiling its wire time is 17.210 us. Adding only the two-node
graph floor gives an intentionally optimistic absolute lower bound of
22.362 us. The measured 52.70 us is 2.36x this bound, leaving 30.34 us (57.6%)
between the current implementation and a zero-cost planner plus perfect-wire
limit.

Nsight Systems graph-node tracing reports median device durations of 9.440 us
for the warp planner and 35.776 us for the TMA copy. The remaining 7.484 us is
graph scheduling plus the eight-rank tail; 5.152 us of that is already present
for two empty graph nodes.

| Target-case component | Time |
|---|---:|
| Warp planner device work | 9.440 us |
| TMA copy device work | 35.776 us |
| Graph scheduling + eight-rank tail | 7.484 us |
| End-to-end P50 | 52.70 us |

Over the full end-to-end interval, logical NVLink bandwidth is 1.176 TB/s, or
32.65% of the physical ceiling. During the measured copy-kernel window it is
about 1.732 TB/s, or 48.1%; subtracting graph and planner time from the
end-to-end interval gives a similar 45.2% estimate. The copy kernel also moves
local retained records, so raw NVLink utilization alone does not describe all
of its memory work.

The size-matched empirical bound is more useful than 450 GB/s. Each GPU receives
7.386 MiB of remote data. Interpolating the reference table's 4-MiB and 16-MiB
peer-TMA graph results gives 27.21 us, or 2.277 TB/s aggregate. The current
35.776-us fragmented multi-peer copy reaches about 76.1% of that contiguous
same-size path. Adding the reference's marginal second graph node (0.736 us)
and the current 9.440-us planner gives a practical target near 37.39 us; the
remaining practical gap is about 15.31 us.

At 65,536 bytes, fixed costs are better amortized: end-to-end bandwidth is
2.068 TB/s (57.45% of physical), while the 117.984-us copy window reaches
2.395 TB/s (66.5%). The size-matched peer-TMA interpolation is 101.23 us, so
the copy reaches 85.8% of that empirical reference. The absolute graph-plus-wire
floor is 83.65 us versus the measured 136.64 us.

The next useful optimization targets are therefore the fragmented span copy
scheduler/source topology (about 15 us at the target point versus the
same-size reference), followed by planner fusion or cheaper planning if the
metadata contract can change. Graph capture is optional and provides a much
smaller incremental gain than the warp planner and standalone copy path.
