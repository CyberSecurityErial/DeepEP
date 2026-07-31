# RailBalance Hop-Aware Minimal Design

Base audit: `docs/rail_balance_hop_aware_audit.md`  
Reference: `tests/elastic/rail_balance_hop_reference.py`

## 1. Non-negotiable semantics

DeepEP sends one payload per token and remote destination node. A payload may
serve several destination-local expert ranks. The planner record is therefore:

```text
(owner o, destination node d, target-rank mask T)
```

The selected Rail is `e`. For `T={t}`:

```text
e=o=t       direct
e=o!=t      destination-forward
e=t!=o      source-forward
e not o,t   two-hop
```

For a shared multi-target payload, exact local cross-GPU copies are:

```text
[e != o] + |T - {e}|
```

One-hop candidates are `{o} union T`. Third-Rail candidates are outside that
set. Final experts and the number of scale-out payloads never change.

## 2. Why the legacy compact plan cannot be extended by one field

The current plan groups only `(owner,destination)` counts and moves an ordinal
suffix to arbitrary deficit Rails. Two payloads in the same group can have
different target masks and hence different legal endpoint Rails. Adding one
`t` to a five-int segment cannot represent that distinction.

Expanding the dense count to `[G,C,D,2^G]` would make the fixed symmetric arena
scale with all possible masks and the maximum 1024 channels. That is not an
acceptable production layout.

## 3. Minimal hop-aware plan representation

The first GPU implementation uses a bounded dense table indexed by data that
already exists:

```text
record index = [owner][token][remote-destination ordinal]
capacity     = G * num_max_tokens_per_rank * num_topk
```

Each token has at most `num_topk` distinct remote destination nodes, so this
does not duplicate a payload. Unused entries have `target_mask == 0`.

```cpp
struct HopCopyRecord {
    uint32_t target_mask;
    int32_t destination;
};

struct HopCopyResolution {
    int32_t egress;
    int32_t channel;
    int32_t remote_slot;
    int32_t proxy_slot;  // -1 when egress == owner
};
```

`path_kind` is derived from `(o,T,e)` and emitted in diagnostics; it is not a
second source of truth in the hot ABI.

The initial structs stay unpacked. A packed 64-bit ABI is allowed only after
measurement shows metadata bandwidth or footprint on the exposed path. This
keeps range checking and review straightforward during correctness work.

## 4. Planner phases

### H0: local record materialization

One local GPU kernel scans top-k once, deduplicates by destination node, builds
the target-rank mask, and writes the fixed local record table in token-major
order. Full expert IDs remain in the existing `TokenLayout`.

### H1: node-local snapshot

The existing LSA barrier and peer-pointer ownership are reused. Each rank
copies the fixed local record table from all local owners into its plan output.
No new cross-node collective is introduced.

### H2: deterministic assignment

The correctness-first kernel scans records in a fixed order and applies the
same endpoint score as the Python oracle:

```text
pair Rail peak, source Rail peak, local forwards, seeded stable tie
```

`one_hop` considers only `{o} union T`. `adaptive` starts from that result and
then applies the bounded third-Rail pass. `legacy_exact` continues to use the
existing compact planner and kernels unchanged.

The first implementation uses one deterministic planner CTA. It is deliberately
simple and may be replaced by a parallel materializer only after correctness
and planner timing identify it as material. GPU completion order must never
change the plan.

### H3: static slot materialization

The planner makes three deterministic passes over the records:

1. choose `e` and count `[e][channel][d]` retained/moved groups;
2. prefix those groups and allocate each egress's proxy range;
3. write every record's `remote_slot` and `proxy_slot`.

The committed data path therefore performs one resolution load and no hot
`atomicAdd(peer_tail)`. Existing `retained`, `moved`, `group_prefix`,
`proxy_required`, proxy arenas, and egress send loops remain reusable.

## 5. Dispatch behavior

```text
e == o
    stage in the existing retained group; no source proxy copy.

e != o
    LSA/TMA directly into e's existing final proxy-dispatch slot; publish the
    existing invocation ready word.
```

On the destination:

```text
e in T
    the ingress copy for rank e is already final; do not issue an LSA copy to
    self. Forward only to T-{e}.

e not in T
    forward to every target in T (third-Rail path).
```

For `T={e}`, source-forward therefore has no destination forwarding. For
`o=t=e`, direct has neither source nor destination forwarding.

## 6. Combine behavior

Combine does not run a planner. It replays dispatch forward metadata and the
same proxy slot:

```text
expert t -> destination ingress e -> source proxy e -> original owner o
```

The existing moved-return path and return-unshuffle already implement this
inverse. Source-forward/direct records with no source proxy return directly to
the original owner slot. The one-shot ticket remains the lifetime authority.

## 7. Mode isolation

```text
off
    native DeepEP V2 objects, JIT keys, buffers, and kernels unchanged.

legacy_exact
    current force planner and data path unchanged.

one_hop / adaptive
    separate experimental plan storage and JIT specializations.
```

The separate specialization prevents endpoint metadata from changing legacy
codegen by accident. It is not a duplicated transport implementation: source
shuffle, arena, egress send, destination forwarding, combine, and epilogues are
shared helpers wherever their semantics match.

Public capability remains false until real multi-node Rail/Gin correctness.
Invalid configuration or capacity fails the complete transaction; there is no
silent partial plan or hidden fallback.

## 8. First correctness checkpoints

1. GPU record materialization equals a Python target-mask oracle.
2. GPU `one_hop` resolution equals the Python plan for `chunk_size=1`.
3. Source shuffle uses the resolution table and preserves payload count.
4. Vnode proves `e in T` does not duplicate destination forwarding.
5. Combine round trip equals `off`.
6. Add bounded adaptive third-Rail assignment and diagnostics.
7. Generalize chunk size only after the copy-level path is correct.

This ordering intentionally avoids a descriptor ring, global optimizer,
independent combine planner, or a second benchmark framework.

## 9. Performance work starts later

After end-to-end correctness, measure a profiler-free baseline, use Nsight
Systems to find exposed planner/shuffle/forward time, and use Nsight Compute
only on a proven critical kernel. Candidate optimizations are planner
parallelism, resolution packing, direct receive layout, source/RDMA overlap,
segment coalescing, TMA tiles, and SM split. None is accepted solely from a
single-kernel profiler number.

## 10. One benchmark entry

`tests/elastic/bench_rail_balance_hop.py` is the shared entry. The workload,
planner configuration, path counters and Rail-load schema are identical across
the three backends; only the executor and claim scope change.

Planner-only reference:

```bash
PYTHONPATH=$PWD/tests/elastic:$PWD \
/home/chen/.cache/deepep-sjlgpt/bin/python \
  tests/elastic/bench_rail_balance_hop.py \
  --backend reference --mode one_hop --case offdiag_hot \
  --num-nodes 2 --gpus-per-node 4 --tokens-per-rank 16 \
  --topk 1 --num-experts 16 --hidden 256 \
  --output-json /tmp/rail-hop-reference.json
```

Eight-GPU 4x2 vnode round trip:

```bash
EP_DISABLE_GIN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHONPATH=$PWD/tests/elastic:$PWD \
/home/chen/.cache/deepep-sjlgpt/bin/python \
  tests/elastic/bench_rail_balance_hop.py \
  --backend vnode --mode adaptive --case diag_hot \
  --num-nodes 2 --gpus-per-node 4 --tokens-per-rank 16 \
  --topk 1 --num-experts 16 --hidden 256 \
  --max-two-hop-percent 50 --hop-penalty-percent 0 \
  --output-json /tmp/rail-hop-vnode.json
```

Real multinode execution uses the same entry under `torchrun`; it delegates to
the existing strict profiler-free A/B runner and always compares the candidate
against native DeepEP V2 `off`:

```bash
torchrun --nnodes=2 --nproc-per-node=4 \
  --node-rank=$NODE_RANK --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT \
  tests/elastic/bench_rail_balance_hop.py \
  --backend multinode --mode one_hop --case offdiag_hot \
  --num-nodes 2 --gpus-per-node 4 --tokens-per-rank 4096 \
  --topk 4 --num-experts 128 --hidden 7168 \
  --warmup-iters 10 --steady-iters 100 \
  --output-json /shared/rail-hop-multinode.json
```

The result marks `planner_only`, `single_node_diagnostic`, or
`real_multinode`. Reference and vnode output are never network-speed claims.
Endpoint-count traces use the JSON files under
`tests/elastic/workloads/rail_balance/` with `--case trace --workload-json`.

## 11. Keep planning off the exposed data path

The current exact GPU oracle makes one ordered endpoint decision per payload.
Nsight Systems shows that decision loop, not source shuffle, dominates the
measured path.  The production split is therefore:

```text
low-frequency control plane (CPU/background)
    threshold, hop penalty, chunk policy, measured cost update

per-routing GPU data plane
    top-k -> small endpoint histogram -> chunk quotas -> parallel slots
```

The CPU must not receive the per-token table: D2H, host planning, H2D and the
required synchronization would replace one exposed cost with three.  A cached
routing handle may reuse its completed plan.  For fresh routing, microbatch
`n+1` may plan on a separate stream after its router output exists while
microbatch `n` runs experts; the dispatch consuming `n+1` still has a true
dependency on that plan.

Reference measurements on the C100 rot1 endpoint matrix establish the first
chunk bound before changing CUDA:

| chunk | Python planner | pair peak | source peak |
|------:|---------------:|----------:|------------:|
| 1 | 608.358 ms | 1792 | 11193 |
| 8 | 74.607 ms | 1792 | 11192 |
| 32 | 17.038 ms | 1792 | 11168 |

Across 128 small random matrices, chunk 8 changed the worst pair/source peak
by 4.37%/3.46%; chunk 32 reached 12.66%/13.88%.  Consequently a large fixed
chunk is not a safe default.  The GPU fast path will aggregate by endpoint
group, use a conservative configurable minimum chunk, and enlarge it only for
large groups under a bounded decision budget.  `chunk_size=1` remains the
exact oracle and comparison path.

### 11.1 CUDA parallel ownership before implementation

Every new GPU stage must define this map before code is written. The first
multi-block prototype uses one warp per block so that one `(owner, channel)`
has a single deterministic writer; the large grid, rather than atomics inside
one group, supplies parallelism.

| stage | grid | block / warp | lane work | work per block |
|---|---:|---:|---|---|
| endpoint record | `G*C` | 1 warp | one top-k lane | about `N/C` tokens |
| endpoint decision | 1 | 1 warp | lane 0 preserves group order; other lanes help reductions | endpoint groups only, at most 32 chunk decisions per large group |
| record assignment | `G*C` | 1 warp | one top-k lane | resolve about `N/C` tokens from group quota |
| group count | `G*C` | 1 warp | one top-k lane | count final `(egress,C,D)` ownership for about `N/C` tokens |
| slot prefix | `G` | 1 warp initially | lanes stripe channel/destination groups | one egress's compact counters |
| slot finalize | `G*C` | 1 warp | one top-k lane | add static group bases to about `N/C` tokens |

The serialized stage may decide only coarse endpoint chunks; it must not scan
channels or allocate per-copy slots. With `G=8`, `C=256`, and `N=8192`, the
dense stages expose 2,048 independent blocks and each block sees roughly 32
tokens. Launch boundaries provide the only grid-wide ordering. A later fusion
is allowed only after profiling proves launch cost material; correctness is
not based on cooperative-launch residency or cross-block spin barriers.
