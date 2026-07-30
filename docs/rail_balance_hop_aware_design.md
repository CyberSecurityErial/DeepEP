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
