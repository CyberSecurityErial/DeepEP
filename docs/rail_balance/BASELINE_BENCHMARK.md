# DeepEP V2 baseline comparison

## Question answered

The production comparison has one baseline:

```text
DeepEP V2 Hybrid: rail_balance="off"
```

The candidate uses the same checkout, inputs, topology, SM/QP geometry and
precision, changing only:

```text
rail_balance="force"
rail_balance_policy=<all|active|adaptive>
rail_balance_threshold_percent=<integer>
```

Local checked-adapter or vnode measurements are component diagnostics.  They
must not be reported as a speedup over DeepEP V2 because they do not execute
the real Gin/RDMA path.

## Measurement boundary

Each measured iteration is the public Hybrid round trip:

```text
dispatch -> combine with that dispatch's handle
```

The force handle is consumed exactly once in the same iteration.  Buffer
construction, JIT warmup, reference generation, correctness checks, report
writing and process-group barriers are outside the steady measurement block.
CUDA events are recorded for dispatch, combine and the full round trip, then
synchronized once at the end of the block so the benchmark does not turn the
pipeline into one synchronized launch at a time.

Host and GPU clocks are not comparable across nodes.  Each rank therefore
computes its own elapsed durations; the distributed iteration result is the
maximum rank-local elapsed duration for that ordinal.  Every rank-local raw
sample is retained.

## Run order and statistics

Fresh buffers are measured in symmetric blocks so the V2 baseline is not
always colder or hotter than the candidate:

```text
off, force, force, off, force, off, off, force
```

Formal collection uses at least 10 warmup and 100 steady iterations per block.
Reports retain raw samples plus median, p95, p99, mean, population standard
deviation, CV, minimum and maximum.  The primary result is full round-trip
latency; dispatch and combine are supporting breakdowns.

The comparison reports candidate speedup as:

```text
DeepEP V2 off median / rail-balance force median
```

Balanced traffic is required as a negative control.  A useful optimization
must not buy skewed-case speedup by imposing material overhead when no move is
needed.

## Acceptance gate

A report is not performance evidence when any of these hold:

- fewer than 10 warmup or 100 steady iterations per block;
- Git is dirty or ranks do not share the same commit and extension binary;
- a profiler or diagnostic NVTX mode is enabled;
- any node has an unexpected GPU compute process, MPS/MIG mismatch, non-benign
  throttle state, or materially changing clocks/power state;
- off and force do not pass the same public dispatch/combine correctness
  reference and exact output comparison;
- topology, input, precision, SM/QP configuration, or run order differs;
- the route is the capacity-failure fixture rather than a valid workload.

Failure of the gate preserves the raw report for debugging but forbids a
speedup claim.  Nsys and NCU runs are separate diagnostic experiments; the
final truth is always the profiler-free off/force rerun.
