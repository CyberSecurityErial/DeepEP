# RailBalance v2 on H800

Measured on 2026-09-01. One node has eight NVIDIA H800 80 GB GPUs. Each GPU
reports eight active NVLinks at 26.562 GB/s per link, or about 212.5 GB/s per
direction. Driver is 580.126.09; the code was built for SM90 with CUDA 12.9.

The correctness test passed balance followed by unbalance for 30-byte,
64-byte, and 14,368-byte records. Every latency below is the maximum CUDA-event
duration over eight concurrent GPU workers, with 100 warmups and 200 measured
samples. CPU scheduling is outside the interval.

## Fair roundtrip comparison

The old table measured dispatch+combine roundtrip. The new number therefore
also measures `balance -> unbalance`; using balance-only latency here would
overstate the improvement.

`uniform` has identical per-Rail counts and zero cross-Rail records. It is the
cleanest replacement for the old alpha=0 overhead column. `skew` moves about
52.6% of records across Rails. The best H800 settings found by the tuning sweep
are TMA/16 blocks per span for uniform traffic and vector/16 for skew traffic.

| Size/node | Record/Rail | Old overhead | New uniform Graph | Overhead reduction | New skew Graph | Projected full-total reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 3 | 141.280 us | 17.730 us | 87.45% | 25.180 us | 19.65% |
| 256 | 5 | 83.408 us | 19.650 us | 76.44% | 25.220 us | 11.85% |
| 512 | 11 | 79.536 us | 20.100 us | 74.73% | 25.340 us | 11.00% |
| 1,024 | 21 | 92.368 us | 20.190 us | 78.14% | 25.950 us | 12.85% |
| 2,048 | 43 | 84.352 us | 23.550 us | 72.08% | 26.820 us | 8.73% |
| 4,096 | 85 | 92.944 us | 26.910 us | 71.05% | 32.540 us | 6.35% |
| 8,192 | 171 | 102.672 us | 35.420 us | 65.50% | 41.860 us | 4.00% |
| 16,384 | 341 | 107.184 us | 50.140 us | 53.22% | 61.310 us | 1.95% |
| 49,152 | 1,024 | 143.456 us | 144.260 us | -0.56% | 136.260 us | -0.01% |

The Graph launch saves only a few microseconds. Eager roundtrip is 21.44-145.98
us for uniform traffic and 26.53-137.73 us for skew traffic; all values are in
`H800_RESULTS.csv`.

## What can be used in the paper

The strong result is the small and medium message region: the isolated
roundtrip overhead falls by 53-87% through Size=16,384. At Size=49,152 the
zero-movement roundtrip is bandwidth-dominated and merely matches the previous
overhead. Under actual cross-Rail movement, the optimized two-way operator
still stays between 25.18 and 136.26 us over the tested range.

`Projected full-total reduction` adds the new uniform Graph time to the old
Native latency. It is a projection, not a new DeepEP end-to-end measurement.

## Integration boundary

The v2 API consumes a partition-major packed record buffer, peer-accessible
pointers, and a ready `[8, num_partitions]` count snapshot. DeepEP currently
enters RailBalance with token-major data and builds its counts inside the old
path. Consequently these measurements include the v2 GPU planner and both copy
directions, but exclude any new token packing adapter, count collection, and
pointer exchange. The table is valid as an operator ablation; it must not be
presented as an already measured end-to-end DeepEP speedup until that adapter
is connected and timed.

For one-way balance only, vector with eight blocks per span is best for the
large skew point (75.42 us at 1,024 records/Rail), while TMA is much better for
uniform local materialization. This traffic-dependent engine crossover is a
natural input to EchoP's existing table-driven policy hook.

## Updating the old 63-point matrix

`RAIL_BOUNDARY_V2_PROJECTED.csv` replaces the old alpha=0 overhead with the new
uniform roundtrip overhead while leaving the measured network term unchanged.
This is appropriate as an overhead-only projection, but it is not relabeled as
a new end-to-end measurement.

The CSV carries an `evidence_status` column. Nine alpha=0 rows use the directly
measured new overhead. Forty-one rows retain the same qualitative conclusion
and are projection-only. Thirteen boundary rows are explicitly marked for
end-to-end retest: eight would flip from a loss to a win, and five more land
within 5% of break-even. These rows must not be promoted to headline results
from subtraction alone.
