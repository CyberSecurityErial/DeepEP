-- O077 Nsight Systems 2024.6.2 interval queries.
--
-- Run this file against one exported SQLite report with
-- `sqlite3 -readonly REPORT < docs/rail_balance/sql/o077_nsys_queries.sql`
-- when that CLI is installed.  This host has no sqlite3 executable; the retained
-- verification used `/home/chen/.cache/deepep-sjlgpt/bin/python` and its
-- read-only Python sqlite3 3.53.2 connection to executescript this exact file.
-- Change only the single row in `o077_params` between the source and return
-- configurations shown below.  All timestamps are nanoseconds.  These
-- queries intentionally use schema fields observed in the retained 2024.6.2
-- reports; inspect a new report's schema before reusing them on another Nsys
-- version.
--
-- Source parameters:
--   ('c100/source/stage/steady/',
--    '%rail_balance_hybrid_source_shuffle_impl<(int)7168, (int)4>%', 0)
-- Return parameters:
--   ('c100/return/stage/steady/',
--    '%rail_balance_hybrid_return_unshuffle_impl<(int)7168, (int)4>%', 1)

DROP TABLE IF EXISTS temp.o077_params;
CREATE TEMP TABLE o077_params(
    range_prefix TEXT NOT NULL,
    target_kernel_like TEXT NOT NULL,
    include_barrier INTEGER NOT NULL
);

-- Default invocation analyzes the post-O077 source report.
INSERT INTO o077_params VALUES(
    'c100/source/stage/steady/',
    '%rail_balance_hybrid_source_shuffle_impl<(int)7168, (int)4>%',
    0
);

-- The report contract requires exactly one outer range.
SELECT COUNT(*) AS outer_ranges,
       MIN(end - start) AS outer_duration_ns,
       MAX(end - start) AS outer_duration_ns_max
FROM NVTX_EVENTS
WHERE text = 'c100_nsys_window';

-- Exact launch-identity/cardinality audit.  globalTid's low 24 bits are the
-- thread id; CUPTI globalPid keeps those bits clear.  Kernel names are decoded
-- through StringIds rather than assumed to live in the kernel table.
WITH ranges AS MATERIALIZED (
    SELECT n.start AS range_start,
           n.end AS range_end,
           CAST(substr(n.text, length(p.range_prefix) + 1) AS INTEGER)
               AS ordinal,
           (n.globalTid & -16777216) AS globalPid
    FROM NVTX_EVENTS AS n
    CROSS JOIN o077_params AS p
    JOIN NVTX_EVENTS AS outer
      ON outer.text = 'c100_nsys_window'
     AND n.start >= outer.start
     AND n.end <= outer.end
    WHERE n.text GLOB (p.range_prefix || '[0-9]*')
), target AS MATERIALIZED (
    SELECT r.ordinal, k.deviceId, k.start, k.end,
           k.end - k.start AS duration_ns,
           k.contextId, k.streamId, k.gridX, k.blockX,
           k.registersPerThread, k.dynamicSharedMemory
    FROM ranges AS r
    JOIN CUPTI_ACTIVITY_KIND_KERNEL AS k
      ON k.globalPid = r.globalPid
     AND k.start >= r.range_start
     AND k.end <= r.range_end
    JOIN StringIds AS s ON s.id = k.demangledName
    CROSS JOIN o077_params AS p
    WHERE s.value LIKE p.target_kernel_like
)
SELECT COUNT(*) AS launches,
       COUNT(DISTINCT ordinal) AS ordinals,
       MIN(deviceId) AS min_device,
       MAX(deviceId) AS max_device,
       MIN(contextId) AS min_context,
       MAX(contextId) AS max_context,
       MIN(streamId) AS min_stream,
       MAX(streamId) AS max_stream,
       MIN(gridX) AS min_grid_x,
       MAX(gridX) AS max_grid_x,
       MIN(blockX) AS min_block_x,
       MAX(blockX) AS max_block_x,
       MIN(registersPerThread) AS min_registers,
       MAX(registersPerThread) AS max_registers,
       MIN(dynamicSharedMemory) AS min_dynamic_shared,
       MAX(dynamicSharedMemory) AS max_dynamic_shared
FROM target;

-- Post-report target-duration distributions.  This preserves the distinction
-- between pooled launches, one physical device, per-iteration stragglers and
-- completion skew; none of these durations are summed across overlapping
-- ranks.
WITH ranges AS MATERIALIZED (
    SELECT n.start AS range_start,
           n.end AS range_end,
           CAST(substr(n.text, length(p.range_prefix) + 1) AS INTEGER)
               AS ordinal,
           (n.globalTid & -16777216) AS globalPid
    FROM NVTX_EVENTS AS n
    CROSS JOIN o077_params AS p
    JOIN NVTX_EVENTS AS outer
      ON outer.text = 'c100_nsys_window'
     AND n.start >= outer.start
     AND n.end <= outer.end
    WHERE n.text GLOB (p.range_prefix || '[0-9]*')
), target AS MATERIALIZED (
    SELECT r.ordinal, k.deviceId, k.start, k.end,
           k.end - k.start AS duration_ns
    FROM ranges AS r
    JOIN CUPTI_ACTIVITY_KIND_KERNEL AS k
      ON k.globalPid = r.globalPid
     AND k.start >= r.range_start
     AND k.end <= r.range_end
    JOIN StringIds AS s ON s.id = k.demangledName
    CROSS JOIN o077_params AS p
    WHERE s.value LIKE p.target_kernel_like
), per_ordinal AS (
    SELECT ordinal,
           MAX(duration_ns) AS max_duration_ns,
           MAX(end) - MIN(end) AS completion_skew_ns
    FROM target
    GROUP BY ordinal
), metrics(metric, value_ns) AS (
    SELECT 'pooled_target', duration_ns FROM target
    UNION ALL
    SELECT 'device6_target', duration_ns FROM target WHERE deviceId = 6
    UNION ALL
    SELECT 'per_ordinal_max', max_duration_ns FROM per_ordinal
    UNION ALL
    SELECT 'completion_skew', completion_skew_ns FROM per_ordinal
), ranked AS (
    SELECT metric, value_ns,
           ROW_NUMBER() OVER (PARTITION BY metric ORDER BY value_ns) AS rn,
           COUNT(*) OVER (PARTITION BY metric) AS n
    FROM metrics
)
SELECT metric, MIN(n) AS samples, AVG(value_ns) / 1000.0 AS p50_us
FROM ranked
WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
GROUP BY metric
ORDER BY metric;

-- Build one raw row per steady ordinal.  The target union merges all eight
-- rank/device kernels.  The excluded union is the intersection of that target
-- union with every exact-range memcpy; for return it also includes the exact-
-- range B1/B4 `barrier_impl` kernels.  Clipped overlaps are merged before
-- summation so overlapping rank work is never double counted.
DROP TABLE IF EXISTS temp.o077_interval_result;
CREATE TEMP TABLE o077_interval_result AS
WITH ranges AS MATERIALIZED (
    SELECT n.start AS range_start,
           n.end AS range_end,
           CAST(substr(n.text, length(p.range_prefix) + 1) AS INTEGER)
               AS ordinal,
           (n.globalTid & -16777216) AS globalPid
    FROM NVTX_EVENTS AS n
    CROSS JOIN o077_params AS p
    JOIN NVTX_EVENTS AS outer
      ON outer.text = 'c100_nsys_window'
     AND n.start >= outer.start
     AND n.end <= outer.end
    WHERE n.text GLOB (p.range_prefix || '[0-9]*')
), target AS MATERIALIZED (
    SELECT r.ordinal, k.start, k.end
    FROM ranges AS r
    JOIN CUPTI_ACTIVITY_KIND_KERNEL AS k
      ON k.globalPid = r.globalPid
     AND k.start >= r.range_start
     AND k.end <= r.range_end
    JOIN StringIds AS s ON s.id = k.demangledName
    CROSS JOIN o077_params AS p
    WHERE s.value LIKE p.target_kernel_like
), excluded_intervals AS MATERIALIZED (
    SELECT r.ordinal, m.start, m.end
    FROM ranges AS r
    JOIN CUPTI_ACTIVITY_KIND_MEMCPY AS m
      ON m.globalPid = r.globalPid
     AND m.start >= r.range_start
     AND m.end <= r.range_end
    UNION ALL
    SELECT r.ordinal, k.start, k.end
    FROM ranges AS r
    JOIN CUPTI_ACTIVITY_KIND_KERNEL AS k
      ON k.globalPid = r.globalPid
     AND k.start >= r.range_start
     AND k.end <= r.range_end
    JOIN StringIds AS s ON s.id = k.demangledName
    CROSS JOIN o077_params AS p
    WHERE p.include_barrier = 1
      AND s.value LIKE '%barrier_impl<%'
), target_ordered AS (
    SELECT ordinal, start, end,
           MAX(end) OVER (
               PARTITION BY ordinal ORDER BY start, end
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
           ) AS prior_max_end
    FROM target
), target_marked AS (
    SELECT *,
           CASE WHEN prior_max_end IS NULL OR start > prior_max_end
                THEN 1 ELSE 0 END AS new_island
    FROM target_ordered
), target_grouped AS (
    SELECT *,
           SUM(new_island) OVER (
               PARTITION BY ordinal ORDER BY start, end
               ROWS UNBOUNDED PRECEDING
           ) AS island
    FROM target_marked
), target_merged AS (
    SELECT ordinal, island, MIN(start) AS start, MAX(end) AS end
    FROM target_grouped
    GROUP BY ordinal, island
), target_per_ordinal AS (
    SELECT ordinal, SUM(end - start) AS target_union_ns
    FROM target_merged
    GROUP BY ordinal
), overlap_parts AS MATERIALIZED (
    SELECT t.ordinal,
           MAX(t.start, x.start) AS start,
           MIN(t.end, x.end) AS end
    FROM target AS t
    JOIN excluded_intervals AS x
      ON x.ordinal = t.ordinal
     AND x.start < t.end
     AND x.end > t.start
), overlap_ordered AS (
    SELECT ordinal, start, end,
           MAX(end) OVER (
               PARTITION BY ordinal ORDER BY start, end
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
           ) AS prior_max_end
    FROM overlap_parts
), overlap_marked AS (
    SELECT *,
           CASE WHEN prior_max_end IS NULL OR start > prior_max_end
                THEN 1 ELSE 0 END AS new_island
    FROM overlap_ordered
), overlap_grouped AS (
    SELECT *,
           SUM(new_island) OVER (
               PARTITION BY ordinal ORDER BY start, end
               ROWS UNBOUNDED PRECEDING
           ) AS island
    FROM overlap_marked
), overlap_merged AS (
    SELECT ordinal, island, MIN(start) AS start, MAX(end) AS end
    FROM overlap_grouped
    GROUP BY ordinal, island
), overlap_per_ordinal AS (
    SELECT ordinal, SUM(end - start) AS excluded_union_ns
    FROM overlap_merged
    GROUP BY ordinal
)
SELECT t.ordinal,
       t.target_union_ns,
       COALESCE(x.excluded_union_ns, 0) AS excluded_union_ns,
       t.target_union_ns - COALESCE(x.excluded_union_ns, 0) AS exposed_ns
FROM target_per_ordinal AS t
LEFT JOIN overlap_per_ordinal AS x USING (ordinal)
ORDER BY t.ordinal;

-- Preserve all 100 raw rows before interpreting the median.
SELECT ordinal, target_union_ns, excluded_union_ns, exposed_ns
FROM o077_interval_result
ORDER BY ordinal;

-- Linear p50 for even or odd sample counts.  For 100 values this averages the
-- 50th and 51st sorted values, matching the retained report convention.
WITH target_ranked AS (
    SELECT target_union_ns AS value,
           ROW_NUMBER() OVER (ORDER BY target_union_ns) AS rn,
           COUNT(*) OVER () AS n
    FROM o077_interval_result
), exposed_ranked AS (
    SELECT exposed_ns AS value,
           ROW_NUMBER() OVER (ORDER BY exposed_ns) AS rn,
           COUNT(*) OVER () AS n
    FROM o077_interval_result
)
SELECT (SELECT AVG(value) FROM target_ranked
        WHERE rn IN ((n + 1) / 2, (n + 2) / 2)) AS target_union_p50_ns,
       (SELECT AVG(value) FROM exposed_ranked
        WHERE rn IN ((n + 1) / 2, (n + 2) / 2)) AS exposed_p50_ns;

-- Expected post-O077 source result:
--   launch audit = 800,100,devices 0..7,context 1,stream 26,
--                  grid 256,block 32,REG74,dynamic shared 14432
--   target_union_p50_ns = 154975
--   exposed_p50_ns      = 140806
-- Expected post-O077 return result after replacing o077_params with the return
-- row above:
--   target_union_p50_ns = 64449.5 (displayed as 64.450 us)
--   exposed_p50_ns      = 56470
