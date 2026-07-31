# Hop-aware RailBalance Leaderboard

更新时间：2026-08-01 UTC。

本表从 sparse multi-target checkpoint 开始整理；更早的逐候选事实仍以
[`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md) 和
[`HOP_AWARE_PERFORMANCE_EVIDENCE.md`](HOP_AWARE_PERFORMANCE_EVIDENCE.md) 为准。
不同 workload、dirty/clean、profiler/profiler-free 数字绝不混排。

## 1. 当前 Leader

| 角色 | 版本 | 完全正确 | clean | matched profiler-free | 结论 |
|---|---|---:|---:|---:|---|
| 最后一个已接受的 clean checkpoint | `8c56939` | 是 | 是 | 是 | C100 checked-adapter 10+100：one-hop 24.001 ms、adaptive 23.116 ms；仅 `checked_adapter_only`，不是 NIC/RDMA |
| 当前功能候选 | `286da0a`（证据/harness与控制面续至 `448a727`） | 20/20 planner；G2/tiny-tail 四 sanitizer；CPU/API/codegen全过 | 核心实现是 | 否 | 尚不能替换 clean Leader；等待 4×2 vnode重跑与 formal `4+4N` source round |

当前没有新的性能 Leader。代码更新、复杂度或较新的 dirty 诊断都不能自动晋升。

## 2. `volume` 同 workload 诊断

| 候选 | Parent | 单一主要假设 | 结果 | Gate | 结论 |
|---|---|---|---:|---|---|
| dense multi-target scan | `ee80a16` | 聚合 target mask 后仍扫描 dense padded group space | ~1.586 s finish | dirty diagnostic | 否定：dense space不可接受 |
| active-group sparse list | dense scan | 只访问真实出现的 `(owner,destination,mask)` | ~30.08 ms | dirty diagnostic | 假设成立，继续作为 parent |
| tied-hot fair batch | active list | 同一轮公平批量处理并列热点 | ~12.05 ms | dirty diagnostic | 假设成立，但需 singleton 反例 |
| bounded `G*32` rounds | fair batch | 用有限轮数和最小 batch封住逐份尾部 | ~5.87 ms | dirty diagnostic | 当前 sparse 主路径；未形成 clean claim |
| bounded G tiny tail | bounded rounds | cap 是上限；最多 G 个小尾轮处理 quota-one | ~6.09 ms | dirty diagnostic；20/20 current correctness | 修复真实语义边，不按微小 timing 判性能 Leader |

这些值来自相邻 dirty tree，不是正式 before/after，也没有越过测量噪声判定。

## 3. `singleton rot1` 反例与失败候选

| 候选 | Parent | 假设 | 结果 | 正确性 | Verdict / 失败原因 |
|---|---|---|---:|---:|---|
| fair batch原规则 | active list | volume batching可泛化到 singleton | ~500.27 ms | 通过 | 否定泛化性；残余循环近8,000次 |
| remove batch-one gap | fair batch | 去掉一个 `gap<=0` 分支可消除长尾 | ~500 ms | 通过 | 无提升，已撤回 |
| chunk size作为最小 adaptive batch | bounded design | materialize粒度可限制adaptive movement | 未测速 | 失败 | capacity-edge合法计划被拒绝，已撤回 |
| bounded `G*32` rounds | failed chunk floor | 有限总轮数可封住 singleton | ~6.05 ms | 通过 | 成立，成为 tiny-tail parent |
| unrestricted small fallback | bounded rounds | 任意小 batch继续搜索会改善质量 | 20.77 ms；peak 7,170 | 通过 | 较 6.05 ms大幅回退，否定 |
| at-most-G tiny tail | bounded rounds | 少量尾轮保留必要 quota-one改进 | 7.24 ms；peak 7,194 | 通过 | 接受语义修复；timing仍是dirty诊断 |

## 4. 基础设施/证据候选

| ID | Parent | 修改/假设 | Compile | Correctness | Benchmark/Profile | 结论 |
|---|---|---|---|---|---|---|
| `vnode-g2-parameterization` | current dirty | 用2×2 vnode在少卡验证数据面 | Python compile PASS；runtime prepare拒绝 | FAIL before data movement | 未运行 | `kWorldRanks=8`固定契约；代码已撤回，不能把planner G2测试冒充vnode |
| `takeover-scaffold-20260801-01` | historical dirty | 无8卡时仍能完成CPU证据与命令冻结且GPU启动数为0 | GPU compile skipped | CPU 6/6 PASS；GPU attempts 0 | not measured | legacy pre-terminal-contract：缺 `FINALIZED.json`，formal evaluator必须拒绝；仅保留 provenance |
| `terminal-smoke-20260801-01` | terminal-schema smoke | 验证终端 record/hash链 | 全部 stage skipped | `--no-execute`，CPU gate未运行 | not measured | 有 `FINALIZED.json`，但 `round_evaluation_allowed=false`；不能替代 fresh scaffold |
| `448a727-control-plane` | `0f38688` | 整轮 lease、路径/hash/时序绑定可防错误晋升 | Ruff、py_compile PASS | supervisor 20/20；coordinator 9/9；evaluator 19/19；executor 16/16 | not measured；GPU attempts 0 | 形式化执行控制面通过 CPU审计；没有 source manifest/live raw round，不改变性能 Leader |
| `takeover-scaffold-20260801-02` | clean `be4a69b` | finalized合同在无独占八卡时保留CPU证据且启动0个GPU进程 | 19个exclusive-8GPU stage skipped | CPU 6/6 PASS；GPU attempts/starts 0/0；SHA tree verified | not measured | 合规 clean scaffold；`round_evaluation_allowed=false`，不能晋升 |
| `source-round-preparer` | authoring commit atop `be4a69b` | 从human spec和clean sibling worktree客观冻结manifest/plan，消除手抄Git身份 | Ruff、py_compile PASS | preparer 10/10；coordinator 9/9；executor 16/16；evaluator 19/19 | not measured；无GPU启动 | 只完成authoring控制面；输出恒为`PREPARED_NOT_RUN`，没有候选或性能结论 |

运行时所有新候选（包括 compile/correctness失败）追加到：

```text
.cache/rail_balance/hop_campaign/leaderboard.jsonl
```

正式优化轮必须从“最快、完全正确、clean、matched profiler-free”的 Leader 出发，声明
2～4 个单变量候选。下一轮假设只能来自上一轮 raw distribution 与 Nsys；没有 Nsys
归因时，不启动随机 CUDA 改写。
