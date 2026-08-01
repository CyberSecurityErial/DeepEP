# RailBalance 离线自动调优手册（v1）

## 1. 适用范围

本手册以黄金提交 `4e00276` 为生产基线。该提交必须是当前 source commit 的祖先；
当前提交可以包含本工具/测试变更。`plan` 会记录 `source_commit/source_clean`，脏树
计划只可诊断，`freeze` 会直接拒绝。v1 不修改生产 CUDA、生产 Python API 或
DeepEP V2 数据路径。

因此，执行 `plan` 或 `freeze` 都不会启用 RailBalance，也不会改变现有任务的
默认行为。冻结得到的 profile 当前**不会被生产代码自动读取或消费**；它只是可审计的
候选配置与证据记录。以后若要接入生产，必须另做显式实现、评审、正确性门禁和真机回归。

所有校验都采用 fail-closed：基线、schema、trace、容量、报告或性能门槛不满足时直接
报错并返回非零状态，不猜值、不静默降级，也不生成“看似可用”的 profile。

## 2. 三层旋钮

| 层 | v1 旋钮 | 规则 |
|---|---|---|
| 算法 | `mode`、2-hop threshold/cap、hop penalty | 只在离线 tuning trace 上枚举；planner seed/chunk 是冻结身份参数 |
| 执行 | DeepEP V2 auto，或有证据的显式 SM/QP | 默认 preset 为零值 auto，复用 V2 自带估算；不得凭经验手填 |
| 容量 | `proxy_slots_per_rank` | 必须覆盖候选的 moved-copy 峰值并满足内存预算；不足即淘汰或报错 |

v1 固定 `planner_seed=0`、`planner_chunk_size=8`，两者都进入 candidate identity 和
配置一致性校验，但不是搜索轴。chunk 旋钮仅为后续版本预留，当前不扫。

`one_hop` 不允许第三 rail，工具会把与 2-hop 有关的 threshold、cap 和 penalty
规范化为零，避免同一语义产生多个候选 ID。`adaptive` 才搜索受限的第三-rail 参数。
参数名中的“percent”是当前整数启发式的输入，不应直接解释成物理带宽百分比。

执行层默认复用 DeepEP V2：profile 中 `num_sms=0, num_allocated_qps=0` 分别表示
自动估算 SM 和自动分配 QP；benchmark 的 dispatch/combine 继续传 `num_qps=0`，按
已解析的 SM 数使用 V2 QP 估算，combine 也复用 dispatch 的 SM 配置。
只有 Nsight Systems 证明 SM/QP 相关工作在端到端关键路径上有可观测的暴露时间后，
才允许把显式值加入搜索空间。Nsys 只用于形成假设，最终取舍仍以无 profiler 的
steady-state 原始样本为准。

CPU 在 v1 仅承担离线 reference/oracle、候选剪枝和 Pareto 分析。它不接收线上逐轮
top-k，不替代当前逐轮 GPU planner，也不引入 `GPU -> CPU -> GPU` 同步链。

容量层在 space 中可给正整数，也可设为 `null` 复用 benchmark 的确定性自动容量。
显式值不足时 CPU oracle 直接拒绝候选；自动容量必须在每份多机结果中解析为相同的
正整数，freeze 会把它写成 `resolved_proxy_slots_per_rank`，而不是留下不受审计的空值。

## 3. 准备真实 tuning trace

调优输入应是业务产生的连续、token 级路由 trace，至少能还原 owner、目标 rank/mask、
top-k、expert placement、shape 和窗口顺序。记录 trace hash，并在开始搜索前固定：

- `4e00276` 祖先门禁、当前 source/build identity 与干净工作树状态；
- 节点数、每节点 GPU、GPU/NIC/Rail 拓扑及可见设备顺序；
- 模型/层、token 数、hidden、top-k、expert placement、dtype；
- driver、CUDA、PyTorch、NCCL/NVSHMEM、DeepEP extension 版本；
- tuning/held-out 切分、warmup、steady iterations、超时和性能门槛。

不要用 held-out trace 反复选参数。合成 `diag_hot`、`closed_block` 等只适合正确性和
压力 smoke；聚合 endpoint count 若不能还原 token 级共现，也不能替代真实 trace。
工具能校验 trace schema 与 hash，但不能替操作者证明数据来源。缺少合格真实 trace 时
应停止正式调优；合成或聚合 trace 的输出只保留为 diagnostic，不能包装成生产结论。

## 4. 生成候选计划

先查看当前 CLI 合约：

```bash
python tools/rail_balance_autotune.py plan --help
```

使用固定的 v1 搜索空间生成计划：

```bash
python tools/rail_balance_autotune.py plan \
  --space tests/elastic/rail_balance_autotune_space_v1.json \
  --case trace --workload-json <TUNING_TRACE.json> \
  --num-nodes <NUM_NODES> --gpus-per-node <GPUS_PER_NODE> \
  --tokens-per-rank <TOKENS> --topk <TOPK> \
  --num-experts <EXPERTS> --hidden <HIDDEN> \
  --warmup-iters <WARMUP> --steady-iters <STEADY> \
  --results-dir <RUN_ROOT>/results \
  --output <RUN_ROOT>/autotune-plan.json
```

`--workload-json` 只在 `--case trace` 时使用；正式调优应使用这一路径。其他 case
仅用于工具合约、正确性或压力 smoke，不能替代真实 trace 结论。

`plan` 是离线步骤，不启动生产 CUDA。它应当完成以下工作：

1. 校验 schema、黄金提交祖先关系、trace/拓扑 identity、固定 seed 和参数边界，
   并记录 source clean 状态；
2. 确定性生成并规范化算法、执行、容量三层候选；
3. 用 CPU oracle 计算 load objective、检查 proxy 容量并计算额外 local-hop 代价；
4. 淘汰非法、超容量或被支配候选，保留 Pareto 候选；
5. 为每个候选写入稳定 ID、完整参数、输入 hash、oracle 指标和真机执行 `env`/`argv`。

离线 Pareto 至少同时观察最坏 `pair_peak` 与 `extra_local_hop_bytes`；前者下降不代表
端到端一定加速，后者、planner、同步和尾延迟都可能抵消收益。保留完整 plan，不能只
抄出“最好”的一行；真机失败和被淘汰候选同样是证据链的一部分。

## 5. 多机执行候选

从 plan 中复制候选记录给出的 `env` 和 `argv`，不手工重拼参数。所有节点必须使用
同一个 plan、candidate ID、trace/window、build、环境和输出目录。每个节点各启动一次：

```bash
CUDA_VISIBLE_DEVICES=<LOCAL_GPU_LIST> \
WORLD_SIZE=<NUM_NODES> RANK=<NODE_RANK> \
MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> \
OMP_NUM_THREADS=1 EP_DISABLE_GIN=0 \
PYTHONPATH="$PWD/tests/elastic:$PWD" \
<PLAN 中该候选的其余 env> \
<PLAN 中该候选的 argv>
```

这里 `WORLD_SIZE` 是节点数，`RANK` 是节点序号。统一 runner 会自行 spawn 本节点的
GPU worker；不要再包一层 `torchrun --nproc-per-node=...`，否则会重复创建 worker，
并破坏 node-rank/world 语义。

plan 中的 `result_path` 是单次运行默认路径。每次完整进程运行结束后，将结果归档为
不同的只读文件（或只替换 argv 的 `--output` 路径）；其余 `env`/`argv` 不得改变。

正式冻结至少保留五次独立、无 profiler 的 `real_multinode` 合法结果。运行要求：

- 同一真实 trace/window 上配对执行基线与候选，并保存所有 rank 的原始样本；
- 分开 cold start 与 steady state，报告 rank-max latency/throughput 的 median、p95、
  p99、mean 和标准差，而不是只保留最好一次；
- 验证 dispatch/combine 数值、payload 守恒、目标 expert/mask 不变且无 drop/duplicate；
- 保存完整命令、退出码、环境、source/build/trace/config hash 和容量水位；
- 任一 rank 超时、OOM、配置不一致或正确性失败时，整次运行失败，禁止局部 PASS。

若要研究显式 SM/QP，Nsys 必须另跑，不得把 profiler 运行的 wall time 混入冻结结果。
先证明相关 kernel、launch、通信或同步具有暴露关键路径时间，再一次只改变一个执行
变量，最后回到同一无 profiler 窗口做配对验证。没有这条证据时保持 `v2_auto`。

## 6. 选择并冻结 profile

只从 plan 保留的 Pareto `candidates` 中选择一个 ID；CPU plan 本身不具备性能声明资格。
选择依据应同时包含离线 Pareto、多机正确性、无 profiler 的端到端收益、p95/p99、
容量和额外 local-hop 代价；不得在 held-out 结果出来后改阈值或逐 workload 挑不同最优值。

`--minimum-speedup` 必须大于 `1`，并在看候选结果前根据 A/A 噪声界和业务最小收益
预注册。工具不提供一个能替代测量合同的默认阈值。

把至少五份独立 `real_multinode` 结果交给 `freeze`：

```bash
python tools/rail_balance_autotune.py freeze \
  --plan <RUN_ROOT>/autotune-plan.json \
  --candidate-id <CANDIDATE_ID> \
  --result <RUN_ROOT>/result-01.json \
  --result <RUN_ROOT>/result-02.json \
  --result <RUN_ROOT>/result-03.json \
  --result <RUN_ROOT>/result-04.json \
  --result <RUN_ROOT>/result-05.json \
  --minimum-speedup <PREREGISTERED_THRESHOLD> \
  --minimum-runs 5 \
  --output <RUN_ROOT>/rail-balance-profile.json
```

`freeze` 只做验证和封存，不重新搜索。它应核对 candidate/plan identity、五份结果的
`real_multinode` 资格、结果 commit 与 plan source commit 一致、配置一致性、正确性、
容量（包括自动值的唯一解析结果）和预注册 speedup 门槛，然后写入完整
算法/执行/容量配置、证据引用与 hash。任一条件不满足就直接报错，不输出 profile。

冻结文件中的 `production_consumed=false` 是重要安全边界：当前生产 DeepEP 不会自动
发现、加载或应用它，生成该文件对默认路径是零影响。不要删除或改成 `true` 来“启用”
配置；真正接入生产必须是后续独立变更，并继续保留 V2 auto、显式 opt-in 和失败直报。

## 7. 快速检查清单

- [ ] 工作树干净，`4e00276` 是当前 source commit 的祖先，输入和环境都有 hash。
- [ ] 使用真实 tuning trace；held-out 未参与候选选择。
- [ ] `planner_seed=0`、`planner_chunk_size=8`，`one_hop` 的 2-hop 参数已规范化。
- [ ] 默认执行 preset 是 `v2_auto`：profile 两个资源值为零，调用保持 `num_qps=0`。
- [ ] 显式 SM/QP 有 Nsys 关键路径证据，并通过无 profiler 回归。
- [ ] proxy slots 覆盖实测峰值；自动容量已一致解析，未依赖 fallback。
- [ ] 每节点只启动一次，至少有五份独立进程的 `real_multinode` 合法结果。
- [ ] freeze 的 candidate ID、配置和结果完全一致。
- [ ] profile 保持 `production_consumed=false`，生产路径未被修改。
