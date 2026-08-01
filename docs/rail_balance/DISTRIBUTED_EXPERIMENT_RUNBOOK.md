# RailBalance 分布式实验执行手册

> 版本：2026-08-01 / v0.1  
> 用途：交给集群上的 Codex，按本文完成实验、留存证据并返回可审计结果。  
> 状态：这是实验方案，不包含任何性能结果。命令里的 `<...>` 必须先解析并写入 manifest；不得原样执行。  
> 黄金回归基线：本仓库 `e5d49d328c85aa210e170a0587a7cb318df0dfb4`。候选必须是它之上的同树 RailBalance 变更；第三 rail 的直接因果对照仍是候选内的 `one_hop`。

## 0. 最终要回答的问题

本轮只回答三个问题：

1. 在真实 MoE 路由和真实多机拓扑上，允许第三 rail 的 `adaptive` 是否比只允许一跳候选的 `one_hop` 有稳定净收益？
2. planner/plan materialization 的耗时有多少被并行隐藏，最终暴露在 dispatch 或模型关键路径上的时间是多少？
3. 在相同输入、阶段边界和资源约束下，当前 DeepEP V2、RailBalance、NCCL EP、UCCL-EP 分别处于什么位置；UltraEP 与 RailBalance 组合后是否互补？

`off -> adaptive` 只能说明 RailBalance 整体是否有用，不能证明“第三 rail”有用。第三 rail 的主对照必须是同一真实 trace 上直接配对的 `one_hop -> adaptive`。

## 1. 不可妥协的证据规则

1. **真实 trace 是主证据。** `diag_hot`、`closed_block`、Zipf、手工 rail-hot 等合成输入只用于正确性、机制压力测试和负控，不得支持业务必要性或泛化性结论。
2. **不把聚合端点计数冒充真实路由。** 当前 `trace` schema v1 只保存 `(src_node, src_local_rank, dst_node, dst_local_rank, count)`，loader 会把每条记录展开为目标 rank 的前 K 个 expert；它无法表达 token 级 top-k 共现、多目标 mask、连续 microbatch、实际 placement、capacity/drop。v1 结果只能标为 `SMOKE_ONLY`。
3. **同树因果比较。** RailBalance 的 `off`、`one_hop`、`adaptive` 必须来自同一候选 worktree、同一二进制和同一环境。黄金提交 `e5d49d...` 是加入 hop-aware/第三 rail 之前的黄金回归基线；不用跨提交的其他差异冒充第三 rail 因果效应。
4. **测量真值不带 profiler。** 稳态结论使用未包裹 profiler 的 CUDA event/consumer dependency 和分布式 rank-max；Nsight Systems/Compute 只做归因。
5. **共同公平与作者复现分轨。** common-fair 使用共同数据类型、输入 tensor、阶段边界、统计方法和资源账本；author-compatible 保留各项目原生参数和计时方法。两轨结果不得混表排序。
6. **缺证据就显式失败。** 缺真实 trace、适配器、runtime path counter 或物理链路 counter 时分别记为 `TRACE_GAP`、`API_GAP`、`COUNTER_GAP`，不能填 0、估算或用 Python oracle 代替。
7. **先冻结再测。** source SHA、镜像、编译参数、拓扑、trace/split hash、参数搜索空间、排除规则、统计脚本必须在 confirmatory run 之前冻结。
8. **不改竞争者语义。** 允许增加薄适配器、原始样本导出、NVTX 和正确性检查；任何算法、buffer layout、路由或同步语义修改都会使该行变成“port/fork”，不能再称作者版本。

## 2. 一手来源与冻结版本

| 对象 | 冻结来源 | 在本手册中的角色 | 边界 |
|---|---|---|---|
| DeepEP V2 | [deepseek-ai/DeepEP@dd758caf](https://github.com/deepseek-ai/DeepEP/tree/dd758caf451848bd150e1046af3d0a73e5fff38d) | clean-upstream 外部基线 | Rail 因果基线仍是同树 `off` |
| NCCL EP | [论文 v3](https://arxiv.org/html/2603.13606v3)、[NCCL tag `nccl-ep-v0.1.0` / 63cf786b](https://github.com/NVIDIA/nccl/tree/63cf786b015b2b6bff6cf263461621acf584bd18/contrib/nccl_ep) | 直接通信库对手 | 论文 HT 评测仍标为 ongoing；不得编造论文 HT 数字 |
| UCCL-EP | [论文 v2](https://arxiv.org/html/2512.19849v2)、[uccl@61ee4240/ep](https://github.com/uccl-project/uccl/tree/61ee42402819cabba3ac2a56dd4addec3363976c/ep) | 直接通信库对手 | CPU proxy 是架构资源，必须计入 CPU/NUMA 账本 |
| UltraEP | [论文 v3](https://arxiv.org/html/2606.04101v3)、[UltraEP@94cab099](https://github.com/Dots-Infra/UltraEP/tree/94cab099b44fffa99a82fea99e7c12d89cf65e4f) | placement/replication/reroute 与 Rail 的组合实验 | 不是单纯 transport 替代；公开 testbed 未披露 GPU/NIC 型号，普通 H100/H200 集群不能称精确硬件复现 |
| MoonEP | [MoonEP@0f385f03](https://github.com/MoonshotAI/MoonEP/tree/0f385f038fc33bec22e3bcf5a07a8a22693e754c) | 单机 placement/layout 附录 | 公开范围是 H20、EP8；不能作为多机 RDMA 结论 |

更长的一手事实审计见 [COMPETITOR_EVIDENCE_2026-08-01.md](./COMPETITOR_EVIDENCE_2026-08-01.md)，论文实验映射见 [PAPER_EXPERIMENT_MATRIX_2026-08-01.md](./PAPER_EXPERIMENT_MATRIX_2026-08-01.md)。若两者与本文冲突，以本文对“真实 trace 主证据”和 `one_hop -> adaptive` 直接配对的要求为准。

### 2.1 总实验矩阵

| ID | 优先级 | 输入 | 处理/对手 | 主要回答 |
|---|---|---|---|---|
| D0 | 必须 | 集群拓扑与空载窗口 | 所有 frozen source/build | 环境是否可比，资源与物理 rail 是否已证明 |
| D1 | 必须 | 合成 balanced/edge/stress | `off/legacy_exact/one_hop/adaptive` | 正确性、守恒、失败闭合；不作业务结论 |
| D2 | 必须 | held-out 真实 route trace v2 | exact `P1*`、`Pall*`、当前 `P1` | 第三 rail 是否有结构可行域价值，还是 one-hop greedy 缺陷 |
| D3 | 必须 | 同一 held-out 连续真实 trace | direct paired `one_hop/adaptive`；`off` 辅助 | 第三 rail 完整净收益、planner 暴露时间、物理路径闭环 |
| D4 | 适配器通过后必须 | LL/HT 共同 BF16 真实 trace | same-tree DeepEP off、Rail、clean upstream DeepEP、NCCL EP、UCCL-EP | 相同边界/资源下的直接 transport 比较 |
| D5 | 真实模型栈可用时必须 | 同一模型/checkpoint/真实 request 或训练 batch | D4 中可接入同一框架的后端 | serving goodput/tail latency 或 training step，不用 microbench 推算 |
| D6 | 条件必须 | UltraEP 论文模型族或目标真实 workload | placement off/on x `one_hop/adaptive` | UltraEP 与 Rail 是互补、替代还是有负 interaction |
| D7 | 附录 | H20 EP8 原生或明确的 port | MoonEP vs DeepEP V2 | 单机 planning/layout；不外推多机 RDMA |

每一行都要保留 `PASS/FAIL/NOT_RUN/UNSUPPORTED/*_GAP`；不能因为没有 adapter 或硬件就从矩阵消失。

## 3. 集群 Codex 的执行合同

### 3.1 允许做的事

- 创建隔离 worktree、conda/venv、容器和只写实验产物目录。
- 在测试/benchmark 层增加 trace v2 reader、公共适配器、原始样本导出、runtime counter、NVTX 和统计脚本。
- 为公共框架增加后端胶水，但必须把胶水单独提交，并证明没有改动库的路由/传输语义。
- 在每个阶段失败后继续执行不依赖该阶段的项目，并保留失败日志。

### 3.2 不允许做的事

- 为了让结果更好而修改生产 Rail 算法、竞争库算法、模型路由或真实 trace。
- 把不支持的 dtype/API 静默改成另一个配置。
- 在 confirmatory split 上选择参数、删异常点或决定分层边界。
- 在共享源码 checkout 中原地切 SHA；冻结版本必须使用独立 worktree/clone。
- 用平均 rank 延迟替代 rank-max，用 kernel 时间相加替代端到端事件，或把 profiler run 填入性能主表。

### 3.3 统一状态码

每个实验单元只能是：

- `PASS`：正确性与证据门禁全部通过。
- `FAIL`：运行完成但正确性、完整性或预注册阈值失败。
- `NOT_RUN`：尚未执行。
- `UNSUPPORTED`：冻结版本明确不支持硬件、dtype 或拓扑。
- `API_GAP`：没有语义等价的公共测量边界/trace 注入接口。
- `TRACE_GAP`：没有合格真实 trace。
- `COUNTER_GAP`：不能验证物理路径。
- `ENV_REJECTED`：拓扑、后台流量、温度、时钟或软件栈不满足门禁。

`UNSUPPORTED`/`*_GAP` 是结果，不是把该格从报告中删除的理由。

## 4. 产物目录与不可变 manifest

在共享且容量足够的绝对路径创建：

```text
<RUN_ROOT>/
  00_manifest/        campaign.json, preregistration.md, SHA256SUMS
  01_sources/         source_manifest.json, patches/, build_logs/
  02_environment/     per_node/, topology.json, resource_budget.json
  03_traces/          raw/, v2/, splits/, validation/
  04_correctness/     synthetic/, real_trace/
  05_tuning/          offline_flow/, microbench/, frozen_candidate.json
  06_confirmatory/    rail_necessity/, planner_exposure/
  07_common_fair/     deepep/, nccl_ep/, uccl_ep/
  08_author_track/    deepep/, nccl_ep/, uccl_ep/, ultraep/, moonep/
  09_end_to_end/      serving/, training/, composition/
  10_profiles/        nsys/, ncu/
  11_counters/        background/, software/, nic/, nvlink/
  12_analysis/        raw_index.parquet, figures/, tables/, report.md
  STATUS.json
```

`campaign.json` 至少包含：UTC 时间、操作者、集群/队列、所有 source SHA 与 dirty 状态、容器 digest、编译命令、GPU/NIC/CPU/NUMA 拓扑、固件/driver/CUDA/PyTorch/NCCL/NVSHMEM/UCX/MPI 版本、所有相关环境变量白名单、模型/checkpoint/tokenizer/dataset/request manifest hash、trace/split hash、随机种子、命令、退出码和状态码。

每轮结束重新生成 `SHA256SUMS`。原始 JSON/CSV/trace/日志只追加，不覆盖；派生表必须能从原始样本一条命令重建。

## 5. Stage 0：环境、源码和拓扑门禁

### 5.1 每个节点采集

以下是采集项，不假定所有命令都存在。不存在时记录 `command_not_found`，不要安装一个相似工具后假装是原环境。

```bash
date -u --iso-8601=seconds
hostname -f
uname -a
nvidia-smi -L
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit --format=csv
nvidia-smi topo -m
<ENV_PYTHON> -VV
<ENV_PYTHON> -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version())'
ibdev2netdev
ibv_devinfo -v
rdma link show
lspci -Dnn
nsys --version
ncu --version
mpirun --version
```

只采集与实验有关的环境变量，禁止 dump 全量环境或凭据：

```bash
for key in CUDA_VISIBLE_DEVICES MASTER_ADDR MASTER_PORT WORLD_SIZE RANK \
  NCCL_DEBUG NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_GIN_TYPE \
  NVSHMEM_HCA_LIST NVSHMEM_IB_GID_INDEX NVSHMEM_DISABLE_NCCL \
  UCCL_SOCKET_IFNAME UCCL_IB_HCA UCCL_IB_GID_INDEX \
  OMP_NUM_THREADS EP_DISABLE_GIN EP_JIT_CACHE_DIR; do
  printenv "$key" 2>/dev/null || true
done
```

额外生成 GPU -> NVLink peer -> NIC/rail -> PCIe switch -> NUMA node -> CPU proxy core 的显式映射。不得仅以“8 rails”命名而不证明 GPU/NIC 对应关系。

### 5.2 拒绝条件

任一条件成立则性能 run 标为 `ENV_REJECTED`：

- 节点的 GPU/NIC 型号、链路速率、driver/firmware、库 SHA 或编译 flags 不一致。
- GPU 有未声明进程、MIG 状态不一致、ECC/Xid 错误、明显 thermal/power throttling。
- 预热前后的时钟/温度状态漂移超出预注册环境阈值。
- fabric 背景流量的空窗 counter 变化超过 confirmatory 测量流量的预注册比例。
- rank 到 GPU/NIC/NUMA 的绑定在不同 run 中变化。
- 一项所需服务依赖未固定，例如路由 daemon、CPU proxy、GID/SL/QP 配置。

环境阈值必须由空载采样和 A/A 预检在看 confirmatory 结果之前写入 `preregistration.md`。

### 5.3 源码冻结

至少准备这些独立、clean checkout：

```text
deepep_gold       e5d49d328c85aa210e170a0587a7cb318df0dfb4
deepep_candidate <candidate SHA, descendant of e5d49d>
deepep_upstream  dd758caf451848bd150e1046af3d0a73e5fff38d
nccl_ep          63cf786b015b2b6bff6cf263461621acf584bd18
uccl_ep          61ee42402819cabba3ac2a56dd4addec3363976c
ultraep          94cab099b44fffa99a82fea99e7c12d89cf65e4f
moonep           0f385f038fc33bec22e3bcf5a07a8a22693e754c
```

对每个 checkout 保存：

```bash
git rev-parse HEAD
git status --porcelain=v1
git submodule status --recursive
git diff --no-ext-diff --binary
```

dirty checkout 只能作为 port/fork，不能作为冻结作者版本。当前本机 `/home/chen/workspace/infra/MoonEP` 曾观察为 `51e64aa...`，集群实验不得因此省略对 `0f385f...` 的重新冻结。

不要为了“同一个 Python 环境”破坏作者依赖：冻结 DeepEP V2 README 所需的 Hopper/CUDA/PyTorch/NCCL 栈，NCCL EP tag 所需的 CUDA 13+/NCCL 2.29+/MPI 栈，以及 UCCL/UltraEP/MoonEP 各自环境为独立容器/venv。common-fair 要求相同硬件、输入与测量边界，不要求不兼容的库共用一个进程环境；但 driver、firmware、时钟策略和作业隔离必须相同，并逐环境保存 image digest/lockfile/build log。

Rail 候选若还需要在集群上做单变量 source round，优先复用现有 fail-closed freezer：

```bash
PYTHONPATH="$PWD/tests/elastic:$PWD" <PINNED_PYTHON> -B \
  tests/elastic/prepare_rail_balance_source_round.py \
  --spec <ABSOLUTE_HUMAN_SPEC_JSON> \
  --manifest-output <ABSOLUTE_ARTIFACT_ROOT>/source-round.json \
  --plan-output <ABSOLUTE_ARTIFACT_ROOT>/plan.json
```

它只会生成 `PREPARED_NOT_RUN` 的 immutable manifest/plan，不会替你创建 worktree、build 或运行 GPU。每轮使用新的权限受控 artifact root，并按 [HOP_AWARE_EXPERIMENT_PROTOCOL.md](./HOP_AWARE_EXPERIMENT_PROTOCOL.md) 的 source-round 合同执行；不得绕过 clean-tree、harness hash 和 GPU mapping 检查。

## 6. Stage 1：真实路由 trace v2

### 6.1 合格 trace 的来源

主证据至少覆盖两个真实模型族，并尽可能覆盖两类实际阶段：低 token/rank 的 decode/低延迟通信，以及高 token/rank 的 prefill 或训练吞吐。优先级如下：

1. 目标集群真实要服务/训练的 MoE job；只能采集路由元数据时就只采路由元数据，不复制用户 prompt。
2. 公开模型与真实数据集/请求集。可复用论文已有的真实锚点：
   - NCCL EP 端到端轨：vLLM 0.10、Qwen3-30B-A3B、1/2/4 节点、1000 requests、concurrency 32；缺作者请求集时只能称“配置复现”，不能称工作负载复现。
   - UCCL-EP 端到端轨：SGLang 0.5.3、Qwen3-235B-A22B-FP8 或 DeepSeek-R1-0528，EP16/EP32，真实 prompt、input 4096/output 5 的 prefill-heavy 配置。
   - UltraEP 泛化轨：论文使用的 Codeforces、SWE-bench、DAPO-Math、GPQA、OpenScience；长上下文混合再加入 LongBench。保留各数据集原始样本和 tokenizer 后的自然长度，不把它们改造成 rail-hot。
3. 若真实训练栈可用，采集同一真实训练 recipe 的至少两个 checkpoint/阶段，例如早期和晚期；不可用则标 `NOT_RUN`，不得用 Zipf 替代后声称训练泛化。

若只能得到一个模型/数据集，仍可完成窄范围实验，但最终结论必须限定到该模型/trace，不能写“普遍有效”。

### 6.2 v2 最小语义

trace 必须保留连续窗口，而不是只存 endpoint histogram。manifest 至少包括：

```text
schema_version = 2
capture: framework/repo/SHA/model/checkpoint/tokenizer/dataset/request hashes
topology: nodes, ranks, local-rank ordering, GPU/NIC map
semantics: layer, EP/DP/TP/PP, expert count, top-k, capacity/drop policy, dtype
placement: 每个 window/layer 的 expert -> physical rank 映射及副本状态
windows[]:
  sequence_id, microbatch_id, layer_id, timestamp/order
  each source rank:
    valid token count
    token-level topk expert IDs（完整 K 元组）
    topk weights（combine/correctness 需要时）
    valid/drop mask 和 capacity 结果
payload tensor 文件的 shape/dtype/SHA256
```

允许压缩、分片或只保存无内容 token ID，但必须能逐 token 重放相同的 top-k 共现和时间顺序。

### 6.3 现有 harness 的硬缺口

当前：

- `bench_rail_balance_hybrid_multinode.py --case trace` 读取 schema v1，并把聚合记录补成本地 token。
- 每个测量 block 重复同一 top-k；不能代表连续路由变化。
- 测量顺序固定为 `off` 与 `candidate` 的 ABBA/BAAB，不能直接配对 `one_hop` 与 `adaptive`。
- 报告的 path/rail 分布来自 `python_endpoint_oracle_expected`，不是物理执行计数。

**也就是说，当前真实多机 runner 中并不存在本文所需的 temporal trace-v2 replay、direct-paired A/B 或 runtime physical-path 证据。现有 common-fair compact-v2 序列化/设计 scaffold 不能仅凭版本号替代这些能力。**

因此集群 Codex 必须先在 benchmark/test 层补齐下列能力，再开始第三 rail 主实验：

1. v2 token-level 连续窗口 reader，严格校验 topology/placement/dtype/hash。
2. `--baseline-mode one_hop --candidate-mode adaptive`，在每个相邻 pair 中直接测二者。
3. 每个 steady iteration 消费预注册的 window 序列，而不是复制一张 top-k 100 次；循环时明确记录 epoch boundary。
4. planner submit、plan ready、dispatch ready、combine ready、consumer ready 的 CUDA event/NVTX 边界。
5. runtime 计数：direct/source-forward/destination-forward/third-rail payload、字节、local cross-GPU copy、fallback/reject。
6. 输出每次迭代的每-rank原始值和 rank-max，不只输出汇总。

这些改动必须是测量语义改动，单独提交，并通过以下测试：v2 round-trip hash、multi-target mask、跨节点 K 元组、placement change、drop mask、窗口顺序、`one_hop/adaptive` 相邻配对、runtime payload 守恒。若做不到，第三 rail 主问题状态为 `API_GAP`。

### 6.4 split 与防泄漏

按真实时间/请求单元先切分，后调参：

- `tuning`：最前 40%，只用于参数选择和自然压力分层边界。
- `confirmatory`：最后 60%，在冻结候选前不可查看性能结果。
- 同一 request/conversation/训练 batch 的窗口不能跨 split。
- 另按模型/数据集做 leave-one-workload-out 报告；不得把同一 trace 随机打散后称跨 workload 泛化。

若时间顺序本身有明显阶段变化，同时报告 chronological split 和整 workload 的 block bootstrap；不要偷偷改成更好看的随机 split。

## 7. Stage 2：正确性与机制压力测试

这一阶段允许合成 case，但只产生 correctness/mechanism 证据。

### 7.1 必测模式

- `off`
- `legacy_exact`
- `one_hop`
- `adaptive`，至少覆盖 `max_two_hop_percent = 0` 和冻结候选值

### 7.2 必测不变量

- dispatch token/weights/indices 与官方 CPU/PyTorch reference 一致。
- combine 输出无 NaN/Inf；BF16 同时报告 max absolute、max relative、L2 error，并使用在 run 前冻结的容差。
- 最终 expert、多播 target mask、scale-out payload 个数不因 egress 选择改变。
- runtime payload 守恒，无 duplicate/drop；fallback/reject 与计划一致。
- `max_two_hop_percent=0` 与 `one_hop` 的选路语义一致。
- 各模式 ticket/handle 恰好消费一次，异步依赖在 consumer stream 上正确。
- 节点/GPU/rank 失败时必须 fail closed，不能部分成功仍记 PASS。

### 7.3 当前多机 correctness smoke 命令

在每个节点各启动一次；这里的 `WORLD_SIZE` 是节点数，`RANK` 是节点序号，runner 自己生成本机 worker，**不要再包一层每 GPU 的 torchrun**：

```bash
cd <DEEPEP_CANDIDATE>
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
WORLD_SIZE=<NNODES> RANK=<NODE_RANK> \
MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> \
OMP_NUM_THREADS=1 EP_DISABLE_GIN=0 PYTHONPATH="$PWD/tests/elastic:$PWD" \
<DEEPEP_PYTHON> -B tests/elastic/run_rail_balance_hybrid_multinode.py \
  --run-id <RUN_ID> \
  --case balanced \
  --output <RUN_ROOT>/04_correctness/synthetic/<RUN_ID>.json \
  --num-processes 8 --num-tokens 128 --hidden 7168 \
  --num-topk 8 --num-experts 256 \
  --num-sms <FROZEN_SMS> --num-allocated-qps <FROZEN_QPS> \
  --proxy-slots-per-rank <FROZEN_SLOTS> \
  --rail-policy <FROZEN_RAIL_POLICY> \
  --rail-threshold-percent <FROZEN_RAIL_THRESHOLD> \
  --sl-idx <FROZEN_SL> --seed 106
```

依次执行 `balanced`、`offdiag_hot`、`diag_hot`、`closed_block` 和 capacity 边界；后四者只进入 synthetic 目录。这个现有入口只验证 `off/force`。现有 `bench_rail_balance_hybrid_multinode.py --candidate-mode {legacy_exact,one_hop,adaptive}` 会在计时外做对应模式的正确性 probe，可用于补齐当前 synthetic 模式矩阵；它仍只配对 `off/candidate`，不能满足第三 rail 主实验的 direct A/B 要求。

## 8. Stage 3：第三 rail 的离线可解性证据

在测性能前，先判断真实 trace 是否存在一跳候选集合的结构性瓶颈。**当前仓库不存在下述 exact max-flow/min-cut 工具或证书输出；这是集群 Codex 必须先实现并用小规模穷举对拍的 CPU-only 分析工具，不是现成 CLI，也不得改生产 planner。**

对每个真实 source node/window/layer，把 payload atom 记为 `i`，流量为 `w_i`，owner 为 `o_i`，目标本地 rank 集合为 `T_i`：

```text
one-hop allowed set A_i = {o_i} union T_i
unrestricted set        = all healthy egress rails
```

exact objective 必须匹配当前 planner 的 destination-egress `pair_load`、source-egress `source_load`、proxy capacity 和 local-forward 次级代价，不能只优化总 egress histogram。对固定目标建整数网络：source -> 实际不可分 planning unit -> `(destination, egress)` -> `egress` -> sink，分别对 pair 和 source 容量设界；用嵌套搜索得到与生产排序一致的 `(pair_peak, source_peak, local_forwards)` 最优 tuple，失败时保存 min-cut。若 chunk/异构 payload 使普通 max-flow 的整数性不再成立，必须改用零 optimality-gap 的整数规划并保存求解证书，不能把 fractional relaxation 称 exact。

必须区分三个量：

```text
P1*   = one-hop allowed set 下的 exact optimum
Pall* = topology-legal unrestricted allowed set 下的 exact optimum
P1    = 当前 one_hop planner 实际得到的 objective
```

三者都是 `(normalized pair peak, normalized source peak, local forwards)` 的 lexicographic tuple。`cut_gap` 取 `P1*` 与 `Pall*` 第一个不同的 load 分量之相对差；`implementation_gap` 同理比较 `P1` 与 `P1*`。同时逐项保存 gap，不能直接对 tuple 做除法。

每个窗口输出：

```text
P1*, Pall*, P1
pair_gap, source_gap, cut_gap
pair_implementation_gap, source_implementation_gap, implementation_gap
binding atom/egress/cut
predicted extra local-copy bytes for unrestricted optimum
```

pair 容量 `C_(d,e)` 与 source-egress 容量 `C_e` 必须来自拓扑与链路校准；若假设对称容量，必须把假设作为单独结果，并补非对称敏感性分析。求解器输入、版本、容差、解和证书全部保存。

解释规则：

- `P1* == Pall*`（仅允许 solver 精度与 token/chunk 离散误差）：该窗口没有“必须使用第三 rail 才能改善最坏 rail 负载”的结构证据。此时 adaptive 若赢而 `P1 > P1*`，收益属于当前 greedy/one-hop 实现缺口，应先优化 one-hop，不能计为第三 rail 的必要性。
- `implementation_gap > 0` 必须单独报告；不得把 one-hop planner 未达到它自己的可行域最优，包装成扩展可行域的收益。
- `cut_gap > 0`：只说明有改善 load objective 的机会，不等于端到端更快；额外 local copy、planner 和同步可能吃掉收益。
- 合成 `closed_block` 的正 gap 只能验证 solver/机制，不能支持真实必要性。

自然压力分层使用 tuning split 的 `cut_gap` 分位点冻结为 low/mid/high，再原样应用到 confirmatory split。不得看 confirmatory latency 后重新定义“高压窗口”。

## 9. Stage 4：候选调参与冻结

本机算子优化完成后，先把 production-intended planner/kernel SHA、默认 rail policy、SM/QP/slot 预算作为候选起点。只在 tuning split 调参。

推荐先在 CPU/reference 上完整筛选的有界敏感性空间（不是业务场景）：

```text
max_two_hop_percent       in {0, 1, 2, 5, 10, 25, 50, 100}
two_hop_threshold_percent in {0, 1, 2, 5, 10, 20}
```

先用 exact reference 在 tuning trace 上画 `load objective / extra copies / selected percent` 的 Pareto 面；真机只跑 `cap=0`、Pareto knee 和一个 aggressive 点，随后总真机有效配置仍限制在最多 30 个、每配置相同时间预算。非法或 OOM 配置也计入预算。优化目标必须在运行前冻结，例如 tuning split 的 rank-max roundtrip p50，同时以 p95、额外 local-copy bytes、planner exposed time 为约束。不能逐 workload 选择不同“最佳”参数后再合并。

`two_hop_threshold_percent` 是当前启发式 score 的整数参数，score 同时受 pair/source relief、added hops、chunk 和规模影响；不能把数值解释为物理带宽百分比。`hop_penalty_percent` 先由本机/集群实测 NIC 与 NVLink/PCIe copy 代价校准，再做小范围敏感性；默认 `50` 也不能解释成“50% 性能代价”。

把唯一候选写到 `frozen_candidate.json`，内容包括 source/build/trace-tuning hash、完整参数、资源、选择规则和所有被淘汰配置。之后 confirmatory 阶段禁止修改。

## 10. Stage 5：第三 rail confirmatory 实验

### 10.1 处理组

同一进程运行、同一连续真实窗口、同一资源下直接比较：

| 标签 | 模式 | 含义 |
|---|---|---|
| A | `one_hop` | egress 仅在 `{owner} union targets` |
| B | `adaptive` | 冻结阈值/penalty/cap，允许 bounded third rail |

辅助组 `off` 用于回答 RailBalance 整体收益；`adaptive(max_two_hop=0)` 用于实现一致性负控。A/B 才是第三 rail 的主效应。

### 10.2 顺序与重复

- 先做 A/A（`one_hop` 对 `one_hop`）估计测量噪声。
- A/B 使用 ABBA 和 BAAB 相邻 block；每个 block warmup >= 10，steady >= 100。
- 至少 5 次独立进程重启；每次重启包含至少一组 ABBA+BAAB。
- 冷启动（JIT、buffer init、首次连接、首次 plan）另测至少 3 次，不与稳态合并。
- 先在 2x8 GPU，再在 4x8 GPU；8 节点只有资源足够且前两档通过后才做。
- GPU/rank/rail binding、request/window 顺序和随机种子在配对内完全一致。

当前 runner 可以做 `off -> one_hop` 或 `off -> adaptive` smoke，但**不能**用两个独立 job 相除得到 A/B 主效应。补齐 Stage 1 harness 后，等价命令必须具备以下参数；实际文件名和 `--help` 输出需保存，不能假装下面 CLI 已存在：

```bash
cd <DEEPEP_CANDIDATE>
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
WORLD_SIZE=<NNODES> RANK=<NODE_RANK> \
MASTER_ADDR=<MASTER_ADDR> MASTER_PORT=<MASTER_PORT> \
OMP_NUM_THREADS=1 EP_DISABLE_GIN=0 PYTHONPATH="$PWD/tests/elastic:$PWD" \
<DEEPEP_PYTHON> -B <TRACE_V2_MULTINODE_RUNNER> \
  --run-id <RUN_ID> --trace-manifest <TRACE_V2_CONFIRM_MANIFEST> \
  --baseline-mode one_hop --candidate-mode adaptive \
  --output <RUN_ROOT>/06_confirmatory/rail_necessity/<RUN_ID>.json \
  --num-processes 8 --hidden 7168 --num-topk 8 --num-experts 256 \
  --num-sms <FROZEN_SMS> --num-allocated-qps <FROZEN_QPS> \
  --proxy-slots-per-rank <FROZEN_SLOTS> \
  --two-hop-threshold-percent <FROZEN_TWO_HOP_THRESHOLD> \
  --max-two-hop-percent <FROZEN_MAX_TWO_HOP> \
  --hop-penalty-percent <FROZEN_HOP_PENALTY> \
  --warmup-iters 10 --steady-iters 100 --order-repeats 1 \
  --sl-idx <FROZEN_SL> --seed 106
```

### 10.3 必报指标

每次迭代先保存每-rank原始值，再计算 rank-max：

- plan submit -> plan ready；其中 CPU、GPU kernel、materialization 分量。
- planner exposed time：consumer 关键路径上未被前一 microbatch expert compute/其他工作隐藏的时间。
- dispatch、combine、roundtrip、真实模型 consumer-ready 时间。
- token/s、payload GB/s；公式和有效/填充 token 分母必须声明。
- p50/p95/p99/mean/std/MAD/CV、每个 process run 的 block median。
- A/B paired delta、latency ratio 与 median log-ratio 的分层 bootstrap 95% CI；重采样层级是 process run -> block，不把相关 iteration 当独立样本。p95/p99 还要按 trace window/block 分层 bootstrap。
- runtime path payload/bytes、local-copy bytes、fallback/reject。
- `cut_gap` low/mid/high 和 workload/model/topology 分层结果。

planner 的“被隐藏”只能由同一真实执行时间线中的 event dependency 或 Nsight 时间区间 union/intersection 证明；用 `expert_time - planner_time` 算出来的数字不能算证据。

### 10.4 判定

先用 A/A 伪配对的 `|median log-ratio|` 95 分位冻结统计噪声界 `epsilon_perf`；用 flow solver 精度和离散 token 单位冻结 `epsilon_cut`。A/B paired ratio 的 95% CI 必须排除 1，效应还必须超过 `epsilon_perf`；同一主图里的多个预注册 contrast 对 p-value 使用 Holm 校正，或预先生成等价 simultaneous CI。

统计显著不等于业务上“必要”。在查看 confirmatory 结果之前，业务/系统 owner 必须把目标指标的最小实用收益 `delta_SLO`、low/no-gap strata 回归预算、需覆盖的流量比例写入 `preregistration.md`。若没有这些真实约束，只能写 `STATISTICALLY_DISTINGUISHABLE` 或 `INCONCLUSIVE`，不得判为 necessary/default，也不得由实验者临时编一个 SLO。

第三 rail 可列为**默认启用**，必须同时满足：

1. confirmatory 真实 trace 中有预注册 workload stratum 的 `cut_gap > epsilon_cut`。
2. runtime counter 证明该 stratum 确实执行了 third-rail payload，且端点/payload 守恒。
3. A/B paired rank-max consumer-ready 或模型 E2E 的 95% CI 排除 1，多重 contrast 的 Holm 门禁通过，方向为 adaptive 更快，效应超过 A/A 噪声界和预注册 `delta_SLO`；planner 与额外 local copy 已计入。
4. 在 low/no-gap stratum 中没有超过 A/A 噪声界或预注册回归预算的退化。
5. 达到预注册流量覆盖比例，并至少跨两个真实模型/数据集族和 2x8、4x8 两种规模复现；否则只能给窄范围或 guarded 结论。

使用以下固定 verdict，不自创新的模糊措辞：

- `NO / OPTIMIZE_ONE_HOP`：`P1* == Pall*`；若 `P1 > P1*`，先修 one-hop 实现。
- `REJECT_THIRD_RAIL`：真实 trace 有结构 gap，但 adaptive 的完整 E2E（含 planner）没有合格净收益。
- `SELECTIVE_ONLY`：仅部分真实 strata 满足结构、统计、实用收益和回归门禁，且只依赖运行时可观测量、在 tuning split 冻结的在线 gate 在 held-out trace 上正确激活。
- `DEFAULT`：满足预注册流量覆盖、`delta_SLO`、所有回归、物理 counter 和泛化门禁。
- `INCONCLUSIVE`：缺真实 trace、`delta_SLO`、直接 A/B 配对、正确性或 runtime/物理 counter。

不要把“统计不显著”写成“二者完全相同”，也不要把仅有逻辑 planner 结果写成 physical rail 结论。

## 11. Stage 6：通信库 common-fair

### 11.1 对象

- 同树 DeepEP V2 `off`
- 同树 RailBalance `one_hop`
- 同树 RailBalance frozen `adaptive`
- clean upstream DeepEP V2 `dd758caf...`
- NCCL EP `63cf786b...`
- UCCL-EP `61ee4240...`

UltraEP/MoonEP 不进入这张直接 transport 排名表：前者改变 placement/replication，后者公开范围为单机 EP8。

### 11.2 共同交集点

现有 `tests/elastic/experiments/rail_balance_common_fair_design_v1.json` 是设计 scaffold：`future_claim_eligible=false`，route/source/resource 等多处仍为 `NOT_FROZEN`，并含 `rail_hot` 合成 pattern。它不能直接执行或产生论文结论。集群 Codex 可以复用其 fail-closed schema/打包工具，但必须先绑定本轮 clean source、真实 route trace、适配器、资源与环境 hash；`rail_hot` 只留在 stress 轨。

先跑所有实现都能保持语义等价的 BF16 交集：

| 轨 | nodes x GPUs | experts / hidden / top-k | tokens/rank | dtype | 输入 |
|---|---:|---:|---:|---|---|
| LL | 2x8、4x8 | 256 / 7168 / 8 | 128 | BF16 dispatch + BF16 combine | 相同真实 decode/低 token trace |
| HT | 2x8、4x8 | 256 / 7168 / 8 | 4096 | BF16 dispatch + BF16 combine | 相同真实 prefill/train trace |

当前 Rail force 路径不支持 FP8，因此 FP8 dispatch/BF16 combine 只能另列 author-compatible/partial-common 轨，不能与 BF16 交集点混排。UCCL 原生 internode 示例使用 E=288；common-fair 的 E=256 必须先通过正确性，若冻结版本不支持则记 `UNSUPPORTED`，原生 E=288 放作者轨。

### 11.3 共同 API 边界

薄适配器必须把同一 trace v2 top-k tensor 和同一 payload tensor 注入每个实现，并输出：

1. `router_ready -> plan/layout/handle_ready`
2. `dispatch_start -> receive_buffer_consumer_ready`
3. `combine_start -> source_buffer_consumer_ready`
4. `router_ready -> roundtrip_consumer_ready`

异步 API 必须在真实 consumer stream 写 dependency event 后计时。初始化、连接建立和一次性 buffer allocation 从稳态排除，但另报 cold-start；每轮必须包含动态 route/handle update 的工作不能排除。NCCL EP LL 还应单独报告其 Complete/完成阶段，不能只测 host C API 返回。

各项目原生 benchmark 的输入生成和计时不满足以上条件时，只能进作者轨。尤其不能把 NCCL EP host C API 时间、DeepEP Kineto kernel 时间、MoonEP rank-mean 当 common-fair 数据。

### 11.4 两种资源轨

1. **fixed-resource**：GPU 数、NIC/rail、SM budget、QP budget、payload/dtype 相同；所有 CPU proxy/core、hugepage 和 daemon 都记录。UCCL 的 4 proxy threads/GPU 若启用，固定绑核并报告 CPU utilization、NUMA remote access。
2. **author-best-with-budget**：允许每个实现使用官方建议模式，在同一 tuning split、最多 30 个配置/同一墙钟预算内选择；完整报告不同资源用量，不称“同资源”。

两轨都用相同 A/A、ABBA/BAAB、5 次进程重启、rank-max 和 bootstrap 规则。库的内建带宽公式只作附表，主表使用本手册统一公式。

### 11.5 作者原生命令（smoke/reproduction，不是 common-fair）

NCCL EP tag 的构建与原生 LL/HT 参数：

```bash
make -j src.build BUILDDIR=<NCCL_BUILD_ROOT>
make -C contrib/nccl_ep MPI=1 BUILDDIR=<NCCL_BUILD_ROOT> \
  NVCC_GENCODE="-gencode=arch=compute_<CC>,code=sm_<CC>"

mpirun -np <TOTAL_GPUS> --hostfile <HOSTFILE> --map-by ppr:8:node \
  <NCCL_BUILD_ROOT>/test/nccl_ep/ep_bench \
  --algorithm ll --tokens 128 --hidden 7168 --top-k 8 --experts 256 \
  --warmup 1 --iters 1 --max-num-sms <FROZEN_SMS> --validate

mpirun -np <TOTAL_GPUS> --hostfile <HOSTFILE> --map-by ppr:8:node \
  <NCCL_BUILD_ROOT>/test/nccl_ep/ep_bench \
  --algorithm ll --tokens 128 --hidden 7168 --top-k 8 --experts 256 \
  --warmup 10 --iters 100 --max-num-sms <FROZEN_SMS>

mpirun -np <TOTAL_GPUS> --hostfile <HOSTFILE> --map-by ppr:8:node \
  <NCCL_BUILD_ROOT>/test/nccl_ep/ep_bench \
  --algorithm ht --tokens 4096 --hidden 7168 --top-k 8 --experts 256 \
  --warmup 1 --iters 1 --max-num-sms <FROZEN_SMS> --validate

mpirun -np <TOTAL_GPUS> --hostfile <HOSTFILE> --map-by ppr:8:node \
  <NCCL_BUILD_ROOT>/test/nccl_ep/ep_bench \
  --algorithm ht --tokens 4096 --hidden 7168 --top-k 8 --experts 256 \
  --warmup 10 --iters 100 --max-num-sms <FROZEN_SMS>
```

NCCL EP tag 要求的具体 CUDA/NCCL/MPI 和 `NCCL_GIN_TYPE=3` 等环境，以冻结 README 与集群硬件为准；保存 `ep_bench --help`。`--validate` 先单独跑，性能 run 不带额外 validation 开销。

UCCL-EP 官方入口：

```bash
# 每节点替换 node_rank；显式冻结 NCCL/UCCL 的 GID 与 socket interface
torchrun --nnodes=4 --nproc_per_node=8 --node_rank=<NODE_RANK> \
  --master_addr=<MASTER_ADDR> --master_port=<MASTER_PORT> \
  bench/test_low_latency.py --num-tokens=128 \
  --hidden=7168 --num-topk=8 --num-experts=288

torchrun --nnodes=4 --nproc_per_node=8 --node_rank=<NODE_RANK> \
  --master_addr=<MASTER_ADDR> --master_port=<MASTER_PORT> \
  bench/test_internode.py --num-tokens=4096 \
  --hidden=7168 --num-topk=8 --num-experts=288 --test-ll-compatibility
```

DeepEP clean-upstream 先按冻结 README 完成 cluster init 定制并运行 `<DEEPEP_UPSTREAM_PYTHON> tests/elastic/test_ep.py`；任何适配改动都作为 patch 留存。不要直接复制 README 性能数作为本集群结果。

## 12. Stage 7：端到端真实 workload

只有通信后端已接入同一个冻结框架、模型、checkpoint、请求列表和到达时间表时才比较端到端。适配器缺失就标 `API_GAP`，不要用通信 microbenchmark 推算 TTFT/step time。

### 12.1 Serving

至少保存每请求原始：arrival、input/output token、TTFT、TPOT/ITL、E2E latency、成功/失败、batch/route window ID。主报 goodput、throughput、p50/p95/p99 latency 和 OOM/error，不只报总 token/s。

建议先复用可落地的论文锚点：

- vLLM 0.10 + Qwen3-30B-A3B，1000 requests、concurrency 32、1/2/4 nodes；四次作者式重复另存，common-fair 仍保留全量原始请求并使用统一 CI。
- SGLang 0.5.3 + Qwen3-235B-A22B-FP8 或 DeepSeek-R1-0528，EP16/EP32、input 4096/output 5；若模型/硬件不足，不缩小成自创“类似业务”，而是 `UNSUPPORTED`。
- UltraEP 的 STEM/LongBench 数据只使用实际数据样本与 tokenizer 长度；Poisson arrival 仅在复现论文到达过程时使用，并保存 seed/rate，不能为了制造 rail 压力调 rate。

### 12.2 Training

若有真实训练栈，比较相同 global batch、sequence、optimizer、checkpoint、EP/DP/TP/PP 和 expert compute：step time、tokens/s、通信暴露时间、planner 暴露时间、loss/gradient 一致性、峰值显存、OOM。至少一个完整 warmup 后的连续训练窗口；仅 replay 通信不能称训练 E2E。

## 13. Stage 8：UltraEP × RailBalance 组合

UltraEP 作用于 expert placement/replication/reroute，RailBalance 选择通信 egress；应做组合而非不对称横比。固定同一真实模型/trace、同一底层 DeepEP V2 transport 和资源，做 2x2：

| | `one_hop` | frozen `adaptive` |
|---|---|---|
| placement/replication off | A | B |
| UltraEP on | C | D |

报告 B-A（Rail 主效应）、C-A（UltraEP 主效应）、`D-C` 与 `D-B`，并给 interaction：`D - C - B + A`。placement 开启后保存逻辑 router top-k、物理 replica/reroute 后目标、weight sync、grad reduce、额外显存和 planner 时间，不能只保存最终 NIC histogram。

真实 workload 优先采用 UltraEP 论文模型族（GLM4.5/4.7、Qwen3-235B、DeepSeek-V3）和实际数据；集群不具备相同模型/规模时称“组合移植实验”。官方分布式 smoke：

```bash
torchrun --nproc_per_node=<GPUS_PER_NODE> --nnodes=<NNODES> \
  --node_rank=<NODE_RANK> --master_addr=<MASTER_ADDR> \
  --master_port=<MASTER_PORT> tests/test_e2e.py --num-experts 256
```

UltraEP 当前公开代码的 DeepEP V2 支持仍有版本/移植风险。若 port 会改变 transport 或 buffer semantics，则停止并记 `API_GAP`，不以 HybridEP/DeepEP V1 的结果替代 V2 组合结论。

## 14. Stage 9：MoonEP 附录

MoonEP 只回答单机 EP8 的 planning、redundant expert、zero-copy/layout 开销；不进入多机 Rail 主结论。

若集群确为 H20 EP8，冻结官方参数 S=8192、E=384、H=7168、K=8、专家 hidden 2048、32 SM，并运行：

```bash
torchrun --nproc_per_node=8 benchmarks/bench_vs_deepep.py \
  --out <RUN_ROOT>/08_author_track/moonep/result.csv \
  --plot <RUN_ROOT>/08_author_track/moonep/result.png
```

官方 benchmark 使用合成 maxvio sweep、20 warmup/50 iterations、CUDA events 和跨 rank arithmetic mean，并明确不含 grad-reduce；这些只属于作者轨。common-fair 附录必须补 rank-max、原始 iteration 和真实 trace。非 H20 或非 EP8 时标为 port study，不能称作者硬件复现。

## 15. 物理路径 counter 闭环

软件 planner/oracle 只能说明“打算走哪条路”。每个 confirmatory block 必须在同一时间窗采集：

1. runtime path counters：四类 path 的 payload/bytes/local-copy/fallback。
2. 每 NIC/port 的 tx/rx bytes、packets、errors/retries/congestion。
3. 可用时的 QP 级 bytes/operations；不可用明确记 `COUNTER_GAP`。
4. NVLink/PCIe local-copy counter 或 profiler 区间。
5. block 前后空窗背景 counter，用差分扣除并给测量误差。

counter 名称和语义依 NIC/driver 而异。先枚举 `/sys/class/infiniband/*/ports/*/counters`、vendor 工具和 `--help`，把原始名称/单位/是否累计/是否 wrap 写入 manifest；不能把相近名字硬映射。

闭环至少验证：

```text
sum(runtime path payload) == endpoint payload count
third-rail runtime count > 0（声称使用第三 rail 的窗口）
NIC byte delta 与 runtime predicted scale-out bytes 在校准误差内一致
额外 NVLink/PCIe local-copy 与选路方向一致
error/retry 没有解释掉延迟差异
```

无法闭环时仍可报告逻辑 planner 结果和 E2E 性能，但措辞只能是“logical third-rail selection”，不能声称真实物理 rail 流量已重平衡。

## 16. Profiling 顺序

只对已通过未 profile 性能测量的代表点做归因：

1. **Nsight Systems first**：1 个 low-gap 与 1 个 high-gap 真实窗口，A/B 各一次。捕获 NVTX、CUDA API、kernel、stream/event、CPU proxy；只用来确定暴露区间和 overlap。
2. **Nsight Compute second**：只采 Nsys 已定位的 exposed planner/materialization/kernel，单 kernel、最少指标集；不拿 NCU 时延回填主表。
3. 将 source -> NVTX -> stream -> kernel -> PTX/SASS/metric 建立引用链。aggregate GPU util、kernel 名或 occupancy 单项不能单独证明瓶颈。

profile run 一律标 `diagnostic_profiler=true` / `performance_claim_eligible=false`。

## 17. 统计、排除与报告

### 17.1 统计规则

- 分布式延迟真值：每 iteration 的 rank-local consumer-ready elapsed 的最大值。
- p50/p95/p99/mean/std/MAD/CV 全报；吞吐与 latency 同报。
- 配对比较使用每个相邻 A/B block 的 median；CI 在 process-run/block 层做 bootstrap。
- A/A 与 background calibration 和主实验使用同样窗口长度、重启次数和统计代码。
- 冷启动、steady、profile、contended smoke 分表。
- 多 workload 同时给逐 workload、宏平均和 token-weighted 汇总；不要只给最有利的一种汇总。

### 17.2 预注册排除

只允许在看结果前定义的机器错误排除：进程崩溃、Xid/ECC、counter wrap 无法恢复、网络 error、环境门禁失败、正确性失败。高延迟不是异常点理由。作者论文使用 IQR filter 时可在 author-track 复现，但 common-fair 主表必须同时保留 raw/all-sample 结果。

### 17.3 禁止的结论跳跃

- “synthetic closed_block 赢”不能推出真实 workload 需要 third rail。
- “planner predicted balanced”不能推出物理 NIC 已平衡。
- “microbenchmark GB/s 高”不能推出 serving TTFT 或 training step 更快。
- “UltraEP + Rail 最好”若四格未在同一 transport/trace 上测，不能称 interaction。
- “MoonEP 单机赢”不能推出多机 RDMA 结论。
- 跨 H100/H200/B200/H20、IB/EFA、FP8/BF16 的作者数字不能放在同一条形图直接排序。

## 18. 最终交付清单

集群 Codex 完成后，先发一条短消息报告 `RUN_ROOT`、完成/失败矩阵和最重要的 blocker；随后提交完整包。`report.md` 必须包含：

1. 一页结论：third rail 是 default、guarded 还是未证明有必要；适用模型/trace/拓扑边界。
2. source/environment/trace/split/resource manifest 的 hash 与可点击位置。
3. 所有实验单元的状态矩阵，包括未跑和不支持项。
4. `one_hop -> adaptive` 的 real-trace paired 原始样本、CI、cut certificate、runtime/physical counter 闭环。
5. planner total/hidden/exposed 时间与 Nsys 区间证据。
6. common-fair 与 author-track 两套独立表，明确 dtype、阶段边界、统计方法、资源。
7. 2x2 UltraEP composition；MoonEP 单机附录。
8. correctness、错误、OOM、超时、排除项和负结果。
9. 一条从 clean checkout/build 到重建所有表图的命令，以及其成功日志。

验收时随机抽一个表格单元，必须能沿着 `table cell -> derived record -> raw rank samples -> command -> binary/source SHA -> trace SHA -> topology/counter` 反向追溯。任一环断裂，该单元降级为 diagnostic。

## 19. 推荐执行顺序与停止门禁

```text
S0 environment/source freeze
  -> S1 real trace v2 + split + replay validation
  -> S2 correctness
  -> S3 offline one-hop cut certificates
  -> S4 tuning + candidate freeze
  -> S5 real-trace one_hop/adaptive confirmatory + counters
  -> S6 direct-library common-fair
  -> S7 end-to-end real workloads
  -> S8 UltraEP 2x2 composition
  -> S9 MoonEP single-node appendix
  -> analysis/rebuild/audit
```

停止门禁：

- S1 未通过：停止 third-rail 性能结论，保留 synthetic correctness。
- S2 未通过：停止对应模式所有性能实验。
- 真实 confirmatory trace 全部 `cut_gap <= epsilon_cut`：不扩跑昂贵 third-rail profiling；报告“该语料未发现结构必要性”，仍可跑 one_hop 和库对比。
- runtime/physical counter 缺失：继续测 E2E，但禁用“物理 rail 已重平衡”措辞。
- common-fair adapter 改变库语义：降级 author/port 轨。
- 2x8 不能稳定复现或正确性失败：不扩到 4x8/8x8。

这套顺序的目标不是把所有格子跑满，而是在每一步只为可证伪、可推广到已声明范围的结论付出集群成本。
