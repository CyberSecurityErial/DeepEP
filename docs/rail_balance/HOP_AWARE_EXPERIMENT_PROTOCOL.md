# Hop-aware RailBalance 实验协议与无人值守脚手架

状态：2026-08-01 bootstrap v3。单 worktree campaign 已能在 8 卡不满足时
fail-close 为 `scaffold_only`，并保存固定命令、CPU gate、环境、失败与 artifact hash。
另有 stdlib-only 的 formal source-round preparer、coordinator、evaluator和整轮 live
executor：CUDA源码候选必须来自独立 clean worktree，只能在相同算法配置下与同一个
sealed parent比较，并由一把 lease覆盖整个 `4+4N` 顺序。preparer把人工预注册信息与
现场 Git身份冻结为 manifest/plan；executor默认 check-only，live必须显式双重确认。
脚手架仍不自动生成/修改 CUDA候选；没有实际执行的计划始终是 `NOT_RUN`，任何组件都
不会自动产生论文性能 Leader。

## 1. 一条命令

```bash
PYTHONPATH="$PWD/tests/elastic:$PWD" \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/run_rail_balance_hop_campaign.py \
  --manifest tests/elastic/experiments/hop_local8_sm90_v1.json \
  --run-id <unique-lowercase-run-id>
```

Supervisor 本身只导入 Python 标准库和纯 schema，不导入 `torch`、`deep_ep` 或 CUDA。
所有 CUDA 入口只能作为门禁通过后的固定子进程启动。manifest 不能使用 shell、任意
executable、`python -c`、`LD_PRELOAD`、任意 `PYTHONPATH` 或逃逸 artifact 路径。

## 2. 冻结口径

[`hop_local8_sm90_v1.json`](../../tests/elastic/experiments/hop_local8_sm90_v1.json)
固定了用户要求的十项实验事实：

1. packed record 输入、planner 输出和 checked source-adapter 接口；
2. dtype、shape、layout、stride 与 32-byte alignment；
3. SM90、CUDA 12.8、PyTorch 2.11.0+cu128 和恰好 8 张 host GPU；
4. CPU planner、vnode payload reference；
5. pinned Python、强制 extension build、codegen 命令；
6. CPU、20-case CUDA planner、codegen、4×2 vnode correctness；
7. 同一 C100 input、10 warmup、100 steady、ABBA+BAAB、rank-max 计时；
8. profiler-free truth → Nsys → 仅按 Nsys 选择 targeted NCU；
9. planner/payload exact，其他未声明 exact 的浮点比较上限为 relative error `<1e-3`；
10. 显著性、理论 headroom、每轮 2～4 候选、最多 6 轮和退出条件。

当前 manifest 是接管 checkpoint 的 bootstrap validation，不是正式的 2～4-candidate
优化轮。正式轮必须从 clean Leader 复制同一 `frozen_contract`，每个候选只声明一个
主要假设，并使用独立 source identity、build 和 artifact 目录。当前 manifest 内的
`one_hop`/`adaptive` 是**同一二进制的算法模式诊断**；它可以回答模式差异，绝不能
证明 candidate CUDA source 比 parent CUDA source 更快。

## 3. Fail-closed 门禁

启动顺序固定为：

```text
strict JSON/schema + raw/canonical SHA256
→ fixed flock
→ Git commit/diff/untracked content identity
→ du -B1 /home/chen
→ 3 次连续 all-host-GPU idle sample
→ render fixed argv/env
→ CPU gates
→ 每个 GPU stage 前重查 source + GPU
→ 子进程期间监测 artifact、/home 和外来 GPU PGID
```

本机硬约束为 decimal `300,000,000,000` bytes。v1 manifest 最多允许 8 GB campaign
artifact，并额外预留 5 GB emergency headroom。不能测量目录使用量、预算加预留可能越过
硬上限、运行中达到上限或 artifact 超预算时均终止并保留失败记录；runner 不自动删除
conda 环境。

GPU 必须恰好 8 张，UUID/index 唯一、compute mode 合法、MIG 合法，且没有任何 compute
PID、MPS 或 profiler。名字不是放行条件：Qwen、短算子、allreduce 和未知任务都一样
阻断。GPU stage 的进程组启动后，只允许同一 PGID 的 compute PID；出现外来 PGID 会
终止当前 stage 并使证据失效。

CPU gate 强制 `CUDA_VISIBLE_DEVICES=''`。每个 campaign child 都进入 supervisor 拥有的
独立 process group；被监督的 build、vnode、C100 与 codegen harness 不再嵌套
`setsid`。timeout、异常或 harness 遗留孙进程时，supervisor 会清理并重新扫描整个
PGID；发现泄漏即使主进程返回 0 也判失败。

`flock`、`nvidia-smi` 和 `du` 仍是用户态协作/采样，不能替代调度器独占和 filesystem
quota。报告显式保留这个 TOCTOU 限制；绝不把采样门禁写成硬件级绝对保证。

## 4. 当前 stage DAG

- CPU：reference、报告契约、Hybrid API、layout、policy、public lifecycle；
- compile：clean-tree 强制 build、extension identity、代表性 JIT/SASS codegen；
- CUDA correctness：dispatch/combine codegen、20-case planner；随后对 G2 与 tiny-tail
  focused cases 分别运行 memcheck、synccheck、initcheck、racecheck，四类 sanitizer
  全过后才允许 one-hop/adaptive 4×2 vnode；
- benchmark：C100 H256 source，one-hop=A、adaptive=B，顺序 `ABBA BAAB`，每次
  10+100；
- profile：bootstrap manifest 中 Nsys **显式 disabled**。只有 formal source-round
  接受的候选才生成 follow-up manifest 跑同 shape Nsys；NCU 必须由新 Nsys 选出 exact
  kernel/rank/range 后另行预注册，不能默认 kernel replay 或 `--set full`。

当前共 25 个 stage。每个 GPU/JIT attempt 使用自己 `{stage_dir}/jit`，C100 每次进程
启动都看到空 cache；不同 block 不得共用 JIT 目录或 cold report。

一个 stage 失败不会抹除同层已经完成的独立结果；依赖它的后继标为 `skipped` 并保留
原因。compile 失败不运行 CUDA correctness，correctness 失败不运行 benchmark，任一
matched benchmark 失败不运行 profile。

## 5. Artifact 与 Leaderboard

每个 attempt 保存：

```text
command.json
stdout.log
stderr.log
result.json
declared artifacts + SHA256
```

campaign 保存 manifest、Git/source identity、pre/post `/home`、GPU snapshots、所有 stage
结果、`result.json` 和 `SHA256SUMS`。`result.json` 在最终磁盘检查前只是 provisional；
只有最后发布且绑定 result/SHA256SUMS/source/manifest/Leaderboard identity 的
`FINALIZED.json` 才是终端提交记录。formal evaluator 必须拒绝缺失、符号链接、hash
不符或 `round_evaluation_allowed=false` 的记录。运行时权威 Leaderboard 是 Git-ignored：

```text
.cache/rail_balance/hop_campaign/leaderboard.jsonl
```

每行包含 parent、单变量假设、预期指标、风险、compile/correctness/benchmark/profile
gate、source/manifest/artifact identity、失败原因和结论。append 使用独立 flock、
`O_APPEND|O_NOFOLLOW`、fsync 和稳定 `entry_id` 去重。tracked 文档只在 review/commit
边界发布 snapshot，避免一次运行先把 source tree 弄脏。

单 worktree supervisor 始终写 `performance_claim_allowed=false`。fail-closed CPU evaluator
`rail_balance_campaign_round.py` 已实现并会检查 coordinator/plan/preflight/source、
`FINALIZED.json`、同 workload、全部 raw distributions、全局 `4+4N` 不重叠时序、paired
噪声和绝对 candidate latency。它只消费证据，不启动 CUDA。整轮 executor
`rail_balance_source_round_executor.py` 已接通，但截至本文冻结时没有真实 CUDA候选
source manifest、完整八卡窗口或正式 raw round，因此没有可供 evaluator晋升的版本。
真实 NIC/RDMA claim 还需要多机 runtime/NIC/QP counters。

artifact 树只能包含普通目录和普通文件；符号链接、FIFO、device 等特殊文件都使
attempt 失败。命令中的绝对路径与 `{root}/..` 逃逸同样在 schema 阶段拒绝。

## 6. Formal CUDA source round

正式 source round 与 bootstrap algorithm A/B 是两条不同证据链。最低合同为：

1. 中立 control checkout 持有同一个全局 GPU lease 直到整轮结束；parent 与 2～4 个
   candidate worktree 属于同一 Git common-dir，realpath、device/inode 唯一，pre/post
   均 clean。
2. 顶层只有一个共享 `parent_ref`：commit/tree/source hash、sealed build+gate bundle、
   extension/SASS/JIT identity、toolchain、runner/schema/harness hash。共享的是不可变的
   parent 构建与门禁身份，**不是** benchmark raw samples。
3. candidate 必须从该 parent 派生，声明一个主要假设、patch hash 和允许修改的 CUDA
   文件；candidate source/build identity 必须不同于 parent 和所有 sibling。infra、
   algorithm、shape、layout、input、timer、GPU UUID mapping 完全相同。
4. 每个 block 都现场创建唯一 run ID、artifact root 与空 JIT 目录。全局顺序固定为
   `P0,C1_0..Cn_0,C1_1..Cn_1,P1,C1_2..Cn_2,P2,P3,C1_3..Cn_3`；抽取任意候选的
   子序列都必须是 `P C C P C P P C`，且 monotonic 时间窗口不重叠。
5. 每个 source 先独立通过 compile、全部 correctness 和 sanitizer；一个候选失败只
   记入它自己的 Leaderboard 行，不阻塞 sibling。全轮 parent drift 超阈值则整轮不
   可晋升。
6. 先用 paired/A-A 噪声门禁判候选是否有资格，再按**绝对 candidate latency**选择
   最快版本。最快两者差异未越过统一噪声时，状态为 undecided 并追加 head-to-head；
   不得按相对改善百分比或候选 ID 决胜。

preparer与 coordinator只冻结/materialize并审计上述执行计划，明确输出 `NOT_RUN`；
它们都不调用 GPU。CPU evaluator能 fail-close验证完整 `4+4N` artifact及选择规则，但
不会制造 artifact。live executor已经实现：默认 check-only；只有 `--live` 与完全匹配的
`--confirm-round-id` 同时出现，才在中立 control worktree获取一把整轮 flock，依次调用
固定 runner，并为每块写出绑定 plan/control/parent/executor身份的 terminal coordinator
record。整轮结束后它生成 formal manifest、现场调用 evaluator、交叉核对晋升布尔值，
最后写 `SOURCE_ROUND_FINALIZED.json`。任一候选失败会完整保留其失败 evidence，但不会
阻塞 sibling 的块；全局证据异常则禁止晋升。

### 6.1 从人工预注册到不可变 source round

真实候选存在后，人工 spec只允许以下顶层字段：

- `schema_version=1`、`round_id`、绝对且规范化的 `control_worktree` /
  `artifact_root`；artifact root必须已存在、owner为当前 UID且权限恰为 `0700`；
- `parent={id,worktree}`，以及按 ID排序的 2～4 个 candidate；每个 candidate只有
  `id/worktree/primary_change/hypothesis/expected_profile_metrics/risks`，其中修改、假设、
  预期 Profile指标和风险都必须在运行前写清；
- 排序且唯一的 `allowed_candidate_files`；候选只能是共享 parent之上的一个独立
  single-parent commit，不能堆叠 sibling，也不能修改冻结 harness；
- `frozen_execution_contract={algorithm,shape,timing,toolchain,gpu_mapping}`；它是 formal
  evaluator的窄合同，不是 campaign template中字段更多的 `frozen_contract`。`algorithm`
  与 `shape` 必须是 evaluator定义的完整字段；`timing` 必须固定 10 warmup、100 steady
  及统一 rank-max timer。`shape/timing` 应逐字段复制选定 C100 public report中 evaluator
  接受的精确字段子集，并与预注册 campaign一致；不能从 prose手抄，
  `logical_bytes_per_iteration` 等工作量事实也不能填占位值。`toolchain` 应从
  clean campaign的 `environment/toolchain.json` 复制完整规范化对象（若有
  `created_utc` 只删除该字段），不能手工简写。GPU mapping必须逐项复制冻结 campaign
  template `resources.expected_gpu_index_uuid_mapping`，并在 live preflight与现场 0..7 的
  八个唯一 UUID一致；不能因当前只有六张空闲卡而缩小正式合同。

在中立 control worktree运行 preparer；它要求当前 CWD确实是 control顶层，且当前执行
文件就是该 control tree中的冻结副本。它现场推导所有 HEAD/tree、共同 Git object
store、候选 binary diff/hash和20个 harness hash，执行两次 preflight，然后以 no-clobber
方式先发布 plan、最后发布 manifest作为终端标记：

```bash
PYTHONPATH="$PWD/tests/elastic:$PWD" \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/prepare_rail_balance_source_round.py \
  --spec <absolute-human-spec.json> \
  --manifest-output <absolute-artifact-root>/source-round.json \
  --plan-output <absolute-artifact-root>/plan.json
```

成功输出仍是 `PREPARED_NOT_RUN`。preparer不创建、切换或修改 worktree，不生成 CUDA
候选，不选择 winner，也不启动 build/GPU/benchmark。发布协议不是 filesystem pair
transaction：它先发布 plan、最后才以 manifest作为“pair可消费”的 terminal marker，
且绝不按 final pathname回删。任何异常、`SIGKILL` 或主机崩溃落在两次 link之间都可能
留下没有 manifest的孤立 plan；该 plan故意不可执行且同路径 rerun会 no-clobber失败。
manifest link之后的晚失败可能留下完整 final pair；不可捕获的进程死亡还可能留下隐藏
stage文件。此时不能凭文件名猜成功，必须重新执行 strict manifest/plan/executor
check-only审计。
每轮必须使用全新的 artifact目录；同一路径并发 authoring、同 UID主动替换目录以及宿主
崩溃都不在协作式原子性保证内。preparer不持有整轮 GPU flock；executor check/live会
再次 preflight，以关闭 prepare到执行之间的 Git漂移。preparer在两轮 preflight期间持有
artifact root与两个 output parent的 dirfd，并在 staging、两个 link前后复验
dev/inode/owner/mode及路径重开身份；这仍不能替代 filesystem quota、调度器隔离或敌对
同 UID进程的安全边界。

human spec原文件及其 SHA不会写入派生 manifest；发布后 executor的权威输入只有 manifest
和其绑定的 plan。如论文 provenance需要保留人工预注册原文，应在运行前另存只读副本与
SHA，且不能事后把它补进已经发布的 pair。

preparer会打印完整的 executor check-only/live命令。先执行 check-only；也可以直接用
下面两条底层命令独立复核 coordinator与 executor绑定：

```bash
PYTHONPATH="$PWD/tests/elastic:$PWD" \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/rail_balance_source_round_coordinator.py \
  --manifest <absolute-source-round.json> \
  --output <absolute-artifact-root>/plan.json \
  --check-only

PYTHONPATH="$PWD/tests/elastic:$PWD" \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/rail_balance_source_round_executor.py \
  --source-manifest <absolute-source-round.json> \
  --plan <absolute-artifact-root>/plan.json \
  --campaign-template "$PWD/tests/elastic/experiments/hop_local8_sm90_v1.json"
```

只有八卡连续通过 exclusive preflight且用户任务均不在运行时，才把第二条命令追加：

```text
--live --confirm-round-id <exact-frozen-round-id>
```

当前没有 source-round manifest：真实 2～4 个单变量 CUDA候选尚未由新测量产生，不能为
验证 executor而伪造候选提交。executor依赖冻结 runner将全部 stage保留在它创建的 PGID；
主机没有可写 cgroup-v2 kill scope。同 UID恶意路径替换、子进程主动逃逸 PGID、executor
遭 `SIGKILL` 或宿主崩溃超出协作式实验威胁模型，终端记录会明确保留这些边界。

## 7. 2026-08-01 scaffold 记录

当前 terminal合同下的 clean-tree finalized记录为：

```text
run_id: takeover-scaffold-20260801-02
source: be4a69b9e363614373b1298d94e338a340fdcfa0, clean pre/post
status/scope: scaffold_only/scaffold_only
reasons: Qwen PID 3047501,3047502; explicit scaffold-only
CPU gates: 6/6 PASS
exclusive-8GPU stages: 19 skipped; GPU attempts/process starts: 0/0
artifact bytes: 128,319
/home pre/post: 271,565,897,728 / 271,567,323,136 bytes
result.json sha256:
b2c12aba2afa55a3aa734c55b2d4c3aab7c136902ceee78e089b555fd3424d04
SHA256SUMS sha256:
4b51d963430d63c756f45378f4105e8a4bc4e8df1e8faa4daca9e2ac05a2a755
FINALIZED.json sha256:
aef4a7bec4d2c343c49da6c28e7ba1ff853abc35cd57075f338f8561480f8db6
```

全部 `SHA256SUMS` 条目已重算一致。`FINALIZED.json` 的
`round_evaluation_allowed=false` 是正确结果：它证明 clean source下 CPU与资源
fail-close路径可复现，不包含 GPU样本，不能晋升或产生性能结论。

更早的历史记录为：

```text
run_id: takeover-scaffold-20260801-01
status / scope: scaffold_only / scaffold_only
reasons: Qwen PID 2753622,2753623; source_worktree_dirty
CPU gates: 6/6 PASS
GPU/benchmark/profile attempts: 0
artifact bytes at report build: 72,207
/home pre/post: 271,249,965,056 / 271,250,386,944 bytes
SHA256SUMS sha256:
f03a48ad989ec98cea33488805296bd95ea153f195df18bb484f70607c99ad38
```

这份 `-01` artifact 是 **legacy pre-terminal-contract** 记录：它没有
`FINALIZED.json`。其 hash可保留历史 provenance，但 formal evaluator必须拒绝，不能
证明当前协议合规、不能晋升，也不包含性能样本。`terminal-smoke-20260801-01` 只验证
了 `FINALIZED.json` schema，却使用 `--no-execute` 且所有 stage skipped，同样不能替代
fresh 6/6 CPU scaffold。脚手架提交并形成 clean tree后必须生成新的 finalized run。

一次探索性 2×2 vnode 尝试也被保留为失败：启动前采样已看到短算子瞬时占满 8 卡；
测试 runtime 随后在 `buffer.hpp` 的 `nccl_context->num_ranks == kWorldRanks` 断言处
fail-close。它没有进入 vnode data movement、没有写 JSON；未验证的参数化代码已撤回。
G=2 planner kernel 仍由单 GPU exact test 和四类 sanitizer 覆盖，但不能替代 4×2 vnode。

## 8. 论文对手入口

第一手论文/代码/命令/图表/原始数据缺口见
[`COMPETITOR_EVIDENCE_2026-08-01.md`](COMPETITOR_EVIDENCE_2026-08-01.md)。当前按
“相关性”和“可执行性”分成：

- `P0-R`：有可冻结代码的直接数据面对手。内部首基线是 DeepEP v2 `off`；外部为
  NCCL EP、SABRE、SwiftEP、UCCL-EP 与 fabric-lib/pplx-garden；
- `P0-E`：UBEP 与 hop-aware scheduling 的思想存在明确 novelty重叠，但只公开 Ascend/CM384
  论文、没有代码，因此只进入 novelty/设计差异，不能制造 NVIDIA 性能点；
- `P1`：UniEP、FEPLB、UltraEP、MoonEP、ECHO、EPLB、LPLB 属于整层融合、专家
  placement 或计算负载赛道，必须做分层/组合归因，不进入纯 transport 主图。

SABRE已经使用节点内 proxy forwarding 做 NIC 负载均衡，UBEP已经按 hop 距离调度
token发送任务，fabric-lib/UCCL-EP等也已有多 NIC 分配。因此 novelty 只能限定为：
在 DeepEP/NVIDIA 多 NIC RDMA 数据面内，保持最终 expert endpoint 不变，做带 hop/cost
约束的 rail/vnode选择与有限节点内 forwarding；禁止使用泛化的“首次”措辞。

逐项 `AUTHOR_REPRO` / `COMMON_FAIR` / `COMPOSED` 预注册矩阵、trace合同、统一计时、
统计、profile、图表和失败状态见
[`PAPER_EXPERIMENT_MATRIX_2026-08-01.md`](PAPER_EXPERIMENT_MATRIX_2026-08-01.md)。所有行
当前均为 `NOT_RUN`。

在 competitor commit、同机硬件、route hash、dtype、API 边界、rank-max、raw samples 和
正确性全部冻结前，只能写“待复现 anchor”，不能写“已打平/超过”。
