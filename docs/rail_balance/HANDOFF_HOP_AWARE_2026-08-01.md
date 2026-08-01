# DeepEP RailBalance Hop-Aware 完整交接

> 用户最新冻结的主线：这台机器的性能优化只关注减少 **plan CUDA 算子时间开销**；
> 最终竞品性能与真实工作任务一定在外部多节点集群运行。当前只写各外部实验的环境依赖
> 与执行口径文档，不在本机安装、构建或部署这些环境。实验保持精简，不为完整矩阵而
> 过度设计。
>
> 精简实验清单：
> [`PAPER_PERFORMANCE_CHECKLIST_2026-08-01.md`](PAPER_PERFORMANCE_CHECKLIST_2026-08-01.md)。

最后核对：2026-08-01 UTC

仓库：`/home/chen/workspace/source_code/DeepEP`

GitHub fork：`git@github.com:CyberSecurityErial/DeepEP.git`

当前分支：`feat/rail-balance-hop-aware`

最近已发布代码检查点：`469e771`；UCCL/NCCL竞品静态预检与论文实验基础设施已经提交
并推送（恢复时实际 HEAD以 `git rev-parse HEAD` 为准）

原型基线：`feat/rail-balance-prototype` / `5199a04`

状态：**目标尚未完成；核心 CUDA/C++ 已形成提交且当前无实现文件 dirty；正式 8 卡
source round、4×2 vnode复验和真实多机 Rail/Gin 仍是门槛。**

本文是给下一位 Codex 的当前唯一入口。旧的
[`HANDOFF.md`](HANDOFF.md) 记录 2026-07-26 以前的 prototype 历史，内容很长且
分支已过时；需要追根溯源时再读，不要用它判断当前工作树。

## 0. 一屏状态

- `469e771` 相对 prototype `5199a04` 有 50 个 hop-aware提交。核心 planner/checkpoint
  已提交为 `286da0a`，benchmark证据门禁为 `771fcc6`，CUDA harness监督契约为
  `0f38688`，fail-closed campaign/formal source-round控制面为 `448a727`，竞品证据与
  论文口径为 `be4a69b`，source-round manifest freezer为`6103180`，fail-closed竞品
  preflight与收紧后的实验矩阵为`469e771`。后续提交会继续增加计数，恢复时必须现场
  查询 Git。
- 当前没有 `csrc/`、`deep_ep/include/` 或核心 CUDA 测试的未提交改动。campaign、
  formal-round执行控制面和 CPU-only source-round preparer已经提交；本次新增的是
  CPU-only competitor preflight、合同测试和论文证据文档，不能用旧的 dirty实现文件清单
  判断是否丢改动。
- 交接时没有遗留 benchmark、sanitizer 或构建进程；2026-08-01 takeover resource gate
  观察到用户 Qwen占 GPU0–1，短算子也曾瞬时占满8卡，均不得终止或干扰。
- 分支已推送并跟踪 `fork/feat/rail-balance-hop-aware`；`469e771` 时本地/远端
  ahead/behind为 `0/0`。新提交后仍须重新 push并现场核对。
- `off` 默认路径和 `legacy_exact` 必须保持不变。
- Python host capability 与编译 capability 仍为 false；不得因为 vnode 通过而解锁。
- 当前 CUDA planner 功能套件为 20/20；one-hop/adaptive 两条 4+4 vnode 在历史
  checkpoint通过，但 `286da0a` 后因完整8卡不空闲尚未重跑。
- 新 tiny-tail 路径的 memcheck/synccheck/initcheck/racecheck 均为 0 错误。
- `8c56939` 仍是最后一个 clean matched checked-adapter性能 Leader；较新的 planner
  只形成正确性候选，没有当前 clean matched加速结论，更没有 NIC/RDMA 结论。
- 接管后已经针对最终 C++/CUDA diff 完成一次 `build_ext --force`；之后只修改 Python
  benchmark、campaign和文档。未来任何 csrc/header改动仍必须重新 force build。
- 最终 CUDA/C++ tree 已通过强制构建、planner 20/20、reference 12/12、报告契约 8/8、
  API 9/9、dispatch/combine codegen、layout 5/5、policy 8/8、public lifecycle和四类
  sanitizer。完整 8 卡 vnode与正式性能轮仍未运行。

## 1. 为什么要做这件事

DeepEP 的 MoE Router 和 EPLB/MoonEP/UltraEP 一类方案决定 Token 最终在哪个专家、
哪个 Rank 上计算。它们可以把计算量和 Rank 总收发量做平，但仍不能保证每个
`source node -> destination node -> Rail` 的流量相同。

典型问题是：

```text
专家计算已经均衡
        ↓
某些 source/destination 对仍集中使用少数 Rail
        ↓
最热 NIC/Rail 决定 dispatch/combine 尾部完成时间
```

RailBalance 不迁移专家，也不改变最终计算 Rank。它位于 Router/EPLB 之后、DeepEP
真正把 payload 下放给 Gin/RDMA 之前，只改变运输所用的本地 egress Rail：

```text
Router / expert placement
        ↓ 最终专家端点固定
RailBalance planner
        ↓ 选择 egress/ingress Rail
source local forwarding（如需要）
        ↓
Gin/RDMA scale-out
        ↓
destination local forwarding（如需要）
        ↓
原定 expert GPU
```

prototype 已经证明 source proxy、静态 slot、destination forwarding、combine
return-unshuffle 和多机 runner 的工程底座可以工作，并且用户在两机 hotspot
workload 上观察到明显收益。hop-aware 这一轮不是推倒重写，而是收紧路径空间：

1. 优先只在两个端点 Rail 之间选择，保持一次节点内 forwarding；
2. one-hop 仍解决不了的热点才允许少量第三 Rail；
3. 再优化 planner 和 forwarding kernel；
4. 最终回到真实多机 Rail/Gin 做 correctness 与 rank-max 性能证明。

## 2. 必须掌握的语义

### 2.1 手册符号

对于一个跨节点 payload：

- `o`：源节点 owner local rank；
- `d`：目标节点；
- `t`：最终 expert 所在的目标 local rank；
- `e`：实际跨节点发送/接收所用的 local Rail。

单目标时节点内 forwarding 次数为：

```text
h(o,t,e) = [e != o] + [e != t]
```

| 条件 | 名称 | 路径 |
| --- | --- | --- |
| `o=t=e` | direct / 0-hop | `o --Rail o--> t` |
| `e=o, o!=t` | destination-forward 1-hop | 源不搬，目标 ingress `o -> t` |
| `e=t, o!=t` | source-forward 1-hop | 源 `o -> t`，RDMA 直接落到目标 `t` |
| `e not in {o,t}` | third-Rail / 2-hop | 源 `o -> e`，目标 `e -> t` |

当 `o=t` 时不存在 1-hop 改路由：要么 direct，要么换第三 Rail 并付出两侧
forwarding。这是 diagonal hotspot 必须用 selective 2-hop 的原因。

### 2.2 DeepEP 的真实传输单位：`target_mask`

手册最初写的是 `(o,d,t,e)`，但审计源码后发现 DeepEP 会把一个 Token 发往同一
目标节点的多个本地 expert rank 合并为**一个网络 payload**。如果为了 planner
把它拆成每个 `t` 一份，会破坏 payload 复用并放大 RDMA 字节。

所以代码中的唯一事实来源是：

```text
(owner o, destination node d, target-rank set T, selected Rail e)
```

`T` 用 32-bit `target_mask` 表示。单目标 `T={t}` 时与手册完全相同。多目标时：

```text
source forwards      = [e != o]
destination forwards = |T - {e}|
total local forwards = [e != o] + |T - {e}|
one-hop candidates   = {o} union T
third-Rail candidate = e outside ({o} union T)
```

这一点非常重要：当前 `path_units[direct,dst,src,two-hop]` 是“共享 payload 选择了
哪类 egress”的 record 分类，不等于逐 target 的真实 hop 总数。dirty tree 已新增
`source_forward_units`、`destination_forward_units`、
`minimum_local_forward_units` 和 `extra_local_forward_units`，从 records +
resolutions 直接计算实际 forwarding 工作量。

不要把共享 `target_mask` 改回每 target 一份 payload，也不要在 PR 中把 record
分类误称为真实 hop 字节。

## 3. 当前四种模式

```text
off
    原生 DeepEP V2；功能和性能基线；必须字节/JIT/buffer 身份不变。

legacy_exact
    prototype 的任意全 Rail 再分配；允许 third Rail；只作为内部对照，不能叫 one-hop。

one_hop
    每个共享 payload 只允许 e in ({o} union T)。

adaptive
    先得到完全相同的 one_hop 计划，再对残余热点做受阈值、cap 和 hop penalty
    约束的第三 Rail 迁移。
```

公开默认仍是 `off`。`one_hop` 和 `adaptive` 仍是实验路径，真实多机验证以前不得
解锁 compiled/host capability。

## 4. Dispatch 与 combine 的完整生命周期

### 4.1 Dispatch

```text
top-k expert IDs
    ↓ GPU record materializer：按目标节点去重，保留 target_mask
HopCopyRecord[o,token,remote-slot]
    ↓ one-hop / adaptive planner
HopCopyResolution{egress,channel,remote_slot,proxy_slot}
    ↓
e == o：保留在 owner 的原始 staging group
e != o：owner 通过 LSA/TMA 直接写 e 的最终 proxy-dispatch slot
    ↓ payload 完成 + fence + invocation ready
egress e 的 persistent scale-out warp 使用自己的 Rail Gin
    ↓
目标 ingress e
    ↓ e in T 时不向自己重复做 LSA copy，只 forward 到 T-{e}
最终 expert ranks T
```

数据不会经过“临时 shuffle buffer -> egress local buffer -> RDMA”的双重 HBM copy；
owner 直接写最终 proxy slot。slot 由 plan prefix 静态分配，不使用每 copy 的热点
`atomicAdd(peer_tail)`。

### 4.2 Ticket 与 combine

Dispatch 成功后把一次性 ticket 附着到 `EPHandle`。ticket 保留 invocation、plan、
proxy slot、destination forwarding metadata 和原 owner 信息。

Combine 不运行第二个 planner，而是逆向 replay dispatch 的路径：

```text
expert t
  -> destination ingress e
  -> Rail e 返回 source proxy e
  -> return-unshuffle 读取 proxy slot 路由
  -> original owner o / original token row
```

成功 combine 后 ticket 只能消费一次；可恢复的 pre-commit 失败恢复 live ticket；
commit 后错误 fail closed。不要添加独立 combine planner，也不要用原始 token index
直接作为共享 proxy slot，多个 owner 会冲突。

## 5. vnode 到底是什么

`vnode` 是 virtual node，纯测试设施。它把本机 8 张 GPU 逻辑切成例如 4+4：

```text
GPU 0..3 = virtual source node
GPU 4..7 = virtual destination node
```

跨“节点”段用本机 LSA/NVLink 模拟，因此可以验证：

- planner 选择；
- source shuffle；
- ingress/target forwarding；
- proxy slot 与 metadata；
- dispatch/combine 逆路径；
- payload 是否丢失、重复或拼回错误。

它不能验证 NIC、Gin、QP、RDMA completion、交换机拥塞或真实多机加速。vnode 是临时
验证 backend，生产代码不依赖它；所有测试结束后可保留为回归工具，但不能变成公开
运行时特性。

## 6. 这轮开发是怎么走到现在的

### 阶段 A：审计与正确语义

- `88f212a`：审计 prototype，发现 planner 丢失目标 local rank；
- `fdb760a`：Python endpoint-aware oracle；
- `5602110`：最小 GPU 设计；
- `2fbc156`：GPU record 保留 target mask；
- `4983ff5`：初始 one-hop materializer；
- `4c0db16`：vnode route 证明。

权威文档：

- [`../rail_balance_hop_aware_audit.md`](../rail_balance_hop_aware_audit.md)
- [`../rail_balance_hop_aware_design.md`](../rail_balance_hop_aware_design.md)

### 阶段 B：接入真正 Hybrid dispatch/combine

- source copy 接入现有 LSA/TMA proxy；
- one-hop plan 穿过 Hybrid dispatch；
- vnode dispatch/combine round trip 闭合；
- `40326a8` 加入 selective 2-hop；
- `2605775` 统一 reference/vnode/multinode CLI 与 JSON schema。

这解释了为什么没有一开始直接把复杂 planner 塞进原 persistent kernel：先证明
endpoint、slot、ticket、combine 逆路径，再优化 planner；否则性能 bug 与语义 bug
无法分离。

### 阶段 C：planner 成为主瓶颈并逐步并行

早期正确性优先版本由一个 warp/线程串行查看 record。Nsys 后来证明，真正慢的不是
source shuffle，而是“搬数据前逐条决定 egress 并分 slot”的 planner。

主要演进：

- batch selective two-hop；
- record validation 并行；
- retained prefix 预计算；
- exact channel assignment 加速；
- 跳过 packed padding；
- adaptive peak cache；
- adaptive candidate search 并行；
- singleton endpoint bypass；
- chunk planner；
- pre-count 与 plan gate overlap；
- endpoint pre-count/materializer 多 block；
- adaptive quota materializer 与 one-hop 共用静态 slot 数据面。

完整 43 个提交可用：

```bash
git log --oneline --reverse feat/rail-balance-prototype..HEAD
```

### 阶段 D：历史 dirty checkpoint（现已收口到 `286da0a`）

C100 的真实 records 不是 singleton，而常见 `target_mask=0b1111`。已提交的 adaptive
quota fast path只覆盖 singleton，multi-target 又退回 lane-0 逐 record 决策。八 rank
Nsys 把这个 kernel 定位为 17.387 ms median，占捕获 kernel time 的 68.1%；source
shuffle 只有约 25 us。因此当时的 dirty tree 做了以下修改；它们现已收口到
`286da0a`：

1. 为 G<=8 引入稠密 mask quota，但只列出真实 active group；
2. 扫描 `(o,d,target_mask)` 实际存在的 group，不扫 `2^G` 空 padding；
3. 先在小 quota 表决定 egress 数量；
4. 用 G*C/G blocks materialize records、channel prefix 和最终 static slots；
5. adaptive 最多 `G*32` 主轮次；
6. 大 batch 优先，找不到合格大 batch 后最多额外 G 个 tiny-tail round；
7. G>8 的 adaptive multi-target 明确 `InvalidSchedule`，不静默退化；one-hop 仍可用。

## 7. 历史未提交清单（pre-`286da0a`，不可当作当前状态）

下面是交接原始时刻的 9 项 dirty 实现，仅用于恢复历史。它们后来已由
`286da0a`/`771fcc6` 收口；当前状态必须用 `git status --short` 判断，不能按这张表
恢复或覆盖：

```text
M csrc/elastic/buffer.hpp
M csrc/kernels/elastic/rail_balance_hybrid_dispatch.hpp
M deep_ep/include/deep_ep/common/rail_balance_hybrid_layout.cuh
M deep_ep/include/deep_ep/impls/rail_balance_hybrid_plan.cuh
M docs/rail_balance/DEVELOPMENT_LOG.md
M docs/rail_balance/HOP_AWARE_PERFORMANCE_EVIDENCE.md
M tests/elastic/bench_rail_balance_hop_plan.py
M tests/elastic/bench_rail_balance_hybrid_lsa.py
M tests/elastic/test_rail_balance_hop_one_hop_cuda.py
```

各文件职责：

| 文件 | 当时的 dirty 改动 |
| --- | --- |
| `buffer.hpp` | 分配/持有 multi-target quota 与 cursor，交给生产/vnode plan |
| `rail_balance_hybrid_dispatch.hpp` | launcher ABI、6-stage materializer、JIT identity、private oracle int32 bound |
| `rail_balance_hybrid_layout.cuh` | G8 dense target-mask 上限与 materialize stage 枚举 |
| `rail_balance_hybrid_plan.cuh` | active-group compaction、quota planner、bounded batch/tiny tail、parallel assignment/static slot |
| `bench_rail_balance_hop_plan.py` | one-hop/adaptive 在同一 workload 上比较；增加 rotating/singleton pattern |
| `bench_rail_balance_hybrid_lsa.py` | 冷态真实 hop/forwarding accounting；status fail closed；报告 JSON |
| `test_rail_balance_hop_one_hop_cuda.py` | multi-target、G2..8/target width、G>8 fail-close、tiny-tail 反例等 |
| 两份日志 | 优化证据、失败、claim scope 与 artifact 路径 |

关键 workspace 别名是刻意的，目的是不再增加 arena：

- precount 时 `multi_target_egress_quota[...,0]` 先存 group count；decision 清 row 后
  改成 per-egress quota；materializer 只读；
- `endpoint_egress_quota[0]` 在 quota 清零前临时作为 has-multi marker；
- `multi_target_cursor` 先是 compact active-group list，rebalance 后清 prefix，再作为
  materializer consumed cursor。

这些生命周期已在 `rail_balance_hybrid_plan.cuh` 放了一处注释。不要为“命名更直观”
复制三份 workspace；如要改，应保持唯一事实来源和原有生命周期。

## 8. 正确性与工具证据

### 8.1 历史 dirty tree 证据（已由 8.3 的最终树门禁取代）

1. 强制扩展构建曾成功：

```text
setup.py build_ext --inplace --force                 PASS
```

接管后针对包含该检查在内的最终 C++/CUDA tree 已重新 force build并通过。

2. 单 GPU planner：

```text
20/20 PASS
```

覆盖：one-hop、adaptive、diagonal、closed block、multi-target、combined channel
capacity、corrupt/capacity fail-close、determinism、zero token、G2..8、K1/2/4/8、
所有合法 target width，以及新的 tiny-tail cap-as-limit 反例。

3. 4+4 vnode：

```text
one_hop / offdiag_hot dispatch-combine round trip    PASS
adaptive / diag_hot dispatch-combine round trip     PASS
```

4. Compute Sanitizer 2025.1，针对新 tiny-tail regression：

```text
memcheck   0 errors
synccheck  0 errors
initcheck  0 errors
racecheck  0 errors, 0 warnings, 0 hazards
```

5. CPU/contract 在交接前最后一轮：

```text
hop reference                                         12/12 PASS
unified hop benchmark contracts                        4/4 PASS
C080-A Hybrid API                                      9/9 PASS
dispatch codegen                                       PASS
combine codegen                                        运行到中途被交接打断
layout / policy / public lifecycle                     本轮需重跑
```

本小节是 pre-`286da0a` 历史；不得替代 8.3 的最终树证据，也不得用于正式性能晋升。

### 8.2 当前 diagnostic artifact

```text
fce381299b1ed1de78ffbf01bddc59c579567d12810c5e5066e1a510408d69e5
  .cache/rail_balance/hop-aware/multi-target/
  smoke-c100-volume-adaptive-tinytail-localhops-v8.json

42f9cfe3347d45e9f22bc1aec6893a934cb44d200946ee8bdcbb0a21b0b43fcf
  .cache/rail_balance/hop-aware/multi-target/
  diag-c100-rot1-adaptive-tinytail-v8.json

e93cc4b71d98b02a33377c6cb696a89a933381e18d41e75e0ff8296824aa1ab0
  .cache/rail_balance/hop-aware/multi-target/
  vnode-onehop-offdiag-tinytail-v8.json

d0ed457261be93bd1eee43f75a90d364148f06a72936bd60beb614465f94feb0
  .cache/rail_balance/hop-aware/multi-target/
  vnode-adaptive-diag-tinytail-v8.json

793e0a58bf28e7176508a22b231c3f86e064a10532869a8b4947583e7987519c
  .cache/rail_balance/hop-aware/multi-target/
  smoke-c100-volume-adaptive-localhop-bytes-v8.json
```

其中 C100 JSON 记录 dirty tree 且 `baseline_collection_eligible=false`；vnode JSON
只记录 `single_node_diagnostic` scope，没有 C100 的 Git/baseline 字段。它们都不得放进
正式 speedup 表。

### 8.3 2026-08-01 takeover final-tree evidence

接管后在不触碰用户任务的前提下完成：

```text
forced extension build                                  PASS
focused CUDA planner                                    20/20 PASS
hop reference                                            12/12 PASS
unified benchmark CPU contracts                           8/8 PASS
C080-A Hybrid API                                         9/9 PASS
dispatch codegen matrix                                   PASS
combine codegen matrix                                    PASS
Hybrid layout                                             5/5 PASS
Hybrid policy                                             8/8 PASS
public lifecycle                                          PASS
tiny-tail + G2 memcheck/synccheck/initcheck              0 errors
tiny-tail + G2 racecheck                                 0 errors/warnings/hazards
independent CUDA/C++ static audit                        no S0-S2 finding
campaign supervisor CPU contracts                       20/20 PASS
campaign process-group supervision CPU contracts          3/3 PASS
source-round preparer CPU contracts                     10/10 PASS
source-round coordinator CPU contracts                    9/9 PASS
formal source-round evaluator CPU contracts              19/19 PASS
formal source-round live-executor CPU contracts          16/16 PASS
```

完整 4×2 vnode 没有在 takeover tree 重跑：Qwen占 GPU0–1，短算子曾瞬时占满8卡。
一次 test-only 2×2 参数化尝试在运行前已看到竞争任务，并在 vnode prepare 的固定
`kWorldRanks=8`断言处 fail-close；没有进入数据搬运、没有生成 JSON，代码已撤回。
不能用单GPU G2 planner通过替代 vnode round trip。

terminal合同下最近的 clean-tree finalized scaffold：

```text
run: takeover-scaffold-20260801-04
source: 469e7712dbf00cc3767a0dbd07ce4c8213960475, clean pre/post
status/scope: scaffold_only/scaffold_only
reasons: Qwen PID 3047501,3047502 + explicit scaffold-only
CPU gates: 6/6 PASS
exclusive-8GPU stages skipped: 19
GPU attempts/process starts: 0/0
artifact: 109,458 bytes before result
/home pre/post: 272,010,797,056 / 272,011,198,464 bytes
result.json sha256:
47744b43c56ed3c24237378bc3380d87c1f9ee8221e96727fa4a80d37e353455
SHA256SUMS sha256:
c4b45c2076b1ea747d12bdccaa744b74cff27c762cd1519329aa04a36e0972a3
FINALIZED.json sha256:
1ed73344f675571ad8de8bee75e0bbd29e6c1b32a40f34a046642dc3bef3d67a
terminal Leaderboard sha256:
b27e2fa2a1070de81a5c6cc93dfc644cd0055d0a42b11804a03693218c0068f8
```

全部 SHA条目已重算一致。因为是 scaffold-only，terminal record正确写出
`round_evaluation_allowed=false`；它只证明 clean source下 CPU和资源门禁可复现，
没有 GPU样本或性能结论。

首个历史 dry-run为：

```text
manifest: tests/elastic/experiments/hop_local8_sm90_v1.json
run: takeover-scaffold-20260801-01
status/scope: scaffold_only/scaffold_only
reasons: Qwen PID 2753622,2753623 + dirty tree
CPU gates: 6/6 PASS
GPU stage attempts: 0
artifact: 72,207 bytes
/home pre/post: 271,249,965,056 / 271,250,386,944 bytes
SHA256SUMS sha256:
f03a48ad989ec98cea33488805296bd95ea153f195df18bb484f70607c99ad38
```

这份 `-01` artifact 早于终端提交合同，缺少 `FINALIZED.json`。其 hash 只保留历史
provenance；formal evaluator 必须拒绝，不能把它当作当前协议合规证明、正式轮证据或
晋升依据。`-02`只绑定旧的`be4a69b`，`-03`只绑定`6103180`；`-04`已补上 clean
`469e771`的 finalized证据。任何后续代码提交仍须新跑 clean scaffold，不能套用旧
source hash。

UCCL/NCCL的当前 CPU-only静态预检为
`.cache/rail_balance/competitors/preflight-20260801-08/result.json`，mode `0444`，SHA256
`8e46f41c081e7838aecfaf7a50c6a64ef4d3b73e23f9d872afb454f2d3e907d5`。它绑定 program
`3ed5e463...`、test `c17e5d3b...`、manifest `8bea6de8...`、root-owned系统 Python和
113文件 runtime closure。结果为`BLOCKED_DEPENDENCY_NANOBIND`并观察到Qwen PID
`3047501/3047502`导致`WAITING_GPU`；build、correctness、benchmark、profile和GPU
kernel均为`NOT_RUN`，future executor为`NOT_IMPLEMENTED`。这只是静态基础设施证据，
不证明RDMA runtime ready，也不是竞品结果；`-01`至`-07`均已 superseded。

协议、可复现命令和运行时 Leaderboard 见
[`HOP_AWARE_EXPERIMENT_PROTOCOL.md`](HOP_AWARE_EXPERIMENT_PROTOCOL.md) 与
[`HOP_AWARE_LEADERBOARD.md`](HOP_AWARE_LEADERBOARD.md)。竞品第一手证据见
[`COMPETITOR_EVIDENCE_2026-08-01.md`](COMPETITOR_EVIDENCE_2026-08-01.md)。当前
会话/goal hash 为 `019fb91e-606e-7a50-966c-76601d96f26a`。

## 9. planner 性能演进与目前能说什么

这里的 `finish` 是 planner transaction 的 rank-max 边界；`source stage` 才是 plan
完成后的实际本机 source shuffle adapter。不同 dirty tree 的结果只能作为方向证据，
不能直接写成正式 A/B。

| 实现阶段 | C100 diagnostic finish |
| --- | ---: |
| dense multi-target padded scan | ~1.586 s |
| sparse active-group scan | ~30.08 ms |
| tied-hot fair batch | ~12.05 ms |
| 上述规则在 singleton `rot1` 反例 | ~500.27 ms |
| bounded main rounds：volume | ~5.87 ms |
| bounded main rounds：rot1 | ~6.05 ms |
| unrestricted tiny fallback：rot1 | 20.77 ms |
| 最多 G tiny-tail rounds：rot1 | 7.24 ms |
| 最多 G tiny-tail rounds：volume | ~6.09 ms |

最后两条 source shuffle stage 分别约 152 us 和 110 us。结论只有一个：planner 仍比
真正搬数据贵一个数量级，优化重点仍是 planner；不能拿 source stage 的百微秒声称
完整方案已经快。

C100 volume 冷态 forwarding accounting：

```text
path units (direct,dst,src,third) = (0,1687,4457,2048)
source-forward units             = 6505
destination-forward units        = 27287
minimum local-forward units      = 28672
extra local-forward units        = 5120
dispatch TokenLayout bytes/unit  = 576
extra local-forward bytes        = 2,949,120
```

2026-08-01 接手审计更正：上面原始 dirty diagnostic 的 minimum 行只计入了 owner bit
zero，正确的 `minimum local-forward units` 是 `28,672`。source/destination 数字不变，
因此正确 extra 是 `5,120`。原 JSON 仅保留为被更正的 provenance，不能作为修正后证据。

当前没有 clean-tree、同 commit、10 warmup + 100 steady、matched workload 的最终
one_hop/adaptive/off 对照。下一个 Codex 必须先建立这个真值，再决定是否继续改 kernel。

## 10. 已碰过的坑与被否证方案

所有失败也写在 [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md) 与
[`HOP_AWARE_PERFORMANCE_EVIDENCE.md`](HOP_AWARE_PERFORMANCE_EVIDENCE.md)。下面是接手
时最容易重复踩的：

1. **只改 `gin.put()` 的 peer 不会换 Rail。** 当前 GPU 的 Gin context 固定属于其
   Rail；必须先把 payload 放到 egress GPU 的通信 buffer。
2. **不能按 target rank 复制网络 payload。** 会破坏 DeepEP 对同目标节点的 hidden
   payload 复用。
3. **逐 copy 热点 atomic 被拒绝。** 使用 plan prefix + static slot。
4. **dense padded mask scan 极慢。** 1.586 s；必须只扫描真实 active group。
5. **单纯删 `gap<=0` 的 batch-one 条件无效。** `rot1` 仍约 500 ms，已回退。
6. **`planner_chunk_size` 不能当 adaptive 最小 batch。** 会拒绝合法 capacity-edge
   plan；materialization chunk 与迁移配额不是一个概念。
7. **2-hop cap 是上限，不是必须花完的预算。** 之前 `required_batch` 会把两个 quota=1
   的真实热点全部拒绝；现在用最多 G 轮 tiny tail 修复。
8. **unrestricted tiny fallback 也不可接受。** 它耗满 256 轮，把 rot1 从约 6.05 ms
   拉到 20.77 ms，只换来很小的 peak 改善。
9. **完整 safe batch 不是全局最优。** CPU falsifier 找到 G3 可实现反例：one-hop load
   `[0,5,10]`，cap=3；一次搬 3 得 `[0,8,7]`，逐份重算得 `[1,7,7]`。在 207 组
   realizable cases 中 full batching 有 29 组比 granular 差、36 组更好，所以不能简单
   恢复逐份扫描。这个问题尚未解决，必须作为独立单变量实验，不要混进 tiny-tail。
10. **旧 host extension 与新 JIT identity 会导致报告拒绝。** 改 csrc 后必须 force
    rebuild；报告拒绝是正确 fail-close，不要绕过 identity gate。
11. **冷态诊断 CPU/GPU device 不一致。** rank 设置默认 CUDA device 后，`torch.arange`
    也会落 CUDA；快照已经 `.cpu()`，所以索引必须显式 `device="cpu"`。
12. **Ruff format 会把旧测试文件整体重排，制造数百行 review 噪声。** 本轮已撤掉
    formatting-only diff。`ruff check` 仍报告 benchmark 原有 E731 lambda 和 SIM105，
    不要为了本次 kernel checkpoint 顺手重构测试框架。
13. **vnode oracle 与生产 chunk size 必须一致。** 早期不一致让某 rank 先失败，其余 rank
    在 barrier 等到 timeout；这不是 GPU deadlock。
14. **proxy capacity 要覆盖 retained + moved staging。** 只按 legacy moved count 分配会
    正确返回 `CapacityExceeded`，不能把失败路径计时当性能结果。

## 11. 尚未完成的工作

按优先级排序：

### P0：收口实验控制面并等待完整 8 卡窗口

已完成的实现检查点：

1. `286da0a`：核心 sparse/tiny-tail/G2 CUDA planner；
2. `771fcc6`：fanout-aware报告与 benchmark fail-close；
3. `0f38688`：CUDA harness接受 campaign监督；
4. `448a727`：campaign supervisor、formal coordinator/evaluator与持有整轮 lease 的
   source-round executor；
5. `be4a69b`：竞品第一手证据、论文实验矩阵、协议与交接文档；分支已推送 fork；
6. `6103180`：CPU-only source-round preparer把 human spec与现场 Git身份冻结成 immutable
   manifest/plan，不创建候选、不启动 GPU；其 contracts为 10/10；
7. 最终实现树 force build、20/20 planner、CPU/API/codegen全套与四类 sanitizer均通过；
8. supervisor 20/20、preparer 10/10、coordinator 9/9、evaluator 19/19、executor 16/16
   CPU contracts，Ruff、`py_compile` 与静态审计均通过；
9. `takeover-scaffold-20260801-04` 在 clean `469e771` 上 finalized，CPU 6/6且 GPU启动
   0 次；它是 scaffold证据，不是性能证据；
10. UCCL/NCCL check-only静态预检 26/26 CPU contracts通过；`preflight-20260801-08`
    fail-close于缺失 nanobind和非独占8卡，竞品 build/GPU仍为`NOT_RUN`；
11. `469e771`提交并推送 UCCL/NCCL预检与论文证据，推送后 ahead/behind为`0/0`。

仍需完成：

1. 只有完整 8 卡均空闲时，先重跑两个 4×2 vnode；
2. 从当前最快且完全正确的 sealed parent生成 2～4 个单变量 CUDA候选，以
   `prepare_rail_balance_source_round.py` 冻结 source manifest/plan，
   再由 live executor执行正式 `4+4N` source round；不得为测试控制面伪造候选提交；
3. 无完整8卡窗口时继续实现 competitor future executor与 COMMON_FAIR adapter，但不得
   越过依赖、source、磁盘和GPU lease门禁；
4. 保留 capability=false，直到真实多机门槛另行通过。

不要大规模 rebase 历史提交。若以后整理 review branch，必须先保留 archive或可恢复 tag。

### P1：当前算法的 clean 性能证据

1. 全部 8 张 GPU 无任何 compute进程时，固定同一 commit、shape、seed；
2. one_hop 与 adaptive 使用完全相同 records；
3. volume、rot1、mesh、balanced/singleton control；
4. 10 warmup + 100 steady；
5. 保存全部 sample、median/p95/p99、rank-max；
6. profiler 关闭的结果才是性能真值；
7. 当前 dirty diagnostic 不进入正式结论。

### P2：Nsys 后再决定下一次 kernel 修改

此前 Nsys 已证明 multi-target decision 是关键路径。当前 sparse/tiny-tail 改完后必须
重新采相同窗口，确认：

- decision kernel 仍占多少 exposed time；
- active-group scan、轮数还是 materializer 成为新瓶颈；
- planner 是否可以与已有 gate/notify 更早 overlap；
- source shuffle 是否仍只是百微秒级。

只有 Nsys 仍证明某个 kernel 位于 critical path，才运行 NCU；禁止直接对所有 kernel
`--set full`。性能工作遵循本机 skill：

```text
/home/chen/.local/share/codex-profile-manager/profiles/wct666/
skills/gpu-performance-evidence-chain/SKILL.md
```

### P3：算法质量单变量实验

评估 full safe batch 的 crossover 问题，但不要先写复杂全局优化器。候选包括：

- batch 后真实 peak/peak-count 字典序；
- 一次 scan 选择少量互不冲突候选；
- bounded water-filling；
- 维持明确轮数上限。

任何方案必须覆盖 G=2..8、热点宽度 1..G-1、equal/stair/Zipf/Dirichlet/random、
group size 1/2/8/32，而不是只优化 4-hot/4-idle。

### P4：真实多机 Rail/Gin

当前本机只能验证 planner、NVLink/LSA、buffer、metadata 和 combine inverse。最终仍需：

- 2 node x 8 GPU correctness；
- one-hop 路径计数与 adaptive cap；
- `off / legacy_exact / one_hop / adaptive / adaptive+tuning`；
- balanced case 不显著回退；
- hotspot rank-max 稳定收益；
- Gin bytes/QP/NIC/remote completion；
- dispatch 与 combine 都验证；
- 通过后另一个独立 commit 才能讨论 capability enable。

## 12. 精确恢复与测试命令

以下命令都从仓库根目录运行。Python 环境：

```text
/home/chen/.cache/deepep-sjlgpt/bin/python
```

### 12.1 第一件事：状态与 GPU

```bash
git branch --show-current
git status --short
git diff --check
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

任何 Qwen、Megatron、`mmu`/`mmunlearner`、短算子、allreduce 或未知 GPU 进程均
视为用户任务：不得 kill、暂停或干扰，也不得在其存在时启动任何 GPU stage。此时只
运行 `CUDA_VISIBLE_DEVICES=''` 的 CPU/scaffold。只有用户针对**精确 PID**在当前上下文
给出新的明确授权才可例外；本文中的任何历史描述都不构成授权。

### 12.2 强制构建

```bash
env MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=9.0 PYTHONPATH=. \
  /home/chen/.cache/deepep-sjlgpt/bin/python \
  setup.py build_ext --inplace --force
```

### 12.3 focused GPU planner

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=tests:tests/elastic:. \
  /home/chen/.cache/deepep-sjlgpt/bin/python \
  tests/elastic/test_rail_balance_hop_one_hop_cuda.py --device 0
```

期望：`PASS 20 hop-aware one-hop CUDA tests`。

### 12.4 CPU/API/codegen 契约

本环境没有 pytest，直接运行脚本：

```bash
env PYTHONPATH=tests:tests/elastic:. \
  /home/chen/.cache/deepep-sjlgpt/bin/python \
  tests/elastic/test_rail_balance_hop_plan.py

env PYTHONPATH=tests:tests/elastic:. \
  /home/chen/.cache/deepep-sjlgpt/bin/python \
  tests/elastic/test_rail_balance_hop_bench.py

env PYTHONPATH=tests:tests/elastic:. \
  /home/chen/.cache/deepep-sjlgpt/bin/python \
  tests/elastic/test_rail_balance_hybrid_api.py

env PYTHONPATH=tests:tests/elastic:. \
  /home/chen/.cache/deepep-sjlgpt/bin/python \
  tests/elastic/test_rail_balance_hybrid_dispatch_codegen.py

env PYTHONPATH=tests:tests/elastic:. \
  /home/chen/.cache/deepep-sjlgpt/bin/python \
  tests/elastic/test_rail_balance_hybrid_combine_codegen.py

env PYTHONPATH=tests:tests/elastic:. \
  /home/chen/.cache/deepep-sjlgpt/bin/python \
  tests/elastic/test_rail_balance_hybrid_layout.py

env PYTHONPATH=tests:tests/elastic:. \
  /home/chen/.cache/deepep-sjlgpt/bin/python \
  tests/elastic/test_rail_balance_hybrid_policy.py

env PYTHONPATH=tests:tests/elastic:. \
  /home/chen/.cache/deepep-sjlgpt/bin/python \
  tests/elastic/test_rail_balance_hybrid_public_lifecycle.py
```

### 12.5 4+4 vnode

```bash
env EP_DISABLE_GIN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  PYTHONPATH="$PWD/tests/elastic:$PWD" \
  /home/chen/.cache/deepep-sjlgpt/bin/python \
  tests/elastic/bench_rail_balance_hop.py \
  --backend vnode --mode one_hop --case offdiag_hot \
  --num-nodes 2 --gpus-per-node 4 --tokens-per-rank 16 \
  --topk 1 --num-experts 16 --hidden 256 \
  --output-json /tmp/rail-hop-vnode-onehop.json

env EP_DISABLE_GIN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  PYTHONPATH="$PWD/tests/elastic:$PWD" \
  /home/chen/.cache/deepep-sjlgpt/bin/python \
  tests/elastic/bench_rail_balance_hop.py \
  --backend vnode --mode adaptive --case diag_hot \
  --num-nodes 2 --gpus-per-node 4 --tokens-per-rank 16 \
  --topk 1 --num-experts 16 --hidden 256 \
  --max-two-hop-percent 50 --hop-penalty-percent 0 \
  --output-json /tmp/rail-hop-vnode-adaptive.json
```

### 12.6 新 tiny-tail 的 sanitizer

把 `<tool>` 分别替换为 `memcheck`、`synccheck`、`initcheck`、`racecheck`：

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=tests:tests/elastic:. \
compute-sanitizer --tool <tool> --error-exitcode=99 \
  /home/chen/.cache/deepep-sjlgpt/bin/python -c \
  'import torch; from test_rail_balance_hop_one_hop_cuda import test_adaptive_tiny_tail_treats_two_hop_cap_as_a_limit; torch.cuda.set_device(0); test_adaptive_tiny_tail_treats_two_hop_cap_as_a_limit(); print("PASS")'
```

### 12.7 C100 checked-adapter diagnostic

```bash
env PYTHONPATH=. \
  /home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/bench_rail_balance_hybrid_lsa.py \
  --stage source --warmup-iters 3 --steady-iters 20 \
  --hop-mode adaptive --watchdog-seconds 900 --timeout 180 \
  --case-name c100_volume_h256 \
  --json-out /tmp/c100-volume-adaptive.json \
  --master-port 29881
```

正式收集必须改为 `--warmup-iters 10 --steady-iters 100`，用唯一端口，分别跑
`one_hop` 和 `adaptive`，并保证相同 case/seed/commit/GPU 状态。`rot1` control 的 case
名是 `c100_matrix_rot1_h256`。

### 12.8 Ruff 与语法

```bash
/home/chen/.cache/deepep-sjlgpt/bin/python -m py_compile \
  tests/elastic/bench_rail_balance_hybrid_lsa.py \
  tests/elastic/test_rail_balance_hop_one_hop_cuda.py

/home/chen/.cache/deepep-sjlgpt/bin/ruff check \
  tests/elastic/bench_rail_balance_hybrid_lsa.py \
tests/elastic/test_rail_balance_hop_one_hop_cuda.py
```

此外，原始交接当时新增：

```text
M  docs/rail_balance/HANDOFF.md
?? docs/rail_balance/HANDOFF_HOP_AWARE_2026-08-01.md
```

上述 pre-`286da0a` 九个核心 dirty 文件相对 `ee80a16` 的历史 binary diff SHA256 为：

```text
d48e542594ed4c4f31a78c12ccc7f97f7353a02a880c855ede4b3bf6d2dab5e0
```

该 hash 仅用于审计历史，当前实现已提交，**不得**据此恢复旧 diff 覆盖
`286da0a` 及后续工作。

Ruff 当前会报告旧 benchmark 的 11 个 E731 与 2 个 SIM105。不要执行全文件 `ruff
format` 后把几百行纯格式改动混进 kernel commit。若要清 lint，请单独提交。

## 13. 多机入口

统一入口已经存在，reference/vnode/multinode 使用同一 workload 与 JSON schema：

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  WORLD_SIZE=2 RANK="$NODE_RANK" \
  MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
  OMP_NUM_THREADS=1 EP_DISABLE_GIN=0 \
  PYTHONPATH="$PWD:$PWD/tests/elastic" \
  /home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/bench_rail_balance_hop.py \
  --backend multinode --mode one_hop --case offdiag_hot \
  --num-nodes 2 --gpus-per-node 8 --tokens-per-rank 4096 \
  --topk 4 --num-experts 128 --hidden 7168 \
  --warmup-iters 10 --steady-iters 100 \
  --output-json /shared/rail-hop-multinode.json
```

每个节点只启动上面命令一次。统一入口会调用下游 runner，而下游自己为本机 8 张卡
spawn worker；不要再套 `torchrun --nproc-per-node=8`，否则每个 torchrun rank 会再次
spawn 8 个进程，并把全局 rank/world 错当节点 rank/count。

上集群前读取并使用现有 DeepEP Rail/Gin 环境变量、NIC mapping 和 launcher 检查；
不要在本机伪造 NIC 性能。真实结果必须 rank-max、paired A/B，并记录所有 rank 的 Git、
extension、GPU、NCCL、NIC 与环境一致性。

## 14. Git 与远端处理

```text
origin = deepseek-ai/DeepEP
fork   = CyberSecurityErial/DeepEP
```

当前远端有：

- `fork/feat/rail-balance-prototype`；
- `fork/review/rail-balance-prototype`；
- prototype archive；
- `fork/feat/rail-balance-hop-aware`，本地分支已设置为其 upstream。

`6103180` 检查点已推送且当时 ahead/behind 为 `0/0`。每个后续本地提交及 clean
scaffold文档更新完成后执行：

```bash
git push fork feat/rail-balance-hop-aware
git rev-list --left-right --count fork/feat/rail-balance-hop-aware...HEAD
```

只有第二条输出 `0 0` 才能写“已发布”；恢复时仍须现场核对，不能套用旧检查点状态。

用户说明 `humor`、`WenhaoHe02`、`CyberSecurityErial` 都是自己团队成员，review 时不要把
他们的提交当作第三方可疑来源。

不要删除 `/tmp` 下已有的 prototype review worktree，除非先确认不再被使用：

```text
/tmp/deepep-rail-review-layered.4hshtV
/tmp/deepep-rail-review.xoBbBr
```

## 15. 环境与用户工作方式

- 用户指定目标机器为单机 8xH200；当前驱动公开名显示 `NVIDIA L20X`、CC 9.0、
  143771 MiB/GPU。按用户约定把它作为 H200/SM90 目标，不要重新质疑硬件；报告中可同时
  保留机器原始 manifest。
- Driver `570.172.08`，PyTorch `2.11.0+cu128`，CUDA runtime `12.8`，NCCL
  `(2,28,9)`。
- Nsys `2024.6.2`；NCU `2025.1.1`；Compute Sanitizer `2025.1.0`。
- 用户要求高性能计算代码高内聚、低耦合、唯一事实来源、直接报错、无 silent fallback，
  尽量少分支/少元数据/少动态分配。
- 测试代码不必过度设计；核心 kernel 必须简练并能审计。
- 遇到 bug 先用最小反例 debug，不要为尚未出现的问题堆门控和脚手架。
- 每个优化保留假设、失败、命令、结果和 artifact；commit 粒度适中，方便审计。
- 性能判断必须走：无 profiler 真值 -> Nsys critical path -> 必要时 NCU -> 单变量修改
  -> 无 profiler 复测。
- 任意 GPU 有 compute进程时只做 CPU/scaffold；不要启动 GPU 功能、sanitizer、benchmark
  或 profile，更不得终止用户任务。
- 用户会间歇断网/关 Mac；每次暂停前都要写日志、留 Git checkpoint 并说明恢复命令。

## 16. 完成定义与绝对禁止事项

目标完成至少需要：

- endpoint-aware Python/GPU planner 对齐；
- one-hop 从不选择第三 Rail；
- adaptive 先 one-hop，再受限 third Rail；
- target endpoint 不变、payload 不复制；
- direct/source/destination/third 路径与 combine inverse 正确；
- capacity/metadata/world mismatch fail closed；
- default off 与 legacy 不回归；
- reference/vnode/multinode 共用 schema；
- clean single-node exhaustive correctness、sanitizer 和性能证据；
- 真实多机 correctness 与 off/A1/A2/A3/A4 对照；
- capability 只有在真实门槛通过后另行解锁。

绝对不要：

1. 实现专家迁移；
2. 把任意全 Rail 再分配叫 one-hop；
3. 丢失 target mask；
4. 把 2-hop 当默认全局精确均衡；
5. 为漂亮柱状图追求完全等分；
6. 添加第二套 combine planner；
7. 恢复 per-copy global tail atomic；
8. 用 vnode/单机结果声称 NIC/RDMA 加速；
9. 绕过 capability gate；
10. 用一次 profiler 运行或平均延迟声称优化有效；
11. 在未保存 dirty checkpoint 前重写 43 个历史提交。

## 17. 下一位 Codex 的第一小时

按这个顺序继续，不要在没有新测量时随机改 CUDA：

```text
1. git status / git log / git diff --check，现场确认 286da0a、771fcc6、0f38688、448a727、
   be4a69b及其后续 authoring提交。
2. 运行 campaign、source preparer/coordinator、formal evaluator、source executor 的全部
   CPU/静态门禁。
3. clean tree运行 fresh finalized scaffold并记录 hash；push fork并核对 ahead/behind=0/0。
4. 检查所有 8 张 GPU，任一 compute PID存在就只继续 CPU/论文脚手架。
5. 完整 8 卡空闲后先重跑两个 vnode；依据新 profiler-free/Nsys事实生成2～4个单变量
   candidate commit，由 preparer冻结 human spec，再执行正式 4+4N source round。
6. 用 profiler-free paired block选候选；只有新 Nsys证明瓶颈后才生成下一轮 CUDA候选。
7. 多机资源到位后完成真实 Gin/RDMA correctness、NIC counters和 matched基线。
```

一句话概括当前状态：

> 语义、核心代码和单卡/静态正确性闭环已在 `286da0a` 收口，但最后一个正式性能
> Leader仍是 `8c56939`；当前没有完整 8 卡窗口，也没有新 source-version性能证据。
> 先完成可审计脚手架、fresh finalized scaffold和 push，随后只在全 8 卡空闲时跑
> vnode与正式配对轮；真实多机 Rail/Gin仍是最终可用性门槛。
