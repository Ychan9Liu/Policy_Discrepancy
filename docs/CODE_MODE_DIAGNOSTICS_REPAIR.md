# 03 · mode 诊断归约与旧日志兼容修复

日期：2026-10-20，Asia/Shanghai。交接基准 `d8124f9157dbddf653354ba98bc0b90ca4debe58`；05 原分析/测试 `2427710a7a89f2a5c90a08634a74a7a11ee87139`。原16条训练提交 `48d71f1541721208322442f5a1d5977124f3e888` 不变。

**已完成诊断修复和 CPU 验收；原16条正式 return 指标不变。诊断错误不是已证明的 hopper 失败根因。** gate 选择作用与输入微扰/探索路径的因果问题仍未解决。本轮没有训练、环境仿真、优化器更新、GPU 作业、W&B 写入或科学协议变更。

## 1. 字段语义与兼容边界

两路 mode 均沿用原 categorical/unimix/argmax one-hot，形状 `[B,T,S,C]`。仅修改 overlay 返回的诊断，D、w、实际 rep 权重、free-nats 顺序、loss reduction、梯度隔离和 shuffle key 均保持原实现。`agent.py`、RSSM、配置和协议执行器未修改。

| 字段（前缀 `train/dt/`） | 旧无版本日志 | 新 `mode_diagnostic_version=2` |
| --- | --- | --- |
| `mode_coordinate_diff_frac` | 未记录；可由旧 factor 字段精确映射 | `mean(qmode != pmode)`，坐标比例 |
| `mode_factor_diff_frac` | 实际为坐标比例，命名错误 | `mean(any(qmode != pmode, C))`，因子比例 |
| `mode_vector_diff_frac` | 实际为因子比例，命名错误 | `mean(any(any(qmode != pmode, C), S))`，整向量比例 |

因子比例在全部有效 `[B,T,S]` 上平均；整向量比例在全部有效 `[B,T]` 上平均。无新增 reset/terminal mask。该版本号只标记 mode 诊断语义，不改变评价日志的 schema v2 或冻结科研协议 v1。

选择修正原字段并新增坐标字段/版本号，而不让错误名字继续代表错误的量。因此**不能跨修复 SHA 直接拼接同名曲线或默认沿用仪表盘解释**；须按提交及版本分开，再映射为同一语义。旧日志、W&B、原表和原报告的历史源码片段全部保留；没有补写旧整向量值。

`analysis/mode_diagnostics.py:normalize`：无版本记录必须由调用者明确确认是已审查 legacy 来源；将旧 vector→真实 factor、旧 factor→真实 coordinate，真实 vector 保持 `null`/不可用。版本2读取三个正确字段；未知/混合版本、缺字段、非法比例拒绝。生产旧产物审计只接受已审查的原训练 SHA 和原16条矩阵，核对镜像 SHA256。

对合法 one-hot，factor 比例 = coordinate 比例 × C/2；这是坐标/因子的确定关系，**不能据此恢复整向量比例**。例如同为因子比例 .5，可以把两个差异放在同一时刻（整向量 .5），或分在两个时刻（整向量1）。旧的 `analysis/analyze_m2_v1.py` 及 hopper 专项脚本保留历史复算用途，其 raw 字段须按本表解释；新兼容审计产生独立、明确命名的表。

## 2. 实际版本、测试和容差

实现/测试提交及所有本轮实际执行版本：
`1ccde38c042d2264b5d5d1e6c93c658d170a9d88`。
最终交付 SHA 见本轮 `analysis/outputs/mode-diagnostics-repair-20261020/DELIVERY.json`、HANDOFF 和最终回复；该测试提交之后只补说明文档及 STATUS，不改变代码/测试。

本机 Python 3.12.14 运行23项统计/兼容/既有分析测试，通过。sv1 现有 dreamer 环境 Python3.11.16、JAX/JAXlib0.4.33、NumPy1.26.4 运行37项测试，通过，无失败/跳过。显式 `CUDA_VISIBLE_DEVICES=''`、`JAX_PLATFORMS=cpu`，两虚拟 CPU 设备覆盖已有跨 shard 置换回归；没有使用物理 GPU，也没有查当前 GPU 占用。

新增7项测试（3兼容、4 JAX）包括：

- `[1,2,2,32]` 只在一个时刻的一个因子不同：坐标 `.015625`、因子 `.25`、整向量 `.5`；全同、全异；`[2,3,2,4]` 跨 batch/time 的归约；argmax 并列按原路径处理。
- 相同因子比例但不同整向量比例的反例，证明旧聚合量不足以恢复整向量。
- 四模式 logging/Dt/constant/shuffle 的冻结输入比较。测试从受信任 Git 对象直接加载基准 `d8124f9:rep_probe.py`，同参数、同输入、同 seed：D/w、实际权重、所有非修复诊断、fixture 的 rep 与附加对照目标及其对 actor scale、posterior/prior logits、history 的梯度均 **逐位相同，容差0**。
- fixture 存在 raw rep KL>1、非零 posterior 梯度，非 logging 权重<.999，避免只测零作用位置。alpha20、c=.6、测试 key19/shuffle seed7/step3 仅工程 fixture；不是新增正式 c、训练 seed 或预算。新增布尔诊断对上述输入梯度精确为零。
- Ninjax 诊断前后 key 与修复前及无 probe 的控制函数精确一致；shuffle 的已有重放、多重集合、实际换位、跨 shard 测试保留并通过。没有重新定义或消耗训练 RNG。

这些是 **CPU 冻结 probe/loss/梯度 fixture**，不是本轮完整 agent 更新、真实 DMC 或 CUDA/bfloat16 验收。此前 CODE_PHASE1/2 及后续真实工程验收保留其原版本和范围，本轮没有重跑训练来扩大证据。

完整命令见 outputs/COMMANDS.md。主要服务器命令：

```sh
cd /data/Policy_Discrepancy/repo
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /data/Policy_Discrepancy/envs/dreamer/bin/python -m unittest tests.test_mode_probe_regression tests.test_mode_diagnostics tests.test_rep_probe tests.test_rep_overlay tests.test_m2_analysis tests.test_return_metrics -v
```

## 3. 既有产物只读补证

本机新目录 `analysis/outputs/mode-diagnostics-repair-20261020/`：

- `legacy-audit-1ccde38/mode_intervals.json`/CSV：16运行×20区间=320行。保留原字段、实际训练 SHA、原路径/行号/hash，并另列真实坐标/因子比例和版本；320行整向量比例均为 `null`。核对 C=32 时两旧字段16倍关系，预先使用 float64 日志聚合容差 `atol=rtol=1e-12`，不是训练容差。
- `return-audit-1ccde38/`：使用既有正式 RUN_INDEX、原统计器 `--require-complete` 复算，退出0。3360 episode、336评价点、16条正式 seed 指标；`tables.json` 与05原始正式复算表**逐字节相同**，SHA256 `3d02f5cca66644877e55e45a79fd160949ff9b8d72c45920bf09e1235d75e9ef`。未改变 AUC、尾部五点、任务/总体或三个 Dt 差值。
- `validation.json`：原 COLLECTION 记录的1161个镜像文件全部仍匹配原 hash，包含旧日志、配置、收据、评价与审计小标记。没有覆盖原分析目录。本轮只读本机已采集产物，没有重新采集或修改 sv2 原训练目录。
- `tests-local-1ccde38.log`、`tests-sv1-1ccde38.log`、COMMANDS、Git bundle、DELIVERY 和 HANDOFF。

sv1 证据：`/data/Policy_Discrepancy/runs/mode-diagnostics-repair-20261020-cpu/`。本机提交后经 Git bundle 校验、快进同步 `/data/Policy_Discrepancy/repo`，无服务器源码直接修改。测试 bundle SHA256 `fc75016863691768650b4ba42e3c71fe9472e25c403534e80f666108014d1131`；最终文档也通过 bundle 快进。sv1 原版本27261到测试版本包含此前已提交的项目历史，并非这些历史均在本轮新增。未推送 GitHub；sv2/sv3 未同步或访问。本机原 RESEARCH/STATUS 修改与未跟踪 docs/proposal 保留，仅本轮 STATUS 新段入提交。

## 4. 解释限制与交接

修复解决诊断归约/命名错误，不改变原16条训练、c 或正式指标。原始 aggregate 日志不足以恢复逐时刻整向量值；新代码的正确值也不能事后冒充原运行记录。没有模型推理或逐位置梯度补采，更未控制初始图像进行反事实训练；故不能从本修复裁决 gate 与原输入微扰的因果关系，也不能把本错误定为 hopper 失败根因或把工程通过写成 M2 有效。

**首选下一任务：用户继续现有02，审查下一项因果辨别复验的协议。** 可立即阅读已有证据、讨论冻结输入反事实/配对观测/重复方案；具体干预、任务/组、seed、预算、判据和资源由02与用户确认，再按职责交03实现、04执行。冻结 v1 不重开，alpha/c/梯度不自动改变；没有新的训练授权，不自动建 chat 或发消息。
