# 03 · size50m CUDA 报告、重跑与恢复边界修复

更新：2026-10-09。事实来源为本仓库确定提交和 sv1 原始产物；这是工程验收，不是 M2 方法效果。承接基准 `d9be5574878d2c3f117cc6667be844f9a527eb84`，04 主运行 `1393548fb6e46d28f92a834204f7b09495a3eab1`、补验 `63d36a111d247aa8cd05af173f74e43e4e35d2e1`。本轮代码和测试最后实际验收提交为 `20ab3486bfaa067cc7950d3f5c8408fe2ed92ec3`；文档提交只更新记录，最终 HEAD 见 Git 与交接回复。

## 范围与冻结条件

- 保留 clean、仅视觉、四任务、size50m、训练 seed0、alpha20、free_nats1、dyn/rep=1/0.1、ac_grads=false、reward_grad=true、repval_grad=true、四组。未改科学协议、算法 gate 或 transfer guard 全局策略。return/score 由 02 审议；没有正式百万步运行、正式 c 或科研结论。
- 仅 sv1 用于本轮测试。代码先在本机提交，再经校验 Git bundle 快进到 `/data/Policy_Discrepancy/repo`；运行前 supervisor 检查 GPU0 `memory.used=0`，CUDA_VISIBLE_DEVICES=0，使用 `/data/Policy_Discrepancy/envs/dreamer`、EGL、A100、bfloat16。没有在服务器直接修改项目代码或覆盖 04 产物。
- 新产物根目录 `R=/data/Policy_Discrepancy/runs/engineering-size50m-repair-20261009-98225e1`。各完整 agent 运行有 `execution-initial.json`、`config.yaml`、`protocol_manifest.json`、`stdout-initial.log`、`resources-initial.jsonl`、收据、评价、checkpoint；失败的首次 fixture、原 04 故障目录仍保留。

## 修复语义

1. `report_on_batch` 接受刚由训练实际消费的**主机 NumPy batch**，先按 `report_length + replay_context` 在主机切片，再通过 `internal.device_put(..., agent.train_sharded)` 显式传到 CUDA；独立 `report` seed/key 放到 `train_mirrored`。不再对设备上 JAX array 做隐式 `[:, :length]` 传输；不重新抽 replay batch，不推进 train policy/update/replay RNG。传入设备 batch 直接报错。报告发生在训练动作网格边界，本轮原配置 `4098/2049` 跑了两次报告。
2. 恢复开始时核对 checkpoint 的 update/evaluation ledger 与 JSONL，拒绝额外、缺失或重复的持久化 ID；即使 checkpoint 动作数为零，只要是已有不完整 checkpoint 也拒绝续跑，因为环境与 replay 运行态不能无缝恢复。完成运行重开核对 final_state 和 matching_result 后只返回，不追加训练、评价或收据。正常 replay 重采样仍照常参与每个新优化更新；`ProtocolState.record_update` 对同 ID 同值返回 False、冲突报错。重复持久化行一律拒绝，避免“看似去重”掩盖损坏。
3. 仅在 `engineering_fixture` 中可启用动作、更新和物理状态追踪。物理状态写作 `log/physics_state`，不进入 agent 观测或 replay。正式模式拒绝追踪开关。更新追踪保存实际消费 batch（排除易变 `stepid` UUID）、train seed、carry 和更新前参数的哈希；动作追踪逐转移保存图像、动作、奖励、物理状态等哈希。追踪只作定位，不改变已冻结科学设置。
4. 工程运行的报告前后对参数、训练及评价 counters、消费 batch、replay sampler RNG 与 replay metrics 逐项比对；不同即停止。评价前后对参数、训练 counters、replay RNG/metrics 比对，单独允许评价 counter/seed 变化；不同即停止。通过的指纹分别写入 `engineering_report_audit.jsonl`、`engineering_evaluation_audit.jsonl`。训练 policy seed 由 seed0+训练动作调用 counter 派生，训练更新 seed 由训练 batch counter 派生；评价有独立 counter/seed 命名空间，报告有独立命名空间。审计覆盖这些未来训练 key 的来源和 replay RNG；不声称完整环境/replay 状态可恢复。

## 真实 CUDA 和回归证据

| 验收 | 实际提交、配置及产物 | 结果 |
| --- | --- | --- |
| 原异常路径 | `98225e186f2c0b039c2df1cf095d9b0b83197972`，`R/report-real`；真实 hopper size50m，batch16×64、16训练环境、train_ratio256；4098 动作、2049 间隔、`report_every_actions=2049`、每点1完整 episode、窗口 `[2200,3600)` | 子进程退出0；4098训练动作、517更新、3点评价、2次报告、2个报告 checkpoint counter；CUDA峰值8088 MiB。旧外层检查误找被 logger filter 排除的 `report/` JSONL 行而返回1，子进程及 checkpoint 确认报告成功。该检查错误不当作训练失败。 |
| 报告及评价只读 | `20ab3486bfaa067cc7950d3f5c8408fe2ed92ec3`，`R/report-and-eval-audit`；真实 hopper size50m，2048动作/间隔、每点1完整 episode、窗口 `[2032,2048)`、2048动作点1次报告 | 子进程退出0；2048动作、5更新、2点评价 `[0,2048]`、2000评价动作、报告ID0。`engineering_report_audit.jsonl` 1行与 `engineering_evaluation_audit.jsonl` 2行只有逐项相等后才写入；参数、训练计数、replay RNG/metrics 与报告消费 batch 未变。峰值8088 MiB。此短预算只验工程路径。 |
| 完成重开 | 同一提交、同一真实运行 `R/report-and-eval-audit` | 重新执行原命令退出0；`reopen_audit.json` 中收据、评价、matching、final、两类审计文件前后 SHA-256 全相同，无追加或覆盖。 |
| 原零动作故障 | 只读 04 原目录 `.../engineering-size50m-20261009-1393548/dmc_hopper_hop/isolation`；输出 `R/original-orphan-audit.json` | 原 checkpoint 动作0/更新0，已有成功收据 ID0..4。新审计报 `extra=[0,1,2,3,4]`，原 checkpoint/收据/评价哈希前后相同。旧 manifest 提交不同，未伪装成对原目录实际续跑；同提交完成重开与单元 fixture 分别覆盖 runner 边界。 |
| 相关回归 | sv1：`CUDA_VISIBLE_DEVICES=0,1 /data/Policy_Discrepancy/envs/dreamer/bin/python -m unittest tests.test_rep_probe tests.test_rep_overlay tests.test_protocol_v1 -q`，实际 `20ab3486...` | 22 tests OK，含保留的 off/logging/overlay/梯度、CUDA transfer guard、零动作多余收据、重复持久化行、评价/报告指纹。此前 `98225e...` 19 tests OK，`dc87e...` 21 tests OK。真实 size50m 运行另列，不用这些 fixture 替代。 |

## hopper 同配置分歧的定位

- 04 的 4098 动作同配置重跑最早**收据**差异在 update2/动作2040；收据相同不证明此前实际输入相同。`dc87e627e66f864e9f0df4257e942181510c3cb6` 的两次独立 2048 动作真实 hopper CUDA 追踪 `R/trace-first`、`R/trace-repeat`，逐转移首个差异在动作31的图像，早于任何训练更新。动作、奖励和训练 seed hash 一致；update0 前参数和 seed/carry 相同，首个训练 batch 的图像不同，update1 前参数已不同。
- 为核查真实环境物理状态，`c22377572193c7385f63481af209261e9f8926c2` 的 `R/physics-first`、`R/physics-repeat` 各跑2048训练动作和5更新，均退出0。`R/physics-comparison.json` 对两次各2064条转移显示：**只有6条图像哈希不同**，第一条是 action_step0/worker1 reset；所有物理状态、动作、奖励、is_first/is_last 等哈希相同。update0 开始动作2032的输入 batch 只有 `image` 不同，训练 seed、carry、更新前全部参数相同；update1 前112个参数键已不同。5次更新的 train seed 哈希全同。第一处输入像素差异先于报告和更新，不能归因于评价或报告随机流消耗。
- `9964689df10a5a1f213cc4c7d6af574cbfd541af` 的独立 DMC/EGL 探针 `R/dmc-render-probe` 两进程同 seed 3988856348、同64个零动作：65帧中 frame11 恰好一个 uint8 值差1，物理状态和奖励逐位相同，见 `R/dmc-render-comparison.json`。这复现了固定物理轨迹下的渲染像素不稳定。
- `b4a78adcdc4bbb5b9e2dd414edb5fdc25b920eb4` 的冻结输入真实 size50m CUDA 计算 `R/fixed-input-v2`，两进程同输入、同初始参数、4次更新逐次损失/参数/匹配值完全相同，见 `R/fixed-input-comparison.json`。首版 fixture 配置覆盖方式报错，原失败产物 `R/fixed-input/first` 保留；修复后结果取 v2。
- **技术结论**：在本机 sv1 DMC/EGL 和现有配置下，视觉渲染可在相同物理状态/动作下产生像素差异；真实同配置训练的首次分歧是视觉输入，之后通过优化更新放大。已排除这组证据中的固定输入计算不确定性与已审计的评价/报告训练 RNG 消耗作为首因。并不保证所有机器/驱动环境逐位复现，也未实现像素确定性修复；不事后放宽比较容差来宣称轨迹相同。

## 同步、限制和 04 补验条件

- 本轮本机提交链为 `79fef3a...`、`98225e...`、`891732a...`、`dc87e...`、`b7418c...`、`9964689...`、`b4a78a...`、`d7fa341...`、`c223775...`、`659b607...`、`20ab348...`。各实际运行前服务器 checkout 干净且与记录提交一致。最后代码/测试验收版本为 `20ab3486bfaa067cc7950d3f5c8408fe2ed92ec3`；后续文档提交不改变测试语义。bundle 在 `R/*.bundle`，逐次 `git bundle verify`、SHA-256 本机/服务器核对后快进；最后同步状态见最终交接。
- `R/report-real` 首次外层脚本误判只影响验收包装器，不影响报告子进程。完成重开只说明已有完整产物不重复写；未知崩溃后的环境、replay 运行态仍无法无缝恢复，必须独立重开。原故障目录保留只读，没有在旧提交上新跑。短预算与冻结输入不是正式100万步任务效果证据。
- 04 可立即在**现有 04 chat**按本报告从已同步确定提交做独立工程补验：先核对 GPU/checkout/完整配置，在新 runs 目录对原 `4098/2049` hopper 报告开启路径与必要随机流审计做最终质量核查，记录图像非逐位确定性的已知边界；若其他三任务或模式需补验，列明目的和判据，勿沿用工程 c 作为正式 c。02 的 return/score 最终口径以及用户的正式运行授权仍是正式比较的独立前置条件。
