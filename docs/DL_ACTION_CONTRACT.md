# DL 动作接口与来源重建审查

版本 `DL-action-contract-review-r1`，2026-10-11。本文是用户总体授权内的 Agent 独立审查及必要工程决策记录，**不是方法有效结论或用户逐项确认的新科学参数**。根 Agent 已停止基于旧八条 calibration 来源冻结反馈协议；旧来源反馈未启动。`DL-source-r1` 来源配置已冻结，待确定代码 SHA、完整工程验收与实际资源记录后执行，当前尚未运行。反馈诊断尚未冻结；本文不修改 gate 或授权训练。

## 1. 已发现的接口事实

正式入口 [main.wrap_env](../dreamerv3/main.py) 的实际动作调用顺序为外层 `ClipAction`，再经 `CheckSpaces`、`UnifyDtypes`、`NormalizeAction`，进入 DMC/FromDM，最后传给 suite。模型输出的 Gaussian 样本是 raw float32；`Agent.policy(mode='eval')` 仍采样，并将该 raw 样本保存在自己的 carry，**没有把采样动作替换为 mean 或物理执行值**。

原 [dl_diagnostic.collect/true_consequence](../scripts/dl_diagnostic.py) 则直接调用 `suite.step(raw)`。旧采集记录的 raw Gaussian 动作没有经过上述生产映射；旧后果复放忠实保留了这一 direct-suite 对象。不是记录文件被损坏，也不能通过修改旧动作数组使其变成生产来源。

真实 quadruped CPU action spec 为 float64、12 维，四条腿各重复以下 bounds；这些值由实际 suite spec 读取，不能以模型 action space 的 `[-1,1]` 替代。

| 每腿关节 | suite low | suite high |
| --- | --- | --- |
| yaw | −1 | 1 |
| lift | −1 | 1.1 |
| extend | −.8 | .8 |

源码 [wrappers.py](../embodied/core/wrappers.py) 定义的映射为：

```text
raw = actor.sample(private_or_production_key)          # float32
clipped = np.clip(raw, -1, 1)
unified = np.asarray(clipped, normalized_space.dtype)  # float32
applied = (unified + 1) / 2 * (high - low) + low
applied = np.where(finite_bounds_mask, applied, unified)
```

`high/low` 来自原 suite spec。前段运算在 float32 上进行，后段与 float64 bounds 运算得到 float64；必须按生产源码逐步保留实际 dtype，不能代数化简或先统一为 float64。即使 bounds 为 ±1，也不能假定逐位 identity。lift 的归一化零动作对应物理输入 .05，extend 的比例为 .8，因此差异不只是超界样本被裁剪。MuJoCo 的内部 control clamp 不能证明两种接口、完整 integration state 或收益等价。

[RSSM._core](../dreamerv3/rssm.py) 自身对动作有 `a/max(1,abs(a))` 处理，但这不是环境端的各关节映射，不能修复 legacy source 的物理轨迹。

## 2. 旧证据保留到什么范围

**已验证事实仍保留**：旧 checkpoint/config/来源 hash、原数组数值、完整 direct-suite 物理复放、冻结模型状态收据、TwoHot 与概率混合复算、私有 RNG/key 核对、mode 字节重演，以及 CPU/CUDA 的局部梯度、旁路和状态隔离工程证据。动作问题不改变这些产物已完成其原对象验收的事实。原报告及失败产物不覆盖、不改数值或统计门。

| 原证据 | 本次发现后的准确适用范围 |
| --- | --- |
| r1 H1 reward 支持及 H1/H10 候选收益 | 来自 legacy direct-suite 轨迹、动作和 reward；不是生产映射下的真实 reward 标签 |
| H100 首动作加共同记录续行机会、cuecost | 是旧物理控制接口上的有限后果，不能称生产策略的自然状态机会或正常反馈价值 |
| mode/MC 的固定控制预测评分 | 模型输入与真实 reward 由旧 direct-suite 对象绑定；不能据此判定生产接口下缺少必要预测技能 |
| MC epsilon/区间及独立 chunk 审计 | 原数值裁决仍为 `MC-inconclusive`；一致性审计不是模型技能或方法有效性结论 |
| gate/excess KL/P 等离线指标 | 原来源条件下的真实计算与对照局限，不能外推成生产训练的压缩或性能结论 |
| off/logging/default-off/梯度工程验收 | 工程不变量继续有效；不替代正确环境对象的科研证据 |

特别地，旧 MC 的数值不稳定不能解释为稳定负结论；接口问题也不使旧 MC 变成阳性。旧 mode 的必要方向未支持只能按 legacy 对象理解，不扩大为 reward head 无技能。两个旧训练阶段仍是同一 baseline seed0，不能称多个模型复现。

本发现主要限制 DL 诊断来源和外部标签。经 `main.make_env/wrap_env` 执行的正式 M2 不因 DL 另建的 direct-suite 采集自动失效；本文未重新审计正式 runner 的全部调用路径，不改变其冻结协议或结果。

**尚无归因证据的推测**：动作接口不匹配可能影响旧 reward margin、后验增量稀少和模型误差；没有证据量化其贡献，更不能保证修复后支持某个候选。gray-body 保留姿态/轮廓的局限、actor 利用能力、未来观测补偿及共同模型误差仍各自成立。

## 3. 为什么须重建来源，而不只改未来执行

接口影响整条行为轨迹的真实状态、图像、reward 和策略 history。后续仅将 q/p 动作正确映射，仍从另一控制接口生成的状态分布与历史出发；这可以另称 legacy-state counterfactual，但不足以支持“生产 baseline 自然状态”的主张。

旧 snapshot 未保存实际 Agent carry。旧预测/MC 以 seed20261012 重新抽样恢复一个条件历史，不是采集时实际 policy latent 的恢复。新的生产来源必须明确实际 carry/RNG 来源，不能仅换 `action_applied` 或沿用该诊断 history seed 后宣称历史一致。

生产 contract 的新来源是必要工程修复。**它对接口一致性验收充分，不对可利用观测机会、辨别信号或方法性能充分。** 当前 Dt 仍是 categorical-mode 上 raw 条件 Normal 的 sym-KL/actiondim proxy；修复物理执行不会自动把它变成实际 clipped/mapped 策略边际的 KL，也不改变用户的 Dt＋另一信号目标。

## 4. 最小工程修复与验收条件

新增独立 [dl_action_contract.py](../scripts/dl_action_contract.py) 和对应 [CPU 测试](../tests/test_dl_action_contract.py)，由实现 Agent 拥有。以下是验收要求；测试及真实新来源尚未完成时不得写成已通过。

1. **生产 wrapper 是权威**。helper 返回独立 copy 的 raw/clipped/unified/applied 与 dtype/bounds；用实际 `main.wrap_env` 接收器和真实 12 维 spec 逐位核对每个阶段，包含超界、上下端点、零、非对称关节及难以代数化简的 float32 值。错误 shape/dtype、非有限值、错误 bounds 或未知 contract 必须拒绝；不能以 allclose 通过动作映射等价。
2. **模型动作和物理动作分开**。raw float32 进入 policy carry/RSSM；applied 是 suite 实际收到的控制。source、未来模型控制、真实后果及记录动作复放均保存两者；禁止将 applied 偷换成 RSSM prevact，或将 normalized 零动作称为物理零 torque。
3. **真实 CPU 环境验证**。从同完整 suite snapshot，生产 wrapper 路径与 helper-applied direct-suite 路径逐步核对 reward、integration、task/RNG、step/reset/terminal；保持原物理误差要求 1e−8。另外保留 raw-direct 与生产路径确有差异的反例，不能依内部 clamp 或 reward 偶然相等宣称接口正确。
4. **完整复原包含 wrapper 状态**。FromDM 的 `_done` 决定自动 reset，不能仅恢复 suite 状态后沿用新 wrapper 的初始 `_done=True`。若使用 wrapper 路径，保存/复原影响 step 的 wrapper 状态；若复放只使用 suite/applied，则独立验证与生产 wrapper 的 repeat1、观测及终止语义等价。覆盖非首步、近 time-limit、terminal/reset 边界，禁止跨 reset 补窗口。
5. **真实小模型契约**。使用 actual tiny Agent 的生产 `policy(mode='eval')` 验 raw 动作及 carry、Gaussian/latent 采样、生产物理映射和私有 RNG 隔离；snapshot 恢复不能改变 policy 的实际 previous raw action。fixture 只支持接口，不证明 size50m 自然任务或方法有效。
6. **来源版本拒绝静默混用**。新来源 plan/complete 必须有明确 `production_clip_unify_normalize_v1` contract/version、bounds/dtypes、helper/source Git SHA、实际配置、raw/applied hash 和依赖。新诊断入口拒绝缺失/legacy contract。旧来源与 `legacy_raw_suite` 语义保留，既有工具/产物不暗改为生产版本。

**当前 CPU 证据**：实现 Agent 在本机 dirty 基准 `dd7da2ef4b99447e7fa867bec48fe6235655c8ae` 报告 action-contract 测试3项通过、5.78秒。独立只读审查已核对应源码与 ignored 收据 `analysis/outputs/dl-action-contract/cpu-audit-r1-final.json`（SHA256 `48f74482d22fa8f4549b30b21ff4f5c02a8b2acd604c160ebfa0c9057df456ca`）。八个预定输入的实际 `main.wrap_env` applied float64 与逐 dtype 公式字节相同，套件 reward/task/integration/counter 对照相同；旧 raw-direct 存在 qpos/qvel 差异，不只是 ctrl 数组不同。例如 normalized-zero 的 qvel 最大差为 .0404340，不能用此单步值量化旧科研结果的损失。

该实测为同 reset 状态的非 terminal 单步、无 RGB render。actual tiny `model.policy` 另验 raw sample/carry 与下一 `dyn.observe` 的 raw previous action；不是通过公共 eval runner 的真实 size50m 来源。脚本 SHA256 `815f07844c4b4bbd85ff529ef5477626bca99fbc77212fea193203b176819137`，测试 SHA256 `af6c0cd4330a8b24950860cb245a55361b974a18d2dea5f471020998d373002c`。完整 wrapper reset/window、来源 provenance 和真实 CUDA/EGL 新采集仍须分别验收，不能将这3项扩写为本节全部条件已完成。

全部要求满足、独立代码审查与源检查完成，是启动正确接口来源的工程条件；有某条 fixture 通过不替代其余条件。

## 5. 根 Agent 冻结的新来源配置，待工程验收

以下是授权内 Agent 经交叉审查选定的来源配置，尚未运行。根 Agent 已冻结数值与接口；运行版本仍须绑定确定代码 SHA、plan、完整工程验收和实际资源，不称用户逐项确认。独立审查同意该设计修复接口对象；它没有冻结反馈算法或性能实验。

| 项目 | `DL-source-r1` 选定条目 |
| --- | --- |
| checkpoint | 先且仅原 baseline seed0 的 300000 训练动作阶段，不按 return 选择；不自动追加 1m |
| 生产环境 | actual DMC + `main.wrap_env`；walk、size50m、64×64 RGB、camera2、repeat1、proprio=False |
| 行为 | 实际 `Agent.policy(mode='eval')` 的 sampled policy；raw 保存到 carry/source，suite 最终 mapped 控制另存 `action_applied` |
| 新 collection seed | `20261020`；沿用 `episode_plan(32,8,seed)` 的独立 environment/policy seed 与 calibration/blind 划分算法 |
| 来源规模 | 32 个完整 episode；8 calibration、24 blind；只先设计/检查 calibration，不打开新 blind 数组用于选择方法 |
| 完整窗口 | 固定1001帧/1000动作及预期 time-limit；提前终止或末端缺失停止来源，不替换 episode |
| 预定位置 | 每 episode 原 `sample_positions(frames,16,100)`；不按 D/K/C、reward 或 q/p utility 选择，不补选有利 episode/位置 |
| 模型/训练状态 | checkpoint 参数、optimizer/norm 和训练 counters 冻结；零 optimizer update、零新 gate |
| 资源与硬预算 | sv3物理GPU4单卡，3600秒父进程硬监督，包含加载、编译、采集、全部参考复放与写盘；超时保留不完整证据并停止，不追加预算/换卡 |
| 来源目录 | 新 source version、确定 SHA、新 attempt 目录，不覆盖任何 legacy source、失败证据或正式 M2 产物 |

固定 checkpoint 文件 SHA256 `2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9`，参数 digest `14859c4e129b51aa488c10eab09f16d56fd110c76239b2c9072f415829d111c2`，原 config SHA256 `11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445`。updates74493；pickle actions18769 是 policy calls，不能代替训练动作阶段。来源证明见 [dl_checkpoint_sources.json](dl_checkpoint_sources.json)。

**已决定的来源记录**：新 [dl_production_source.py](../scripts/dl_production_source.py) 保存每个预定位置采集时实际 pre-policy 完整 carry（enc/dyn/dec/previous raw action）、逐叶数值/dtype/hash、实际 eval seed/counter 收据、suite snapshot 与 FromDM `_done`。这些是 source 采集时的实际状态，私有另采历史不能冒称恢复。原 dtype host numeric leaves 是当前设计；若未来改用 float32 数值保存 BF16，须保留原 dtype 信息并明确是无损数值转换，不能 hash 嵌套 object 指针。只读源码支持该记录设计，不等于真实采集已完成。

collection 的 eval seed/counter 依法为每 episode 设置和推进，要记录每次 reset/calls 与预期值；不能要求全部 counters unchanged。训练 updates/batches/actions、optimizer/norm 参数应与加载 checkpoint 相同。完整 snapshot、当前观测 flags、raw/applied、reward 编码及 source plan/complete hash 均绑定同一 epoch/contract；实际 source window 缺失即报告不完整，不追加 seed 补齐。

新来源验收至少包括：实际 wrapper 收到控制与 helper 精确一致；从保存 snapshot 使用 recorded-applied 序列完整复放 H100，float32 reward 编码与 collection 精确对照，float64物理完整状态保持 1e−8；必要 source policy/carry/RNG 重演；选定位置/seed/分割完整；参数不变与训练/eval counter 各自语义正确。参考 replay 本身也必须明确经过哪个 wrapper，不对 applied 再 normalize 一次。

来源生产时可以对全部32条轨迹做动作/物理接口工程核查，但不能计算 blind 的 D/K/C/utility 标签用于科研设计。最后没有执行动作的 terminal frame 若使用占位 raw/applied，必须明确 no-action mask，不能将其伪称映射等式或实际 suite 调用。根 Agent 已报告 source 本机3项 CPU fixture 通过；这包含 actual DMC/physics 与模拟 render/agent 的来源流程，不替代确定提交、真实 checkpoint、真实 renderer 或独立新来源审计。最终测试 SHA 与结果由运行日志记录。

资源仅在既有授权 sv3 GPU4–7 范围内由根 Agent 另行记录实际安排、fresh 占用、CUDA/EGL 同卡映射、确定 SHA/配置/seed/环境及预算。本文没有启动记录，不把旧空闲快照当当前资源状态。

## 6. 反馈机会诊断仍是未冻结候选

只有正确 production 来源验收后，才考虑另立 `DL-feedback-r1`：新八个 calibration episode，每条16个预定位置取索引 `[0,5,10,15]`，共32状态；每状态4个预先命名的 paired latent/actor RNG。private seed20261018、统计seed20261019、H100、episode等权 signed return/2000 bootstrap、单卡3600秒硬上限均是当前候选，**未冻结、不由本来源设计自动授权运行**。原32状态闭环建议不会复用 legacy source 继续执行。

候选模型层定义是：同真实 snapshot 与实际 t 前 carry，在同 h 上 q 吸收当前 clean RGB，p 使用真实 `_prior(h)`，初始 latent 按实际 unimix 分布同噪声采样；p 只跳过当前一次观测。此后每分支从自己的真实环境获得 RGB，使用自己的 previous raw action 正常 posterior 更新，actor 继续 raw Gaussian 采样，物理控制统一按生产映射执行。不能使用零图像假装 prior、将 q 的 current latent 交给 p carry、让 p 持续不看图、使用记录续动作，或将 eval 冒称 mean。

冻结前尚须决定实际初始 clean 图像和 source RGB 的核对、task terminal 与末端100步边界、完整 wrapper/suite 状态复原、policy 离线重演/renderer 非逐位限制、物理重复与硬预算的真实成本。真实 terminal 不能通过删除失败分支造成选择偏差；time-limit 不能跨 reset 凑 H100；明确吸收终止或窗口停止规则后再执行。

`lower95>0` 若采纳，只能是“固定自然来源样本下平均一次观测控制机会”的探索充分门，**不是选择性后验保护的必要门**。少数有价值目标状态可能被其他状态的损失抵消；平均正差也可能来自 prior/actor 表征不匹配。32状态×4 RNG 是嵌套于8 episode的组内重复，不能给精确逐状态正负标签，也不是独立训练 seed 或新方法验收。非正/不确定结果停止这个有限自然对象，不否定全部 selective protection；合法有限状态挑战仍须另立必要性、对象与预算，不能反复改 H/样本/RNG 求通过。

即使平均机会得到支持，用户的完整目标仍需：目标子集的独立外部证据、第二信号相对 D 的辨别增量、nuisance/OOD/actor 能力反例、真实活跃 rep 压缩、有效 S/P/W 等对照、真正未见数据及条件训练。真物理状态只可作诊断证据，不能泄露为 vision-only 部署 gate 输入。

## 7. 当前完成边界与下一步

**2026-10-11 13:09后实际结果**：r2执行/测试完整SHA f2d24da1831da2ffca22d313e7f534c96fc0d3eb，clean服务器source5项及GPU4新的32episode来源attempt02完成；独立artifact和八calibration CPU物理12800步审计passed，状态误差0。source440.142416秒/2585MiB、零optimizer，实际CUDA/EGL同物理4，已释放。完整结果/实际来源/失败历史/证据限制与资源见 [DL_PRODUCTION_SOURCE_RESULTS.md](DL_PRODUCTION_SOURCE_RESULTS.md)。反馈详见 [DL_FEEDBACK_DIAGNOSTIC.md](DL_FEEDBACK_DIAGNOSTIC.md)，仍未冻结/执行；不因来源通过宣布方法有效。下段是12:56时的失败/修复历史，source pending已由本结果解除。

**12:56失败/修复历史**：bb78e8c确定提交的sv3 clean action3/source4项及pip check通过；GPU4 source attempt01在首episode之后的actual policy重演处失败，原host BF16 carry进入 `_stack` 引发implicit transfer guard，exit1且没有complete。root新增engineering-r2显式按local policy_mirrored放置原dtype carry及实际公共tiny/BF16/strictguard重演测试，本机source5项通过；其后f2的新clean测试/attempt/独立审计见上段。没有放宽transfer guard或覆盖旧失败；源协议数字/对象不变，反馈仍未冻结。

本次裁决是动作契约不匹配及其证据范围，未改任何旧数值、科学门或算法。旧 MC 停止保持；新来源工程验收完成，反馈和新方法没有科研结果。不能保证修复接口会找到有效方法，也不能以接口错误抹掉其他已知局限。

首选下一步是在本聊天，以已验收production source和实际carry为前置，论证并冻结一个有限反馈对象，先完成真实policy/key/F64 reward复放语义验收，不自动重跑全部旧诊断或开启训练。平均机会不作为selective保护的必要门；来源通过不等于第二信号或方法有效。保留旧/新scope、真实失败与未决问题，继续推进形成可正式实验方法的原目标。
