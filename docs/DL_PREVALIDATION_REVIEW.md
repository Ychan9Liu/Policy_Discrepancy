# DL 正式实验前的科学审查与判据

记录版本：`DL-review-r1`，2026-10-10。审查基准 HEAD：`3979d67813a65f665bf3ee1cb176f11929b8acfb`。本文是本次授权下 Agent 内部提出的审查意见与可执行判据；根 Agent 整合后的实际冻结协议、代码 SHA 和结果优先见本任务最终运行记录。本文不把自主设计写成用户逐项认可，不把预注册门槛写成科学定理，也不把尚未完成的检查写成通过。

当前整合协议为 `docs/DL_PREVALIDATION.md`（DL-protocol-r1）；其中已冻结的8 calibration episode、D tertile/活跃K median、C门槛等优先于本文早期建议。本文 DL-R10 补记独立代码审查，不构成运行验收。

用户本次已授权必要文档、代码、正式实验前算力验证与证据驱动修订，并指定 **sv3 物理 GPU4–7**。此前 RESEARCH 的 r0 归档“不授权实现”的历史边界不再阻止此次验证，但原 M2 v1 的四任务、四组、参数、正式指标和既有运行不由本扩展改写。资源现场由执行 Agent 核查；本文未核查占用、未启动 GPU、未修改算法代码。

## DL-R1 研究主张应缩到证据真正覆盖的范围

待检验的可实施主张是：在已有 actor 对当前观测有响应、一步奖励模型有留出预测能力、rep KL 活跃的样本上，真实下一步标签支持能否使保护更集中于有外部短时任务价值依据的更新，并使 gate 在活跃压缩梯度上按预期工作；其区别是否超出总体保护更弱或 Dt/K 粗分配。

这仍是工作假设。即使检验通过，也只能得到“值得正式比较的候选方法”，不能得到长期性能增益、因果信息分离、最优策略充分性、跨任务或跨训练 seed 泛化的结论。

必须分别裁决以下问题：

| 检验 | 需要的直接证据 | 不能代替该证据的材料 |
| --- | --- | --- |
| 工程正确性 | 对齐、编码、mask、detach、初始化/随机流和实际模型梯度检查 | 数学公式正确、fixture 为绿 |
| 信号辨别 | 留出真实任务数据、独立外部标签和合法 nuisance 配对 | e 的正均值、posterior 更拟合自己的标签 |
| 压缩作用 | 活跃 raw KL、实际权重、超额 KL 释放、局部梯度作用 | free-nats 底座造成的 loss 数值下降 |
| 样本对应价值 | 与强度对照及有效条件置换的比较 | 只超过原 Dt |
| Dt 的额外价值 | 在相近释放强度下与 reward-only 的比较 | 保留 Dt 的设计理由、五组比较 |
| 性能可行性 | 预先冻结范围内的短训练实际指标及失败记录 | 冻结模型诊断、局部梯度、代理有效 |

## DL-R2 主要不可辨识性和反例

1. **行为动作泄漏**：有用 cue 先促成好动作，给定该动作后 prior 已能预测奖励，可能出现真实 cue 有价值而 e=0。因而“同动作消除动作混杂”不等于消除动作中的 cue 信息。
2. **即时代理与动作价值不同**：奖励预测更准确不保证动作选择更好。相同即时 reward、不同未来转移的状态可需要不同动作；H=1 会系统漏掉此类信息。
3. **训练支持不同**：q/p mode 经同一 prior 前向仍可能处于不同支持区域。q 路径的预测更好可以来自 head/RSSM 支持偏差，无须有新增决策信息。
4. **mode 与尺度**：q/p mode 跳变或 actor 小方差可同时放大 D 和奖励 head 响应。两信号不能自动互相校正；低 D 也不证明观测无用。
5. **共享 h 与混合因素**：历史已经包含 cue，posterior 的当前增量可能很小；标量 gate 还会一起保护该时刻无关因子。本文不声称逐因子分离。
6. **真实标签仍可有捷径**：背景伪相关、奖励中的不影响控制的加性部分，都可能获得预测支持。detach 只阻断直接 gate 梯度，不阻断以后学习产生的反馈。
7. **正部偏差**：无真优势的有符号噪声也可产生正 e；必须报告有符号 `u_q-u_p`，并与独立标签置换的完整分数比较，不能只看截断后的 e。

只用 D/K 不可能识别任务价值：固定 q/p/actor，可改真实奖励或转移使相同 D/K 对应有益、无益或有害更新。额外 raw KL 可提供状态更新尺度和梯度活跃性诊断，不能作为第三方任务标签。

## DL-R3 工程与梯度最低验收

检查源：`dreamerv3/agent.py` 的 replay-context 后 loss/reduction，`dreamerv3/rssm.py` 的 observe/core/prior/free-nats，`dreamerv3/rep_probe.py` 的既有 Dt，`embodied/jax/outs.py` 的实际 TwoHot 编码。

- `prevact[:,t+1]` 与 `reward[:,t+1]` 必须对应从状态 t 执行的同一步动作和结果；reset、context 裁剪、terminal 端点和 chunk 尾部各有独立例子。下一步 terminal reward 有效，跨 reset 无效。m 只控制许可，不新增 rep loss mask。
- 两路径从**同一个 h_t**出发，只改当前 q/p mode；不能复用已吸收下一帧的 h 或 posterior。`_core` 前将起点展平为 batch；直接送 `[B,T,S,C]` 会与当前 flatten 约定冲突。
- TwoHot target 必须与实际 reward loss 一致，包括 squash、bins、clip/equal 和熵；用 `CE-H(y)` 得到兼容度。非均匀 bin 不得替换成先 symlog 后等距标签。
- 所有最终 gate 与 probe 输出显式 sg；共享 actor/reward/core 参数经 gate 的直接梯度为零。正常 reward/head/actor/dynamics 的既有训练梯度保留；rep 经 posterior 与 shared h 仍可能更新 dynamics core，不能断言所有 dynamics 梯度为零。
- free-nats 后乘 v、完整原 BT mean 前作用，scale 只乘一次；无效 m=0 返回 v=1，不回退原 Dt。严格 `K>tau` 的局部 posterior-logit rep 梯度按 v 缩放，严格 `K<tau` 梯度为零，阈值相等沿用 baseline/JAX 约定。
- rho=.5 给出每位置局部梯度的缩放下界；不同位置梯度抵消后，总参数梯度范数甚至可能增加，因此不保证 optimizer 更新、长期信息保留或性能损失上限。
- uniform 初始化 e=0；关闭、alpha=0、rho=0 静态旁路与 baseline 对齐参数键、初始化、模型随机流和状态。独立 probe/置换 RNG 不消耗 baseline RNG。
- 非有限有效值和超出声明容差的负 KL 必须失败；不得静默丢点、替换 gate 或改容差求通过。下溢到 0 与没有有效样本分开报告。

小型 fixture 验证上述数学/接口边界；至少一次真实 size50m、真实视觉 batch 的梯度和状态检查才能补足工程范围。二者都不证明信号有任务价值。

## DL-R4 独立诊断：标签、干预和动作支持

首选单任务 `dmc_quadruped_walk`、vision-only/64×64、repeat=1、size50m，与既有原任务保持可对照。该选择属于 DL 自主设计，不是抗干扰训练任务已确认；若实际资源与可用 checkpoint 不支持，应先记录事实再冻结替代任务，不能看到信号结果后追选任务。

### DL-R4.1 真实配对与合法 nuisance

对一条真实轨迹中的预定位置，保存完整环境/物理状态及 clean 观察，固定历史 carry 和当前之前的动作。只对当前图像作干预，重新编码当前 posterior；共享 clean h。

背景变换只允许修改可证明属于 all-geom/body 外的像素，保留 agent、地面接触、目标和任务几何；状态、动作、真实 reward 均不变。mask 必须有可核查来源和覆盖图。覆盖 ground/target/contact 或改变动力学的变换不当合法 nuisance。新背景在留出集与校准集分开，以减小样式特定捷径。

当前 body/cue 遮挡称为**信息剥夺**，不称合法 nuisance，也不直接贴“重要”标签。当前 cue 是否值得保护，需要独立真实动作后果证据；若存在 off-support，结果仍只能解释该剥夺试验。

合法 nuisance 中原始关键线索仍存在，原 C 非零可能合理。因此误保护统计使用 `ΔC=C_nuisance-C_clean`，不能把 nuisance 图像上的全部 C 当假阳性。

### DL-R4.2 第三方任务依据

在保存的同一个真实物理状态，比较 q/p mode actor 的**裁剪后均值动作**实际模拟后果；同时使用预先固定、与 D/e 无关的候选动作集合得到真实 H1 reward 与 H10 短时 return。标签取自环境真实反馈，不取自 reward head、critic、D/K 或 e。

H10 必须明确控制序列。首选“只替换第一步动作，后 9 步共用原记录动作序列”，更接近当前信息影响；如果工具选择将初始均值动作恒定执行 10 步，必须明确这是**常值动作 10 步的有限收益**，不得称完整策略收益或全局最优依据。两种定义不可混用或按结果选择。

初始候选动作与后续模拟不吸收另一路未来观测来修正信号；诊断真实模拟可看到环境结果，但训练 gate 仍只使用原定义的记录标签。

恢复须覆盖 physics、任务可变状态、环境步计数、必要 RNG 和隐式积分状态，或用重复同动作得到的实际结果证明完整恢复。仅复制 qpos/qvel 无法默认覆盖全部环境状态。每类样本先重复同状态/动作分支；复现失败时外部标签解释停止，不通过放宽容差冒充确定性。

“q 动作优于 p 动作”的标签只覆盖这两个候选动作与该 horizon；若二者近似一样、都差、或固定动作集合无区分性，结论是外部诊断不足，不能推出 cue 无用。

### DL-R4.3 动作泄漏与代理是否足够

对每个预定固定动作 a，从同一物理状态取其真实下一步 `r(s,a)`，分别计算 q/p 两路径的同动作分数。该 e 与 behavior-action e 比较：

- 有外部动作价值证据、behavior e 小，而固定动作 e 有支持：符合行为动作遮蔽 cue 的反例，记录覆盖缺口。
- 两者都无支持而 H10 后果有区分：H1 代理不足或 head/latent 支持不足；进一步用真实 H1 与预测误差区分。
- 对合法背景变化大、对外部有效 cue 无区分：代理辨别主张不支持。

不得给固定零动作输入，却仍使用原 behavior action 的 reward 标签。该错误会人为制造“动作泄漏被解决”。固定动作的 support 距离、真实后果分布和 head 误差必须联报；离线动作若明显不受模型支持，结果是支持范围不足。

## DL-R5 样本、预先门槛与停止规则

以下数值是为了决定是否值得进入短训练的 **DL 自主筛选门槛**，不是统计功效保证或理论常数。根 Agent 应在观察盲测输出前冻结；若已有输出，不得追调门槛求通过。科学协议的最终版本和运行记录须注明实际采用值。

| 项目 | 首轮建议 |
| --- | --- |
| checkpoint | 至少两个训练阶段；若只有一个，明确限定单阶段。未训练 checkpoint 仅作零信号工程对照 |
| episode | 每 checkpoint 32 个独立环境 episode：8 校准、24 盲测；诊断环境 seed 固定列表，与训练 seed 分开 |
| 位置 | 每 episode 最多 32 个预定间隔位置，不按 D/e 挑点；靠近尾部不足 H10 的位置按事前规则排除真实 H10，但 H1/m 统计继续保留 |
| 不确定性 | episode 作为 bootstrap 单位，2000 次；不能把关联 frame 或评价 episode 当训练 seed 重复 |
| 外部正类 | `(R10(q)-R10(p))/10 >= .01`；负类 `<=0`；(0,.01) 为灰区，保留连续结果但不硬贴二分类 |
| 最低覆盖 | 正负类各至少 8 个盲测 episode、128 个有效位置；否则 AUROC 检验不充分 |
| 辨别 | C 对外部标签 AUROC 点估计至少 .60，95% 下界>.50；连续外部收益关系与分层结果必须同步报告 |
| cue 差值 | 有独立价值依据的 cue/信息剥夺配对 ΔC 应方向正确，95% 下界>0；.01 为预先设定有意义均差参照，低于此值须报告效应小，不能仅靠大量样本宣称实用 |
| nuisance | 合法 nuisance 新增 ΔC 的 95% 上界<=.01；超过时不支持预设背景误保护控制目标 |
| 活跃性 | m 且 raw KL>tau 的盲测集合至少 8 episode/128 位置；超额 KL 释放占基准超额 KL 至少 1% 才值得检验训练机制 |

门槛未通过的解释须分类：

**新增 e 的证据不能由总体 C 的关联替代**：外部 q/p 动作收益标签也可能随动作差/D 变化，C 中又含 fD。即使总体 AUROC 通过，也可能完全来自原 Dt。须同步报告 D/fD、signed compatibility、e、C 对同一外部标签的表现，在冻结 D/K 层内检查 e 的增量关系，并给 C 相对 S（仅 D 单调分数）的配对 ΔAUROC/区间。没有条件增量证据时，只支持总体保护与局部收益关联，不支持新来源提高辨别；分层样本不足仍记检验不足。D/K 不应按测试结果重新分箱。

- **工程失败**：对齐、复制、数值、梯度错误。修复后使用新尝试标识，重做受影响证据；不作科学否定。
- **检验不足**：外部动作收益无区分、正负类不足、活跃覆盖低、head 预测能力差或置换近似 identity。允许在未看盲测结果前确定增样计划；已看结果后的增样属于新版本验证，保留旧结果。
- **候选不支持**：足够覆盖、合法外部标签下方向反向/消失，背景增加保护过大，或短期 proxy 系统遗漏外部有效 cue。应否定对应主张，不能加模块把 r0 失败抹掉。
- **有限支持**：仅某阶段、某动作集合或某 checkpoint 支持。保留范围，不扩大到长期/训练 seed 泛化。

32 episode 提供初次诊断额度，不能承诺任何效应的统计功效；episode 内帧多不自动增加独立重复数。训练 seed=0 的多个 checkpoint 和几千帧仍是一条训练路径。

## DL-R6 gate、强度和替代解释

令 `E=(K-tau)+`、`A=1[K>tau]`、`C=1-v`，释放统计取 `sum(C*E)/sum(E)`，分母与有效 m 集合的定义分别记录。需记录 signed compatibility、两侧 CE/KL、e/D/K/m/C 的分布、活跃覆盖、每 episode 和 checkpoint 阶段；不只展示平均 loss。

至少在冻结真实 batch 上比较：

| 组 | 权重 | 排查问题 |
| --- | --- | --- |
| B | 1 | 原模型 |
| D | `1/(1+20D)` | 原保护 |
| N | `1-m*rho*e*fD`，rho=.5 | DtLatch r0 |
| S | `1-m*kappa*fD` | 仅调整保护强度/原 Dt 形状 |
| P | 在预定层内置换本组完整 v | 超出粗 D/K 层的样本对应 |
| W | `1-m*lambda*e` | reward-only 是否已足够 |

冻结校准集上：

```text
kappa = rho * sum(m*A*E*e*fD) / sum(m*A*E*fD)
lambda = rho * sum(m*A*E*e*fD) / sum(m*A*E*e)
```

lambda 给 reward-only 与 N 同样的校准超额 KL 释放，不用未匹配强度的 W 解释 Dt 必要性。分母为零或非有限则相应比较未成立，禁止默认值。分子为零可以为 0，但说明候选未释放任何校准压缩。

这只匹配校准期的标量释放，不匹配以后每组的梯度向量、真实数据分布或更新轨迹；各组自己的当前 batch 都计算同样 detached 前向以控制新增调用。

P 建议使用校准 D 三分位、活跃 K 中位数形成最多 6 个活跃层，加 m/活跃状态，重复分位边界合并；所有 cut points 从校准集冻结。单卡仍须明确 batch-time 置换范围。每更新独立、可重放的 RNG；只做一次预定置换，不挑一个强度更合适的 permutation。

报告真实移动率、层大小、singleton、释放量变化及其区间。建议活跃权重位置换位>=50%、相对 weighted-excess 变化<=10% 才给“条件置换有效且近似匹配强度”的解释；不满足时只能称对照无力/残余强度混杂，不能将其与 N 接近当精细选择无价值。不得靠在线改变 bins 或重复置换追通过。若校准数据不能形成有效 P，先否定该检验设计，保留候选未定。

离线层析能排查信号/释放覆盖，但不能证明长期性能价值；后者须实际训练。五组没有 W 仍可检验较弱主张，但不能完成“双信号均有必要”的替代解释排除。新增 W 是必要科学比较，先离线计算，不要求为每种诊断机械新增训练组。

## DL-R7 证据驱动修订树与短训练

1. 工程通过但 H1 无独立辨别：停止 r0 效果主张。先区分 head 未成熟、模式误差、行为动作泄漏、长时后果遗漏；不能把信号不显著统一归为模型不够大。
2. 外部有效 cue 在低 D 上被系统遗漏，而 e 有支持：reward-only W 是首选候选修订；保留 r0 失败证据，新版本命名 `DL-r1`，不通过调 alpha 隐藏。
3. H1 真实 reward 本身无法区分，而 H10 外部后果能区分：承认 H1 适用范围不足。仅有这种证据时讨论多步真实奖励标签，不直接加 critic/continuation；新 horizon/mask/成本和对照均重新冻结。
4. gate 只在不活跃区有释放：方法训练机制尚未被检验。保持 free-nats 和原 loss scale，不降低阈值求活跃；选择有自然活跃 KL 的阶段/任务需新协议，不能按效果追选。
5. N 仅超过 D、与 S 接近：最多说明旧保护过强；若 P 有效且同样保留效果，不支持更细对应价值；若 W 与 N 接近且有相似覆盖，不主张 Dt 必要性。
6. 不同阶段支持方向冲突：把 head/actor 成熟阶段写为适用边界；任何 warm-up 或可靠性条件都是新候选，必须独立再验证，不能在本轮盲测中试到通过。

短训练是条件阶段：在工程、独立信号、活跃机制满足后才值得开展。首轮可用同一任务、六组、训练 seed0/1/2、每组 100000 动作（含 prefill、排除 reset/eval）、repeat1，评价 0/25k/50k/75k/100k 每点 5 完整 episode，原始 return AUC/100k 及100k末点评价并保留各 seed/episode。以上是 **DL 自主资源和协议建议**，最终整合须先冻结；不得称 v1 正式指标或改写其 1m/21×10 设置。

S/W 来源应是同任务 DL baseline 的预定校准窗口，例如 [20000,40000) 的成功更新；kappa/lambda 后冻结再启动 S/W。窗口本身属于自主待冻结设计，不因现有 v1 [100k,300k) 而自动适用。若早期无足够信号，则这段短训练不能检验成熟代理，需记录不足并重新设计完整验证，不能填系数。

100k 的三个训练 seed 只能作短预算探索；短训练无增益不能否定长预算收益，也不能无条件用于证明候选值得正式试验。达到正式实验候选门槛至少需要：定义与代码可复现；独立辨别与活跃机制有支持；主要已知替代解释经过有效对照或明确限定主张；短训练没有可重复的严重退化且已执行预算有真实机制覆盖；未决长期范围列为正式实验假设。是否具有最终长期性能增益留给正式实验。

## DL-R8 资源与结果交付

已有工程 size50m/A100 数据是资源先验，尚不是 DtLatch 测量。历史单卡 warmed 约24.2–24.6动作/s；绑定 CUDA/EGL 的独立短验计算卡采样峰值8088MiB，见 `docs/SERVER_RETURN_CUDA_AUDIT.md`。新增 probe 的吞吐/显存必须实测，不从旧表推新 gate 的百分比开销。

sv3 GPU4–7 足以**计划**四条独立单卡任务并行，无需模型多卡。不可依据低显存共享卡；既有任务或未知进程占用时不干扰、不改卡。可将工程真实模型验收、两个 checkpoint 独立诊断、数据/状态恢复核查分别排程；依赖校准的训练后启动。六组×三 seed×100k 共1.8m训练动作，按旧吞吐仅训练部分约20.4 GPU小时，四卡理想排程约5.1小时；编译、数据收集、评价、校准等待及新 probe 开销另计，这不是完成 ETA。

产物应包括：协议冻结文件、Git SHA/配置 hash、设备 UUID/CUDA/EGL映射、checkpoint来源/hash、episode/干预manifest、恢复与mask证据、按 episode 的盲测统计、校准系数来源、梯度/随机流证据、实际训练组和失败尝试、方法版本裁决。大数据/检查点/replay留服务器独立 runs，仓库只提交小证据索引与工具。

正式候选审计不能只查文件存在：逐条核对 DL-R1 表中证据是否覆盖真实范围。允许结论为 r0 否定、检验不足或限定场景支持；若形成修订候选，必须给出新定义、独立再验证、对照和剩余假设。只有上述证据齐全才能声称完成“形成可正式实验的思路方法”。

## DL-R9 本审查交付和交接

DL-R1至DL-R9对应最初理论/设计审查阶段，不报告真实诊断或训练通过；该阶段未修改代码。之后新增的独立runner及fixture见DL-R11；本审查Agent没有commit/push/服务器同步或启动GPU。最终代码与测试 SHA 以根 Agent 集成验收记录为准，本文审查基准与运行版本不同必须列差异。

继续本次已授权任务：执行 Agent 在指定资源完成工程与独立诊断，根 Agent 结合本审查和实际证据冻结后续阶段。完成后按 AGENTS 报告实际组/资源、已完成与未完成、完整最终/测试 SHA、文档与结果位置、当前 GPU 状态及下一阶段 prompt。不得用跨 chat 交接替代当前已授权的完整验证。

```text
项目根目录 D:\program\PD\dmr3；当前工作树
C:\Users\97370\.codex\worktrees\1f9a\dmr3。
审查基准3979d67813a65f665bf3ee1cb176f11929b8acfb；先核对集成后的实际SHA。
必读AGENTS.md、docs/RESEARCH.md的DL章、docs/DL_PREVALIDATION_REVIEW.md、
本任务最终冻结协议/运行索引与docs/STATUS.md。
用户授权本次正式实验前完整验证及必要证据驱动修订，资源仅sv3物理GPU4–7。
执行工程正确性、留出独立辨别、活跃压缩、S/P/W替代解释和条件短训练。
不要把自主配置写成用户逐项确认；不要更改原M2 v1或既有作业。
外部标签使用真实模拟反馈，恢复/观测mask必须验收；e/D/K不得自己当标签。
按冻结协议执行，失败证据保留，新修订/增样另记版本，不追阈值求通过。
结果和大产物存/data/Policy_Discrepancy/runs的独立目录，仓库提交小索引。
结论分别裁决工程、信号、机制、替代解释和性能，未验证项明确保留。
完成后给完整SHA、实际测试SHA/差异/同步范围、验收证据、资源状态和正式实验交接。
```

## DL-R10 独立代码审查补记（实施期间）

只读审查 `dreamerv3/dt_latch.py`、`agent.py`、`embodied/jax/outs.py`、`scripts/dl_diagnostic.py`；代码 owner 正在修改，运行版须以最终完整SHA和文件hash冻结，不能以本审查的临时稿替代。此审查没有启动 GPU 或把 fixture 的结果当作真实方法验收。

静态检查：TwoHot target 抽取保留原编码运算；DL 前向从同h/q-p mode出发、不用未来观测，最终输出sg；Agent在原free-nats后、完整mean前接入。W的匹配系数可以通过该组rho传入，但配置必须明确其值为lambda_W；默认rho=.5不自动匹配强度。alpha=0将W也关闭是当前接口的统一退化约定，当前alpha20协议不受影响。

已向根和owner报告并在后续稿静态核对到修正的工程问题：

- float64恢复reward与采集float32标签直接比1e-8会误报正常编码舍入；现将重复float64物理误差与float32编码一致性分开。
- selected帧batch4返回的clean latent随机抽样可能受batchshape影响，改变采样位置即可改变历史；现每帧统一batch1 advance，batch4诊断carry丢弃。
- collection的policy RNG原继承checkpoint连续eval counter；现逐episode显式set_eval_seed并记录独立policy_seed。
- 初稿主门槛使用e而非冻结C、cue缺.01点差、校准分箱不一致、缺W lambda和条件P有效性；后续稿已加入C/D配对增量、S/W/P离线证据，并固定不由分析自动启动短训练。

仍须在最终代码/真实产物复核的科学范围：

1. q/p均值真实收益只覆盖均值动作。D含std差，不能将std变化误解为无价值；至少报告std-only高D比例，若该比例主导则需共同噪声采样动作的辅助诊断或明确限定主张。
2. qclean优于prior并不证明body剥夺真的损害控制；已有q-deprived真实后果须形成独立cuecost标签，否则只能称剥夺响应。
3. 参数shape不能证明RSSM配置语义一致，例如stoch64×classes16与32×32都flatten为1024；必须核对完整size50m categorical分组/unimix/视觉camera及源配置。
4. collection与scoring checkpoint默认应一致；有意跨模型采集数据须明确行为模型/评分模型及hash，不能隐式混用。
5. 冻结参数hash与common-h是工程证据；真正的simulator恢复、mask语义、模型成熟度、盲测覆盖和条件增量仍须真实验收。正部e的标签置换只作负控制，不能替代这些证据。

## DL-R11 独立短跑入口与运行前证据契约

新增 `scripts/dl_train.py` 和 `tests/test_dl_train.py`，不修改 M2-v1 runner/checker。默认只prepare：

```text
python scripts/dl_train.py prepare --output <全新计划目录> --run-root /data/Policy_Discrepancy/runs/<独立DL尝试> --stage-c <实际analysis报告.json>
```

不传stage-c时生成18条等待索引和配置，不构造agent、不训练。S/P/W此时尚未绑定有效校准，不能运行；实际运行前须在最终clean Git SHA重建带实际报告的计划。每条run输出目录必须全新，无恢复入口。

```text
python scripts/dl_train.py run --index <计划>/RUN_INDEX.json --run-id N-seed0 --stage-c <实际analysis报告.json> --acceptance <DL前置验收.json> --resource-record <本次启动现场资源.json>
```

`DL前置验收.json`由Agent根据实际证据写出，必须绑定：`protocol=DL-protocol-r1`、`git_sha`为本次实际clean SHA、`stage_c_report_sha256`、`evidence_paths`，以及`real_size50m_engineering_accepted`、`segmentation_visual_accepted`、`stage_c_joint_accepted`、`training_control_design_accepted`四项真实验收布尔值。该记录是已授权任务内的证据审计，不是再次请求用户授权；不得用fixture或没有审阅的文件存在填true。

`本次启动现场资源.json`由执行Agent/外部supervisor核查生成，须有：`logical_server=sv3`、`physical_gpu`属于4–7、`occupancy=idle`、`exclusive=true`、`checked_at_utc`（启动前120秒内且含时区）、`gpu_uuid`、已验证相同的`compute_uuid/egl_uuid`、`cuda_visible_devices/mujoco_egl_device_id`与子进程实际环境相同。stdout/PID/pmon完整资源监测由外部supervisor保存，runner同时将输入记录/hash、实际host/JAX设备、依赖、SHA和配置hash写入每条DL_MANIFEST。过期核查需重新现场读取，不是资源再授权。

训练流程直接复用原ProtocolState、Driver.step_selected、make_replay/make_stream、prepare_batch/train/take_train_result/replay.update和评价指纹helper，未调用v1.run也未假冒v1配置。原模型参数、优化器、replay更新模式不更改；DL独立namespace生成环境/replay/eval种子。评价每点固定参数快照、5完整episode、独立eval RNG，比较训练参数/训练counter/replay状态前后fingerprint，写实际rawreturn。

实际run还会读取Stage C的score_complete/score_plan及32条实际score文件并核对hash，不仅接受报告中的布尔值；证据迁移可传`--score-artifacts <实际目录>`，保留原plan。入口拒绝fork JAX的环境worker，要求spawn，并保存实际config.yaml。

2026-10-10实际本机CPU环境运行9项契约fixture通过，覆盖18条prepare/禁止覆盖、真实阶段门槛契约、S/P/W冻结参数、SHA/report绑定、实际score文件hash/禁止伪用布尔结果、资源范围与新鲜度、16worker reset边界100k计步、完整grid原始AUC；首次发现Config list→tuple导致校验误报，按同一列表语义修复后重跑通过，并补测Windows计划保持Linux POSIX run路径。没有构造真实训练agent、启动GPU或执行DL短训练。精确计步fixture不能替代真实runner验收，目录状态始终待运行/审计。

独立审查Agent新增`tests/test_dl_train_flow.py`，直接调用实际`dl_train.run`，使用真实serial Driver、ProtocolState、EpisodeReturn和Checkpoint、agent/replay doubles，缩小为80动作/3env/每点评价2episode。5项流程fixture验证reset计步、5点完整grid、10个完整评价episode、11条实际更新receipt、参数快照，以及评价污染训练counter、非有限优化器结果、错误资源、改动关键科学配置的停止路径。Windows fixture将Checkpoint保留数设为None以绕开LocalPath的POSIX清理假设；实际Linux并行环境/模型/图形/显存不在此证据范围。2026-10-10合跑`python -m unittest tests.test_dl_train tests.test_dl_train_flow -v`共14项通过，exit0。模拟流程的Stage C/验收记录明确是fixture，不能用于实际启动。

后续接入共享stdlib入口`scripts.dl_resources.require_resource`：实际run在JAX/agent初始化前重新读取现场NVIDIA UUID/显存/XML进程/pmon，核对sv3实际hostname、物理4–7及零占用；要求CUDA绑定单张GPU UUID、EGL绑定对应物理index、EGL环境和device_probe文件hash一致，将返回的现场证据写入DL_MANIFEST。pure契约仍保留；流程fixture显式mock这一外部现场helper，所以14项通过不构成GPU资源检查证据。外部supervisor负责生成真实映射probe/记录及监测进程。真实size50m工程验收、图形mask验收、真实Stage C及完整短训练都尚待执行。
