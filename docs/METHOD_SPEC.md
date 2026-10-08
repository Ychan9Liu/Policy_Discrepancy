# Dt 首版方法与验收规范

更新日期：2026-10-08。版本：v0.1（跨文档一致性核对）。

状态：研究主问题、首版范围和基础对照已由用户确认。本文件将这些决定与回退、梯度、free-nats 和随机流要求整理为供 03 · 代码与测试使用的实现规范。任务与实验参数尚未齐全，不妨碍先实现可配置 overlay 与验收测试；正式实验须待第 6 节字段定稿。当前阶段只写文档、只读核对代码，不在本任务中实现算法或启动实验。

**03 的当前执行依据**：第 2–5、7 节是现行首版方法与验收规范，已可用于实现。第 6 节区分待用户确定的实验参数与由 03 落实的工程方案；“待定”不表示 M2、参考定义、基础组别或验收原则需要重新批准。`docs/RESEARCH.md` 中带日期的历史草案保留追溯用途，不能覆盖本规范。

来源：用户 2026-10-08 的明确选择；《DT方法迁移研究说明.docx》的参考定义；`docs/RESEARCH.md` 中对提交 `f0236c7699a87895a8cfde8e13a0d1c198024f75` 的源码核对。来源文档中的其他建议不因本规范而自动成为决定。实现前重新核对实际 HEAD 和工作区差异。

## 1 研究问题与证据层级

**已确认主问题 M2**：Dt 是否提供优于统一减弱 representation KL 的逐时刻选择信号？

**核心待检验假设**：在选定任务、观测、预算和校准协议下，Dt 的逐时刻权重相比预先校准的固定常数权重及全局打乱权重，能带来可重复的主要任务指标增益，并且 gate 确实作用于活跃 rep KL 梯度。用户确认的是检验该问题，假设是否成立仍无本项目实验证据。

baseline 比较提供原始性能参照；固定常数比较检验超出指定统一减弱方案的收益；全局打乱比较检验权重与当前样本/时刻整体对应关系的价值。常数及全局打乱不足以自动排除活跃 KL 比例、KL 大小、时间结构或其他优化动态解释。要提出更强的 M2 主张，再增加预先规定的条件打乱；条件打乱分层字段尚未决定。

M3 的任务信息因果保护、抗干扰或控制充分性不作为首版已确认结论。四类局限的诊断可以说明作用边界，不能替代主要对照。

## 2 首版支持范围

**已确认范围**：本仓库当前标准 RSSM，factorized categorical stochastic latent；所有任务 action head 都是连续动作，actor 分布使用本地 `bounded_normal`。同一个 actor 的条件输出是普通对角 Gaussian，其均值经过 tanh、标准差受限；Dt 按裁剪前分布计算。

首版不扩展连续 latent、替代世界模型、离散或混合 action、TanhNormal、截断分布及其他策略参数化。overlay 启用时对不支持组合给出明确配置错误；关闭时保留原 baseline 的动作与任务支持。技术配置字段命名由 03 按仓库风格选择并记录，不构成新增科研因素。

gate 是逐时刻标量，同一时刻所有 stochastic 因子的 rep KL 总和使用相同权重。它不提供逐因子选择。保留原模型规模、观测处理、采样 latent、world-model/actor/value 目标、优化器、replay 与数据增强；跨组只改变预先声明的 rep 权重策略。

## 3 数学定义与接入位置

### 3.1 共享 history 与代表状态

在 observe 路径中取得当前 `h_bt=feat['deter']`、posterior logits `qlogit_bt=feat['logit']`、prior logits `plogit_bt=_prior(h_bt)`。`b,t` 是 replay context 处理后的训练位置。prior 从同一个 h 计算，不额外 rollout；h 可以包含过去 posterior 的信息。

按 baseline 的 unimix 构造两侧 categorical 分布，并对每个 stochastic 因子取 mode one-hot。使用 baseline 的 actor 特征拼接与 dtype：

```text
zq_bt = onehot(argmax(q_bt, category_axis))
zp_bt = onehot(argmax(p_bt, category_axis))
xq_bt = sg(concat(h_bt, flatten(zq_bt)))
xp_bt = sg(concat(h_bt, flatten(zp_bt)))
piq_bt = same_actor(xq_bt)
pip_bt = same_actor(xp_bt)
```

actor 取分布参数，不采样动作。`OneHot.pred()` 的 straight-through 行为不能替代显式 stop-gradient。原训练中 posterior/prior 的采样与随机流保持原路径，不因 probe 改为 mode。

### 3.2 Dt 与权重

对于全部连续 action head，使用独立 Gaussian 的解析 KL：

```text
d_total = sum_heads(prod(action_event_shape))  # scalar event counts as 1
D_bt = sum_heads(KL(piq_head || pip_head) + KL(pip_head || piq_head)) / (2*d_total)
w_bt = sg(1 / (1 + alpha*sg(D_bt)))
```

每个 head 的所有 event 维只求和一次，最终 D、w 均为 `[B,T]`；已经由 Agg 聚合的 KL 不再重复聚合。均值和方差差异都纳入，不以动作均值距离替代。alpha 是有限非负标量，不可训练；非负有限 D 时 `0<w<=1`。

解析 KL 及权重运算使用 float32，保持 actor 前向的原 dtype。对数值误差导致的微小负 KL、NaN/Inf 和 alpha=0 的处理由 03 提交明确技术方案与测试，不得静默将任意非法值变成有效 gate；阈值和容差在验收前记录。它们是数值实现细节，不是新增方法变体。

### 3.3 baseline KL 与 reduction

```text
Rdyn_bt = KL(sg(q_bt) || p_bt)
Rrep_bt = KL(q_bt || sg(p_bt))
Fdyn_bt = baseline_free_nats(Rdyn_bt)
Frep_bt = baseline_free_nats(Rrep_bt)
L_KL = beta_dyn * mean_BT(Fdyn_bt) + beta_rep * mean_BT(v_bt * Frep_bt)
```

`baseline_free_nats` 完全沿用 RSSM 的行为：当前阈值启用时，在类别 KL 和 stochastic 因子求和后做 `maximum(R,tau)`；free_nats 关闭时保留 raw KL，不能新增截断。beta_dyn、beta_rep 和 tau 从实际配置读取，各 scale 只乘一次。当前 defaults 为 1、0.1、1，正式实验值尚待固定。

gate 在 free-nats 后、reduction 前相乘。当前 rep loss 无 padding/terminal mask，使用完整 `[B,T]` 的原 mean；is_first 是递归重置，is_terminal 等不新增为 rep loss mask。replay context 的已裁剪前缀不进入 gate 的统计与 reduction；gate 不使用 imagination 的 K/H 作为长度。

probe 放在正常 actor/value 参数已创建的前向之后、最终 loss 聚合之前，避免改变惰性初始化顺序。若需要暴露 prior logits 或 raw KL，保留原先计算结果，避免额外采样、改变默认初始化或重算出不同随机状态。03 可根据代码依赖调整最小接入方式，但必须证明下文等价性。

## 4 权重策略与基础对照

| 策略 | 实际 rep 权重 v_bt | 要求 |
| --- | --- | --- |
| baseline/off | 1 | 原始路径，可作为 identity 参照；不计算或采样 overlay |
| logging-only | 1 | 记录 D 与候选 w，训练目标及原训练状态不改变 |
| Dt | w_bt | 按当前样本/时刻对应关系使用权重 |
| constant | c | c 在正式比较训练前校准并冻结；全程、所有 b/t 相同，不在线自适应 |
| global shuffle | permutation(w)_bt | 同一次更新内对整个有效 `[B,T]` 位置集合置换，保留权重多重集合；不分层 |

logging-only 是校准/诊断路径，不自动构成第五个正式实验组。用户已确认至少比较 baseline、Dt、预先校准的 constant 和 global shuffle；alpha 与 c 的具体数值及 c 的匹配口径尚未决定。

**全局打乱的技术语义**：置换单位是本次更新的完整训练 batch 与 time 集合；当前无 padding，包含所有 rep 位置。采用普通置换，记录实际换位比例，允许不动点；不强制错排或新增时间移位等干预。分布式训练时跨 shard 的实现方式须记录并验证；仅在各设备本地分别置换不能默称为同一个全局对照。资源受限时如需改变范围，先在协议中明确该差异。

打乱在本组自己的当前 batch 上计算 w 后执行，所保留的是本组本次更新的权重集合，不是另一训练组的轨迹。使用独立、可重放的 RNG，不额外消耗 baseline 的 nj.seed/latent/action/初始化随机流。打乱分布以及各组长期的数据和参数轨迹不要求一致。

所有组保持实际 baseline KL scale、free-nats、模型规模、repval_grad、reward_grad、ac_grads、训练频率、预算、评估协议及调参预算可比。baseline 与各 overlay 模式的诊断开销单独报告。

## 5 已确认局限与诊断义务

用户已决定将以下四类列为局限并诊断，首版暂不叠加修复。具体任务干预及采样方案在实验协议中补齐。

| 局限 | 首版诊断要求 | 证据边界 |
| --- | --- | --- |
| mode 漏检 | 逐因子/整向量 mode 不一致率；raw KL 与 D 联合统计；检查 raw KL 活跃但 D 接近零的位置 | 能暴露候选漏检，不能据此认定信息对任务有用 |
| actor 早期不敏感 | 按训练阶段报告 D/w、mode 差异、actor 均值/方差差异、rep 活跃比例与任务行为 | 低 D 可能来自 actor 不使用信息、mode 相同、信息已丢失或信息无关；单项统计不足以区分 |
| 干扰误敏感 | 在可控制物理状态/历史的诊断中改变 nuisance，联报 D、权重、策略及任务行为；与任务相关线索变化对照 | 高 D 不保证任务相关性；需要可解释的配对干预 |
| 延迟线索 | 若任务有明确延迟线索，比较线索出现时和未来决策时的 D、权重及行为；若主任务没有，单列诊断 fixture/任务 | 只检查即时动作的探针可能漏掉未来需要的信息；不以即时 D 代替长期信息价值 |

任务若不能支持所需诊断，应记录诊断缺口并讨论补充 fixture；不得把“无法观察”写成“已排除”。多样本/概率加权 Dt、延迟开启 gate、actor 预训练、因子 gate 等均不加入首版。裁剪前 Dt、整个时刻统一缩放及 free-nats 非活跃区作为额外解释边界保留。

至少记录：D 与源 w 的均值/分位数；实际权重 v 的分布；raw rep KL、free-nats 后 rep_before、加权 rep_after、rep_active_frac、active_weight_mean、加权超额 KL；均值与方差差异；dyn/其他 loss、原始梯度范数及吞吐。shuffle 另记录置换前后活跃权重、加权超额 KL、`cov(v,Rrep)` 和实际换位比例。空活跃集合与退化分母标为不可用，不伪造零均值。

## 6 实验定稿顺序与未决字段

**已确认顺序**：先固定任务、观测通道、校准规则、主指标、训练 seed 与预算，再确定 alpha 和效果判据。校准规则含 c 的匹配口径；校准所得 alpha/c 及效果阈值均须在正式比较结果出现前冻结。

| 未决字段 | 需要具体写明 |
| --- | --- |
| 任务与环境 | task、环境/依赖版本、动作 head、repeat、边界和 seed 控制；真实环境可运行性 |
| 观测与诊断 | 图像/本体感觉等通道，任务线索与 nuisance 的控制方式、延迟线索诊断及数据划分 |
| 实际 baseline | 实现基准提交、模型规模、loss scales、free_nats、replay/更新配置及原有梯度开关 |
| 校准规则 | logging-only 阶段窗口、数据来源、校准 seed、D 分位数/目标权重或其他已明确规则、零分位数处理、c 的匹配口径与公平调参预算 |
| 主要指标 | AUC 或固定末尾窗口等具体选择、评估频率/次数、动作采样规则、次要指标 |
| 重复与预算 | 固定训练 seed 列表、评估及环境随机性控制、环境交互步/更新数、校准预算、停止/续跑规则 |
| alpha/c 与判据 | 根据已冻结规则得到的值、最小有意义增益、等效范围、区间/重复汇总方法、主要比较集合 |
| 工程落实（03 制定并记录） | 测试 dtype/后端与容差、KL 数值误差处理、配置字段、全局跨 shard 置换及独立 RNG 的技术方案；不重新改变已确定的数学定义、范围或验收原则 |

分位数校准公式 `alpha=(1/w*-1)/D*`、`c=mean(w)` 等仍是候选规则，用户尚未选择窗口、分位数、目标权重或匹配统计量。不得擅自给出 alpha、c、任务或预算默认值作为正式科研决定。

条件打乱不属于首版基础四组的必需实现项；如后续提出更强 M2，02 需先补充分层变量、区间、边界和退化处理等协议，再交给 03 实现对应对照。工程方案及 fixture 测试值可由 03 在已确定范围内制定，不等待正式任务/alpha/c，也不将其冒充科研选值。

**解释规则**：主要比较尚无结果；未显著不等于等效。gate 几乎无作用或只在 free-nats 零梯度区作用的运行，应说明机制没有得到充分检验。只优于全局打乱不足以排除活跃性/KL 大小等混杂；更强 M2 需要条件打乱协议与证据。具体支持/否定阈值留待上述顺序定稿，不根据正式测试结果追选。

## 7 03 必须满足的验收条件

下述条件是本次整理后的实施验收要求，尚未运行通过。03 需将具体配置字段、数值容差和技术方案记录下来；验证失败时修正实现或报告原因，不能调整科研定义来使测试通过。

### 7.1 回退与初始化

- 与明确记录的未改动 baseline 对照，固定同环境、参数、batch、carry 和 seed；覆盖完整初始化、一次 loss、全参数原始梯度和一次 optimizer 更新。
- off 必须保留参数键/初值、原训练状态和随机流；Dt 的 alpha=0 必须回退，优先静态旁路避免不必要计算或非法 `0*Inf`。logging-only 的原目标、梯度、更新及 baseline RNG 与原代码一致，新增日志不能更新原 normalization 或优化状态。
- prior=posterior、代表输入相同或 actor 输出相同应得到容差内的 D=0、w=1；显式 D=0 fixture 必须还原 rep 作用。不能用长程曲线“看起来相同”代替局部数值验证。
- 精确相同的项与允许浮点误差的项分别声明。参数创建顺序和 RNG 必须精确一致；浮点值容差按 dtype/后端事前写明。

### 7.2 解析 KL 与形状

- 解析 fixture 覆盖仅均值差、仅方差差、双向对称、相同分布、标量及多维连续 event、多 action head 总维数归一化。
- 核验 unimix、mode 与 baseline 特征拼接，结果严格对齐 `[B,T]`，无重复 event 聚合。gate 不替换 baseline 采样 latent。
- 明确非法 alpha/c、非有限参数/D/w 和微小负 KL 的处理；constant 必须有限且 `0<c<=1`，与“统一减弱”语义一致。不支持的启用路径有明确错误。

### 7.3 梯度路径与其他分支

- D/gate 分支对 actor 参数、q/p logits 和 h 的梯度为零；rep 项对原 posterior/encoder/递归 history 路径的梯度保留。
- 人为固定逐点权重时，超过 free-nats 的 rep 局部梯度精确按该点权重缩放；阈值以下为零；阈值相等处保留 baseline/JAX 约定。free-nats 关闭时保留原梯度路径。
- 同参数、batch、carry、随机样本下，dyn 独立 loss/梯度及 actor/value/reconstruction/reward/continuation 原分支保持一致；总原始梯度差等于加权 rep 分支带来的差。
- 不要求 rep 对 h 的总梯度为零，不要求训练后 prior/actor 轨迹保持一致，不用 `mean(w)` 缩放整个 batch 梯度作为验收替代。

### 7.4 free-nats、reduction 与序列

- scale 只乘一次，权重在原 free-nats 后和原 mean 前作用；不将 `w*max(R,tau)` 改为 `max(w*R,tau)`。
- 覆盖 replay_context 开关、连续 chunk、reset/terminal 和不同 imag_last；gate 对齐 world-model T，不混用 K/H。episode 首尾不是新增 rep 排除 mask。
- 对齐现有 dtype、loss shape 和 reduction 分母；若未来接口增加 padding，另行明确它与原 baseline 的有效位置规则。

### 7.5 随机流与对照

- actor probe 不采样；初始化或训练采样使用的 baseline nj.seed 序列不因 overlay 多出调用。验证 off、logging-only、alpha=0，及同参数下启用 Dt 时原前向采样的可重放性。
- shuffle 独立 RNG 能由训练 seed 和更新位置等记录信息重放，不与环境/replay/latent/action 随机流混用；本次更新源权重多重集合保留，实际权重与损失位置正确对应。
- constant 全程只用校准后冻结的 c，不偷偷改成 c(batch)/c(step)。shuffle 与 constant 的 loss/梯度实现均验证；条件打乱留待更强主张时新增协议和测试。

### 7.6 完整 agent 与交付证据

- 至少一个受支持的连续动作 fixture 完成 agent 初始化、一次 loss、全参数梯度和 optimizer 更新，所有必要值有限；默认离散 dummy 不能代替连续路径验证。
- 报告诊断的吞吐/内存开销；测试环境、提交、有效配置、seed、验证命令、结果及局限写入状态文档。不用单元测试结果宣称 M2 得到实验支持。
- 03 按明确设计做最小范围实现与测试，未定实验参数不自行填成正式决定。正式训练仍须另行完成实验协议并记录运行信息。

## 8 交接状态

已写成规范：M2 主问题、连续标准 RSSM 参考方法、四组基础对照、四类局限与诊断义务，以及回退/梯度/free-nats/随机流验收要求。

方法实现交接范围：03 阅读 `AGENTS.md`、本规范、研究与状态文档，按第 2–5、7 节实现可配置 overlay、日志和验收测试，记录具体技术方案、有效配置、基准/实现提交及验证结果。所有 alpha/c 测试值仅是 fixture，不作为正式科研选值。第 6 节未决实验字段继续在 02 讨论，03 不自行选择正式任务、校准规则、seed/预算或效果判据，不启动正式实验。

交接材料已准备。2026-10-08 先前检查现有任务及 Codex 归档时未找到 03 · 代码与测试任务，当时没有派发消息或创建任务。接收任务的指定或创建属于交接安排，不是方法确认的前置条件；不需要等待正式 alpha/c 才实现可配置模块。

本次一致性核对仅修改 `docs/RESEARCH.md`、`docs/METHOD_SPEC.md`、`docs/EXPERIMENTS.md`、`docs/STATUS.md`；用户已要求提交这四份文档并推送 main。提交与远端状态以 Git 记录及完成报告为准，不改算法代码、不启动实验，验收条件尚无实际通过记录。
