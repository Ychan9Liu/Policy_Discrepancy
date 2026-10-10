# DL 长后果机会诊断

版本：**`DL-opportunity-r1`**，2026-10-10。状态：协议已冻结，尚未执行。本文是当前用户授权内的 **Agent 自主设计**，不是用户逐项确认的设计，也不是方法有效结论。只新增一个外部后果诊断对象；不修改 `DL-protocol-r1`、其阈值、DtLatch H1 信号、gate、alpha/rho、训练配置或原 M2 作业。实现与实际运行须另记完整 SHA。

## 1. 原因及问题

原 r1 两个固定 checkpoint 的真实主诊断已经完成。300k/1m 的 blind 正例分别为21/4个（9/4 episode），cuecost 正例为26/7个（8/6 episode），未达到原128位置、8 episode的充分覆盖。活跃 rep 位置为768/760、均覆盖24 episode；超额KL释放比例为0.12944%/0.22305%，未满足原1%门槛。这是活跃覆盖充分后的幅度判据未满足，不能统一写成缺数据。条件P释放偏差11.883%/62.730%则属于对应关系对照解释力不足，不自动构成方法失败。

独立 CPU 数组审计显示：q/p 首均值动作有差异，但 H1/H10 真实收益通常很小；300k/1m 的正 log-score margin 中位数为0.006755/0.000426。u_q中位数为0.97896/0.99959，因此绝对 exp 抑制不是多数位置弱信号的普遍原因。1m的 next reward≥.99占85.03%，300k只占10.16%，奖励上限是1m的可能解释，不能解释两个阶段全部现象。去除绝对支持因子的事后数值反事实不能作为候选验收；1m该反事实释放的85.17%来自u_q<.5的21个位置，其H10真实收益均值为负。

证据为 ignored 镜像 `analysis/outputs/dl-prevalidation/cp300000-analysis-r1.json`、`cp1000000-analysis-r1.json` 及对应32个score NPZ；独立审计为 `bootstrap/r1-independent-scientific-audit.json`（SHA256 `6e37754e599b9ed98c26d4c587047d4f82df7c35384294e8dd23d29feb0f5da1`）和 `bootstrap/r1-independent-opportunity-decomposition.json`（SHA256 `55024f6f4bd3a145eb8bd7e71f6fd9ffe3e3c093f29fd15b903d456907e46d21`）。两个阶段来自同一baseline训练seed，不能称独立模型复现。

本诊断的研究问题是：**在同一真实状态，只改变后验/先验引起的首动作，之后采用共同记录动作，H10之外是否存在持续的真实累计收益；这类收益是否与当前body外观/纹理剥夺的实际损害同时出现？** 后者的交集接近“当前观测提供共同历史之外可被actor利用的信息”，但只针对这一有限剥夺对象。剥夺图像可能分布外，交集仍不证明posterior因果正确、潜在特征必要或保护压缩必然改善训练。

root已检查两个stage的18个预定calibration帧（各ep4/6/10×frame50/500/900）：protected RGB及cue区域外RGB保持，背景只改天空、机器人/ground/边缘/shadows保留；gray-body仍保留当前机器人轮廓、四肢姿态和地面shadows。mask随当前body shape变化，因此字段 `cue_deprivation` 主要是外观/纹理剥夺，**不是完整关键运动学信息剥夺**。18帧不是all-frames保证。原证据为 `analysis/outputs/dl-prevalidation/operational-evidence-r1/cp<stage>-calibration-mask-montage.png/json`。低cuecost可能源于线索仍可见，不能直接推断prior足够；本轮继续冻结同gray-body干预，不暗换为真正遮挡。

工作假设是短窗口/平均奖励口径可能漏掉持续的一次动作影响。反例是延长后收益仍小、续行动补偿差异、先验已有足够信息、q/p actor无法利用新信息、cue遮挡只制造分布外错误，或H1 reward支持与长收益无对应。**不预设长窗口会支持假设。**

## 2. 冻结对象、来源与样本

| 项目 | 本轮固定设计 |
| --- | --- |
| 任务/模型 | `dmc_quadruped_walk`；size50m、64×64 RGB、proprio=False、repeat=1、camera2，原baseline实际配置不变 |
| 来源 | 原baseline seed0，固定300000和1000000训练动作阶段；不按return选checkpoint |
| 新轨迹seed | `20261012`，与r1及资源pilot不同；使用既有 `episode_plan` 的环境seed、policy_seed及8/24分割规则，完整写入plan |
| 轨迹 | 每阶段32个完整episode：8 calibration、24 blind；两阶段分别分析，禁止混合掩盖失败 |
| 位置 | 每episode预定16个等间隔位置，要求有后续100动作；沿用 `sample_positions(length, maximum=16, horizon=100)`，位置在读取后果前确定 |
| 主窗口 | H=100；H1/10/50仅为预定描述性prefix，不根据其结果替换主窗口 |
| 信号 | 原 r1 同h、当前q/p mode、记录首动作与真实 next reward 的H1探针；alpha=20、rho=.5；所有score/gate detached |
| 干预 | 原clean、background_a、background_b、cue_deprivation四种图像；背景geometry/ground保护与gray-body外观/纹理剥夺不改，保留轮廓/姿态这一已知限制 |
| 随机流 | 每帧clean B1推进历史；诊断B4 carry丢弃；无未来观测进入t处信号；完整环境与policy seed记录 |
| 统计 | episode为重采样单位；2000次bootstrap，分析seed `20261012`；连续分布、分母、实际coverage均保存 |

300k checkpoint SHA256：`2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9`；1m：`c9c893081485c8ad22468611d046bc5bf9fbe463d55996fda15cf85b2a0323fe`。config SHA256：`11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445`。参数digest分别为 `14859c4e129b51aa488c10eab09f16d56fd110c76239b2c9072f415829d111c2`、`fb8c429cb13861d90f2aeba32842eda8958bd17d1b80f2dfa15c0363aca7549d`。来源及完整训练/检查点counter区分见 `docs/dl_checkpoint_sources.json`；独立来源原证据为 `analysis/outputs/dl-prevalidation/bootstrap/source-checkpoints-scientific.json`。检查点actions=18769/62563是policy calls，不能冒充300k/1m训练阶段。

32×16×2阶段最多1024位置，其中每阶段blind最多384位置。无合法100步窗口、缺episode、参数digest不符或文件hash不符时停止相应阶段，不事后补选有利位置，不复用旧r1数据凑coverage，不混入calibration点。

## 3. 真实后果及实施验收

在每个t的同一完整物理状态上，分别执行各观测变体q actor的clipped mean首动作，以及共同的p actor clipped mean首动作。之后99步使用该episode记录的共同动作序列。每条分支恢复 `mjSTATE_INTEGRATION`、task/RNG、step/reset counter等，不在分支之间延续状态。所有prefix来自**同一条100步分支**，不能分别模拟或混用不同动作。

主要解释使用clean-q、cue-q、p三条分支；保留两个背景q分支的H100后果便于同时检查干预产生的实际动作影响。它们是有限候选动作后果，不是最优控制、完整q/p策略、closed-loop rollout或长期policy价值。后续共同动作来自原clean行为，可能对某分支有利或补偿错误；此限制随结论保留。

另以实际记录首动作及99个记录续动作，从同snapshot重复完整100步，逐步对照采集reward，而不只核对第一步。每条候选也独立重复100步，并比较整个reward序列、最终integration/task/counter。保持原float64复原容差1e-8；实际采集float32标签以其原编码精确对照。任何失败停止标签比较，不能放宽容差或仅核对累计reward。

必须保存每位置各候选的float64逐步reward（shape明确为位置×候选×100），H1/10/50/100 prefix sum/mean、候选首动作、参考和候选终态hash、最大恢复误差及原r1全部信号。H1 prefix须与同分支第一reward严格一致；sum与mean须可独立NumPy复算。提前terminal不能跨reset补齐；记录并停止该不符合窗口的比较，不能填零掩盖。

实施只新增外部后果字段/独立机会分析，不把 `--horizon 100` 原r1通用analysis的“平均≥.01”标签误当本协议标签。原r1分析必须仍有明确原版本记录；本轮analysis写 `DL-opportunity-r1` 及本节因果对象，并明确 `proceed_short_training=false`。

H1信号仍以真实**记录动作**及其next reward评分，不能用候选q/p首动作的不同reward悄然改写e。所有图像变体从共同pre-observation carry出发，prior动作/std/logits保持逐点相同；checkpoint文件、模型参数、config来源在collection和score之间一致，score完参数未变。prefix字段不能改变原history/model随机流。default-off/边界/梯度及size50m工程证据继续由原验证记录承担。

## 4. 独立机会标签和充分覆盖

对h∈{1,10,50,100}，令 R_q(h)、R_p(h)、R_cue(h) 为对应prefix真实reward累计。联报：

```text
U_h = R_q(h) - R_p(h)
B_h = R_q(h) - R_cue(h)
mean_gain_h = U_h / h
mean_cuecost_h = B_h / h
```

预定主标签仅用H100：`U_100>=.1`为positive，`U_100<=0`为negative，其余为gray；`B_100>=.1`为useful-cue；同时满足positive与useful-cue为joint opportunity。`.1`是固定一次动作累计效果尺度，与原H10平均.01的累计尺度相同，但在H100意味着平均.001，**是新的研究对象，不是r1门槛放宽或r1通过**。不将H1/10/50的好结果替代H100；均发布有符号连续分布、分位数和逐episode摘要，禁止只发布正部或好点。

每阶段分别沿用128位置、至少8 episode的充分覆盖标尺：positive和negative均需达到才足以开展新的信号判别；useful-cue也需达到才足以审查cue证据；joint opportunity需达到才称有足够样本检验“cue必要且q相对p有新增控制收益”的目标情形。它们是本轮事前覆盖选择，不是普适功效保证。达不到时区分“少量真实机会存在”与“充分覆盖未建立”，不能把一些确定的正例当作可正式实验准备就绪。

所有24 blind episode中的机会率及episode-bootstrap区间必须联报，避免忽略稀少性。交集覆盖不足时，不用cue正例或q/p正例单独替代交集；各阶段不能相互借样本。两阶段同训练seed的局限保持。

## 5. 信号关系及裁决

在新机会标签下，报告C=1−v、D/fD、e、signed support、loss_gap、u_q/u_p、K；positive/negative/gray/joint各自的分布和活跃超额KL贡献，以及与D/K冻结分层的关系。report episode-bootstrap的AUROC(C/D/e/K)、配对C−D差及其覆盖；不足时保留点估计但标为证据不足。原r1校准S/P/W若继续计算，只用8 calibration episode且完整记录hash，不把这轮机会校准替代既有r1验收。

实际释放比例使用 `sum(m*(K−1)+*C)/sum(m*(K−1)+)`，如给区间，须在每次episode重采样后重新计算该比率；`mean(C*(K−1)+)`区间不是该比例区间。保持原1%未满足事实，不根据长窗口重新解释或覆盖。背景、mode、fixed-action H1、label-shuffle等原诊断继续联报，不能因新机会对象而省略反例。

本轮结论分为以下情况，均**不能启动短训练或宣称方法有效**：

| 真实证据 | 下一步解释与条件 |
| --- | --- |
| 两阶段有足够joint、正负覆盖 | 长窗口提供了有样本基础的目标机会，可设计新代理候选；先证明H1是否漏检以及额外信号增量，不默认需改评分 |
| 只有一个阶段有足够覆盖 | 可以开展明确限定模型阶段的候选设计；另一阶段局限保留，不能声称两阶段稳健或跨训练seed复现 |
| q/p增量有覆盖，但cue/joint不足 | 一次动作机会有证据，新增观测必要性仍未充分；不得称已证明应保护后验 |
| cue有覆盖，但q/p增量/joint不足 | 观测剥夺有损害，先验可能已够用或遮挡分布外；不能以cue损害替代posterior相对prior新增价值 |
| H100仍低覆盖 | 此有限对象不足以承载目标验证；停止该长窗分支，不能在同数据继续延H/改阈值/选episode直到通过 |
| 长机会存在，H1 C低或C−D无增量 | 才考虑另立multi-step支持、proper-score margin或MC概率混合候选；一项项归因，另用新数据验证，保留共同错误/动作泄漏/截断偏差反例 |
| 长机会存在，H1 C已有辨别但作用弱 | 可研究有理由的强度/压缩机制问题；不自动提高rho，也不能用强度对照失效解释选择有效 |

不存在充分覆盖不等于数学上不存在有价值信息，也不否定整个DL目标。gray-body保留了重要形状线索，故joint不足不能单独否定关键运动学信息的价值。若此分支未满足，下一必要研究对象是另立**真实key/遮挡或真实关键状态/有限短控制**：候选遮挡应使用静态、只由calibration定义的rectangle，不随当前shape泄漏姿态，并审查残留body/shadow信息；或采用合法physical-state挑战，先确定视觉可获取性及q/p actor能力。正常接收真实观测的closed-loop诊断也须另立因果对象。不得永久剥夺p未来观测并称完整策略价值，也不得把物理扰动挑战上的支持泛化到clean常规轨迹。新对象、新数据和停止规则另立版本，不能用反复延H替代有效线索干预。

本轮不新增5个固定动作的H100搜索：它只说明有限动作集合有控制敏感性，不能证明q/p能利用新增信息，且扩大对象与预算。保留既有零动作及±.25固定动作的真实H1评分；其H1 reward span较小的事实不能证明更长控制后果无敏感性。需要独立有限控制证据时另立协议，不悄然补进本轮。

## 6. 资源、预算和执行顺序

资源沿用用户已指定sv3物理GPU4–7；计划GPU4承担300k、GPU5承担1m，每张卡仅一个对应阶段任务。仍须启动前fresh占用、实际hostname=lyg0326及CUDA/EGL UUID映射核查；已有占用不共享、不停止、不擅自换卡。此文档不是占用快照。

最多64个新完整episode、1024位置，不训练、不更新optimizer。外部5候选×2重复×100步加参考记录动作×2重复×100步，为至多1,228,800物理步；原5固定动作H1×2重复再加10,240步，总上限**1,239,040物理步**，不含新episode采集本身。所有prefix共用H100序列，不能为各prefix另跑而超过预算。若只存三条核心候选应另记录实际更小成本，不把它冒充五条后果齐全。

r1两阶段score分别167.27/159.84秒，本轮position减半但真实后果H增加10倍；不能线性推算整个walltime或把GPU前向与CPU physics混为同一吞吐。工程预留建议为每阶段一张卡最多1小时、合计2 GPU小时，含编译/采集/评分；这是有限运营预算而非实测ETA，超出则保存进度和原因后审查，不增加episode或共享卡。峰值显存/实际CPU物理耗时/渲染时间须实测。只需单卡模型，无多卡同步。

先在CPU使用真实MuJoCo证明H100参考复原、完整分支重复和prefix字段正确（非mock）；再完成代码review、静态H1信号不变/字段形状/终止规则的必要检查。确定SHA和新目录后，启动fresh GPU采集/评分。失败使用新attempt，不覆盖原r1、失败证据或旧source。

服务器产物根建议 `/data/Policy_Discrepancy/runs/dl-opportunity-20261010-<sha>/`，分别 `cp300000-collection/score/analysis-attemptNN`、`cp1000000-collection/score/analysis-attemptNN`。本机只镜像小结果到 ignored `analysis/outputs/dl-opportunity/`；完整plan/complete、依赖、source hashes、资源、退出码、逐episodeNPZ及独立CPU复算报告保存。若跨日执行按实际日期命名，不改冻结版本/seed含义。

## 7. 交付和完成边界

本轮交付应包括两阶段真实H100机会分布及覆盖、joint标签、原H1信号对应关系、复原/参数/hash工程证据、失败及局限、下一候选分支的理由。结果不改写原r1失败事实，不作正式性能结论，不进入正式短训练。

完成此诊断只解决“是否有足够可利用机会、H10是否漏检”的一部分。形成可正式实验方法仍依赖新候选的工程正确、独立信号增量、活跃压缩机制、有效S/P/W及替代解释对照、条件短训练与正式协议；这些不能由本文件或一次长窗口正例代替。需要否定分支时直接记录否定，不把目标改成仅完成工具。
