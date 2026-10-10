# DL r1 真实诊断裁决

版本：**`DL-r1-results-v1`**，2026-10-10。**当前裁决：尚未形成可正式实验的方法，不进入短训练。** 两个冻结模型的真实r1诊断及独立CPU数组审计已完成；活跃释放幅度门未满足，label/cue覆盖及第二信号增量证据不足，P对应关系对照不充分。背景平均新增保护满足原上界，18个预定校准帧已由root视觉检查，但尾部反例及全数据合法性仍需审查；gray-body干预保留了轮廓/姿态，不能称完整关键运动学信息剥夺。本报告不把工程正确、局部公式或少量高AUROC点解释为方法有效。

本轮配置与判据为授权内Agent自主设计，不是用户逐项确认的科学结论。原协议见 [DL_PREVALIDATION.md](DL_PREVALIDATION.md)，独立理论审查见 [DL_PREVALIDATION_REVIEW.md](DL_PREVALIDATION_REVIEW.md)。本报告及小型精确汇总 [dl_r1_scientific_summary.json](dl_r1_scientific_summary.json) 新增归档，**不覆盖原analysis、score、pilot或失败输出**。

## 1. 证据身份与实际执行

科学方法代码：`DL-code-r1`，Git **`9e6da39e59ad347f53db93af649dc8e8b63425ad`**。两条实际主score执行均为 **`1280bea7bf409e9cc6fa975704bcb370b8880a0a`**（`DL-engineering-r2`），该版本只修资源判占用，不改变r1科学方法。目录名称中的9e6da39不冒充实际执行SHA；source checkpoint的baseline源码另为 `48d71f1541721208322442f5a1d5977124f3e888`。

共同配置：`dmc_quadruped_walk`、size50m、64×64 RGB、proprio=False、repeat=1、camera2；alpha=20、rho=.5、free-nats=1、原loss scale/gradient flags。每阶段32完整episode（8 calibration、24 blind），每episode32预定位置，H10。表中资源检查时间来自score plan，只是当次资源记录，**不是当前占用或精确作业起止时间**。

| 固定训练阶段 | 更新counter / checkpoint actions | score物理GPU / hostname | score plan资源检查UTC | score循环耗时 |
| --- | --- | --- | --- | --- |
| 300000动作 | 74493 / 18769 | sv3 GPU4 / lyg0326 | 2026-10-10 09:40:56.214707 | 167.268517秒 |
| 1000000动作 | 249493 / 62563 | sv3 GPU5 / lyg0326 | 2026-10-10 09:18:49.718845 | 159.840813秒 |

checkpoint的actions是policy calls，训练阶段依据 `evaluation_snapshot.json.actual_action_step`，两者不能混用。score循环timer不含采集、此前模型加载/初始化；循环内首次JIT编译可能计入，不是总GPU walltime。完整开始/结束、exit-code、资源和设备证据以运行目录及 [DL_VALIDATION_LOG.md](DL_VALIDATION_LOG.md) 为准。

两个模型都是baseline训练seed0的不同阶段，**不是两个独立模型replicate**。各自24独立episode的bootstrap区间只反映该固定模型的轨迹变异，不代表跨训练seed泛化，不能通过混合两个阶段扩大n。

| 来源 | 300k | 1m |
| --- | --- | --- |
| checkpoint SHA256 | `2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9` | `c9c893081485c8ad22468611d046bc5bf9fbe463d55996fda15cf85b2a0323fe` |
| 参数digest | `14859c4e129b51aa488c10eab09f16d56fd110c76239b2c9072f415829d111c2` | `fb8c429cb13861d90f2aeba32842eda8958bd17d1b80f2dfa15c0363aca7549d` |
| score plan SHA256 | `ea32e3e53b6003bef6a588130f7470ae273a1d60bd1ae728f38e97f901a0eb08` | `8d4f461b8475017c56119fe998a36d337d07841603d169deb7ee4d66ad5c0ffc` |
| score complete SHA256 | `82e740259d137d10895d84fdad9ab389a14897e374d26e9b57dee3821dc3399d` | `1f72668bb077736e520986a436f0834e83aafd635f721242caa077f16d15bdea` |
| 原analysis SHA256 | `abb37f413fb2293a0c0dd132525e7cacfd5b1b67fd11921fbee5c622d7ea230c` | `02461c43461d05d21d9802536c4e51b9ced776a65d5ecc8ad5ea7458b4311263` |

共同config SHA256：`11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445`。来源路径、文件大小、metadata、done/latest及counter独立审计见 [dl_checkpoint_sources.json](dl_checkpoint_sources.json)，其SHA256为 `b7d81f8a123ca44a8abdddf37da535980b4586e116e0c3349c6755f4b4da1600`。

实际产物根 `R=/data/Policy_Discrepancy/runs/dl-prevalidation-20261010-9e6da39/`；主score分别为 `R/cp300000-score-attempt01/scores`、`R/cp1000000-score-attempt01/scores`，采集为对应 `cp<stage>-collection-attempt01/collection`。本机只读镜像为 `analysis/outputs/dl-prevalidation/cp<stage>-score-attempt01/scores` 和 `cp<stage>-analysis-r1.json`。

独立CPU审计核对两阶段各32个score NPZ的sha、plan→complete→analysis哈希链，均匹配；collection/score的checkpoint及参数digest一致，score complete报告参数未变，真实分支重复最大恢复误差均为0，reward超出bins计数为0。这支持这些冻结诊断的文件身份和恢复工程证据，**不能替代完整训练梯度/optimizer验收或证明方法有效**。

真实size50m工程工作独立记录在 [DL_VALIDATION_LOG.md](DL_VALIDATION_LOG.md)：GPU7 attempt03在 `55a527c620fb14339ee00f76f045a5dfddc81068` 五worker完成，paired检查因logging两个optimizer状态hash不等而exit1；总体DL-B未通过，工程修复待重验。不得据已有tiny Agent fixture、冻结score前向或本报告的物理复原标DL-B通过。后续状态以该日志和真实attempt证据为准。

## 2. 原判据下的裁决

原独立外部标签为：从同一完整真实状态，q/p actor **clipped mean首动作**，再用共同记录的9动作；H10**平均**reward差≥.01是positive，≤0是negative，中间gray。不是累计reward、最优动作或完整策略价值。cue另以clean-q对cue-deprived-q的H10平均reward损失≥.01定义，不用D/e/K生成标签。

| 证据 / 原门槛 | 300k | 1m | 当前裁决 |
| --- | --- | --- | --- |
| blind位置 / episode | 768 / 24 | 768 / 24 | 采样完成 |
| positive / episode；要求128 / 8 | 21 / 9 | 4 / 4 | 正例覆盖不足；不因1m高AUROC宣称成功 |
| negative / episode；要求128 / 8 | 377 / 24 | 393 / 24 | 负例覆盖充分 |
| gray | 370 | 371 | 保留，不当作negative补覆盖 |
| useful-cue / episode；要求128 / 8 | 26 / 8 | 7 / 6 | cue覆盖不足 |
| positive∩useful-cue | 9 | 1 | 描述性目标交集，不能替代原标签或充分证据 |
| active `m & K>1` / episode；要求128 / 8 | 768 / 24 | 760 / 24 | 活跃覆盖充分 |
| weighted-excess释放；要求≥1% | **0.12944%** | **0.22305%** | **幅度门未满足** |
| P protected实质权重改变；要求≥50% | 99.4778% | 99.4872% | 能改变权重 |
| P weighted-excess偏差；要求≤10% | **11.8832%** | **62.7301%** | **对应关系对照解释力不足**，不自动当方法失败 |
| 背景mean ΔC上界；要求≤.01 | A .001840，B .001403 | A .000868，B .000738 | 平均上界支持，条件于干预合法性；尾部/视觉待审 |

原analysis把 `screening.active_release` 写成 `inconclusive`。这是状态分类错误：活跃覆盖充分且释放比例可得、低于原1%，应为 **`not_supported`（当前幅度判据未满足）**。root已修当前分析源码的分类分支，**没有改变阈值、公式或原数值**；上述原analysis文件与hash保留，不在旧文件上覆写新状态。本报告及科学小汇总显式记录该纠正，避免把“门槛失败”与“信息不足”混在一起。

独立episode-bootstrap在每次重采样后重新计算释放比率，得到300k **0.12944% [0.08969%,0.17679%]**，1m **0.22305% [0.10687%,0.36124%]**。两上界也低于1%；这些区间是事后独立补充，不修改原判据。原analysis的 `weighted_excess_release` 是 `mean(C*E)` 区间，不能误读成百分比；比例为 `sum(C*E)/sum(E)`，E=`m*(K−1)+`。

## 3. 信号、cue和第二来源增量

以下为原episode-bootstrap 95%区间；1m有4个正例，2000次draw中1978次同时含两类、可定义AUROC，不能拿1978有效draw代替原覆盖要求。

| 统计量 | 300k：点 [95%区间] | 1m：点 [95%区间] |
| --- | --- | --- |
| AUROC(C) | .5726 [.4409,.7015] | .9711 [.9330,1.0000] |
| AUROC(D) | .6967 [.5946,.7782] | .7545 [.4970,1.0000] |
| AUROC(e) | .5425 [.4157,.6712] | .9478 [.8439,.9974] |
| AUROC(K) | .6308 [.5231,.7672] | .9447 [.8601,.9943] |
| 配对AUROC(C)−AUROC(D) | −.1242 [−.2929,.0217] | +.2166 [−.0381,.4870] |
| 配对AUROC(C)−AUROC(reward-only) | +.0301 [−.0008,.0560] | +.0232 [−.0122,.0891] |
| useful-cue上的clean−deprived C | −.003123 [−.007366,.000352] | +.011505 [.000936,.030043] |

300k C点估计也未满足原.60、下界>.5的观察条件；由于正例覆盖不足，不能把该结果当充分的普适负效应证据。1m的C及cue点/区间看似支持部分观察条件，但正例/cue覆盖未满足，仍不能认定信号有效。两阶段C−D区间都没有下界>0，C−reward-only也未确立；**第二来源提供D以外增量的主张尚未建立**。

冻结D/K层内的正例更少：300k六层分别只有1/2/3/4/3/8个，1m六层只有0/0/0/2/0/2个。不能以某层漂亮AUROC证明条件增量或用K-only高点估计决定新门控。S/W只在各阶段8 calibration episode按原公式冻结，不能把离线匹配叫全程训练梯度匹配；P虽然改变了原保护权重，但释放质量偏移超10%，不足以排除粗粒度强度解释。不存在DL六组×3seed短训练结果。

高D的std主导位置占比均未达到“多数”阻断条件（300k约1.574%，1m为0），说明当前均值动作后果没有因大范围std主导而被直接拒绝。**这不证明策略方差差异没有控制价值，也不把均值动作标签扩展为完整策略分布验证。** 局部近tie top-two扰动在399/454个位置有作用，受扰动点的平均|ΔC|约.000249/.000200；这里只审代表状态敏感性，不给真实性标签，不能凭小平均误差排除个别mode反例。

## 4. 背景上界与仍存在的反例

平均统计通过仅表示两种规定背景干预的**平均新增保护**上界≤.01。它不约束所有单点、每episode或其他背景，不等于“已消除背景假保护”；该门槛本身也不排除整体保护很弱这一替代解释。原冻结平均门槛保持，以下尾部只作为反例与QA重点联报，不事后替换原门槛。

| 阶段 / 背景 | mean ΔC | 最大ΔC / episode,位置 | ΔC>.01位置 / 768 |
| --- | --- | --- | --- |
| 300k / A | .001012 | **.178864 / ep7,t1** | 28 |
| 300k / B | .000752 | **.069760 / ep18,t351** | 28 |
| 1m / A | .000376 | **.116874 / ep25,t1** | 13 |
| 1m / B | .000346 | **.093309 / ep25,t1** | 10 |

root已实查两个stage各前三个预定calibration episode（4/6/10）的frame50/500/900，共18帧。样本的protected geometry/background RGB逐像素保持、cue区域外RGB保持；可见背景替换只改天空，机器人、ground、边缘和地面shadows保留。证据位于 `analysis/outputs/dl-prevalidation/operational-evidence-r1/cp<stage>-calibration-mask-montage.png` 及同名JSON，来源记录为 `cp<stage>-calibration-visuals.json`。这是预定校准样本的像素/视觉QA，不能扩写成所有frame保证；上述blind最坏点及全数据合法性仍待审查，pilot三帧也不能替代主数据。

**已观察到的干预限制**：gray-body mask随当前真实body shape变化，灰色机器人仍显示轮廓、四肢姿态及地面shadows。故历史字段 `cue_deprivation` 实际主要剥夺body外观/纹理，不能叫完整运动学或全部关键状态信息剥夺。它仍可能分布外；低cuecost可能因为控制线索仍可见，这是工作假设，**不能直接推断prior足够或观测没有关键价值**。有cue损害也不能直接标posterior正确。现r1和H100机会协议保持同一gray-body干预，不暗换遮挡方式。

## 5. 低作用的量化与未证实解释

已验证的数值事实：300k/1m的C总体均值为.0009393/.0005326，e均值.0103277/.0028000；K中位10.5604/4.1765，D中位.005545/.010301。当前弱释放不是大范围mask不可用或free-nats未活跃造成。q/p首均值动作MSE均值.027301/.032377，策略动作确有改变；真实H10平均收益中位却仅+1.004e−6/−2.542e−6。动作差异本身不是控制收益。

正log-score margin的中位为.006755/.000426；u_q总体中位.978964/.999591，有正margin位置的u_q中位.977934/.999731。故多数位置的q/p奖励预测几乎等同，去除绝对支持衰减并不会自动解决这一原因。1m next reward≥.99占85.026%，300k为10.156%；**奖励上限影响1m机会**是工作假设，不能单独解释300k或证明后验无用。

对δ=`d_p−d_q>0`，原e满足 `e=u_q*(1−exp(−δ))`。独立审计仅作数值反事实，去掉u_q因子为 `e_rel=1−exp(−positive_part(δ))`，不改rho/alpha，得到释放比例：300k .28019% [.19474%,.37164%]，1m 1.08923% [.33269%,2.05985%]。这是**事后候选诊断，不是采用/校准/通过**。

特别是1m反事实释放的85.169%来自u_q<.5的21个位置，其真实H10平均收益均值−.001197；反事实总释放中46.912%来自negative、42.928%来自gray，仅10.160%来自4个positive。300k反事实仍远低于1%，58.274%质量在negative。因此“去绝对抑制”会强化两个模型都不充分支持真实reward的尾部，尚无充分理由将其认定为正确保护改进。这些类别质量分布也受标签稀少影响，不单凭它判定所有保护错误。

log loss可作为proper score；proper要求真实分布使期望得分最大，定义见[Gneiting与Raftery原论文](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)。按该定义，本报告判断相对差经此非线性/截正后的e_rel**不是proper scoring rule**，不能把“proper log-score差”偷换为“proper gate”：例如固定p=(.5,.5)、真实分布F=(.9,.1)，该e_rel期望在q=(.99,.01)约为.44545，大于真实q=F时的.40。原r1的exp兼容性差和截正e也不应称proper score。相对优势可以在q/p都错误时很大，有限MC噪声的正部还有偏差；未来若采用有界quadratic/Brier score差、logmargin或概率mixture，须另立方法、证明语义并用新数据验证，不能为达到1%选择变换。

**工作假设而非结论**：H1信号条件于记录行为动作及其真实next reward；外部H10标签比较q/p clipped mean首动作后果，动作与窗口对象不同。行为动作可能泄漏cue，H1可能漏掉延迟后果，共同记录续行动可能补偿首动作，modal latent可能偏离实际随机策略。当前数据不能区分这些解释与任务/阶段本来缺少posterior增量、actor利用能力不足或共同模型错误。固定五动作H1真实reward span≥.01仅覆盖300k的75位置/13 episode、1m的15位置/14 episode；有限集合低span不能证明长控制也无机会。

## 6. 当前决定与下一必要步骤

**已确定的裁决/设计**：保留r1原始结果及失败；不降低阈值、不调rho/alpha、不用新命名覆盖旧输出；不启动DL短训练。下一阶段已冻结 [DL_OPPORTUNITY.md](DL_OPPORTUNITY.md)，版本 `DL-opportunity-r1`，Agent授权内设计：同两checkpoint、新seed20261012、每阶段32 episode/8 calibration/24 blind、每episode16预定位置、同状态首动作干预与共同99续动作，H100累计收益为主机会对象，H1/10/50为预定prefix描述，原H1信号不变。

该新对象positive为H100累计q−p≥.1，cuecost累计clean−deprived≥.1，另报告交集及连续分布；128位置/8 episode充分覆盖分别审计。它与r1的H10平均≥.01不是同一验收对象，**不得称r1改为通过**；也不是closed-loop完整策略价值。主机会仍少时停止该长窗分支，不在同数据延H或筛选好点到通过。

**尚未采用的建议**：先独立证明存在足够目标机会，再决定是否需要多步reward支持、proper-score margin、MC概率混合、真实关键状态或更完整控制对象，参见 [DL_REVISIONS.md](DL_REVISIONS.md)。若H100机会存在而H1对应不足，才有针对代理错配的修订依据；若长机会也少，不能靠制造更强内部数值建立目标。若交集不足，下一必要的真实key/遮挡诊断应另版采用静态、仅由calibration定义的rectangle（不随当前body shape泄漏姿态）或合法physical-state挑战，并审查残留轮廓/shadows与可获取性；不能通过反复延H代替关键线索操纵。后续新候选必须有新版本/数据、工程验收、信号增量、活跃压缩机制、有效强度/对应关系/reward-only对照与条件短训练，正式性能仍待正式实验。

本报告只完成r1真实诊断的当前裁决。**形成可正式实验方法的完整目标仍未完成，继续必要验证与改进**；这不是将目标缩成工具交付，也不以新协议文件存在宣布准备就绪。

## 7. 可复查的小型证据

机器可读归档 [dl_r1_scientific_summary.json](dl_r1_scientific_summary.json) 不含参数、回放或大数组，记录来源/score/report哈希、原始screening和纠正、独立比率区间、背景尾部、18个校准帧QA来源及事后反事实。当前SHA256为 **`416dcc96c5205652adb6981119c5ceedec9f68c9c7225645d2697d96131e743e`**。

独立原证据：`analysis/outputs/dl-prevalidation/bootstrap/r1-independent-scientific-audit.json`，SHA256 `6e37754e599b9ed98c26d4c587047d4f82df7c35384294e8dd23d29feb0f5da1`；`r1-independent-opportunity-decomposition.json`，SHA256 `55024f6f4bd3a145eb8bd7e71f6fd9ffe3e3c093f29fd15b903d456907e46d21`。这些是公开主结果后的独立CPU审计，不是修订候选的新盲测，不能再次用于调参后称未见证据。
