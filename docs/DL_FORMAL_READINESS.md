# DL 正式实验准入与因果证据边界

版本 **DL-readiness-r1**，2026-10-15。本轮为完整预验证目标内的独立0GPU审查；不改变正在运行的 [DL_R4_UTILITY.md](DL_R4_UTILITY.md)、实际5ce源码或任何停止门，不读取其中间科学值。下述下一干预均是**建议/需求，尚未采用的新实验**，不是用户逐项确认或已支持的算法。

**16:12终态优先**：上述“正在运行/尚未完整”的初始快照已由真实cal8失败和独立CPU证据更新。
原5ce在null门停止，4ddad2e取证/23null/4600CPU物理步确认只有e2/t64平均差.00319120049295762超.0025；
held没有生成，原32EP对象未成功。首差图像/logit先于latent/raw和积分状态只是存储序列顺序，不是renderer根因或方法阴性。
详见[DL_R4_CALIBRATION_AUDIT.md](DL_R4_CALIBRATION_AUDIT.md)。当前无GPU作业，下一先CPU明确可靠测量对象/资格条件，
不因本失败改门、补held或自动采用下述B/N与替代标签方案；整体目标active。

## 目标的完成条件

正式实验前应形成：定义自洽且有可证伪主张的候选；真实接口正确；新信号对独立任务后果有适用范围明确的增量；正常压缩/保护有对应的因果机制证据；关键替代解释有有效对照；能落地的正式协议及判败条件。不能用一个toy、工程通过、非零gate或一条回报曲线替代整条链。

**不循环要求正式结论已被证明**：跨任务、多训练seed的稳定性能可保留为正式实验要检验的主假设。缺少这类最终结论不应单独成为永远无法进入正式实验的理由；正式前仍需足以支持“值得正式检验”的有限机制与有效对照证据。原文档所列跨seed/任务“尚未验证”保留为事实，但不等于必须先完整执行正式实验才可准入。

| 用户目标 | 已有证据及其范围 | 仍缺的最小对应证据 |
| --- | --- | --- |
| 主张与判据 | RESEARCH/DL-33和R4固定当前动作U_act、总观测U_obs、H100/top2/主比较；CPU解析反例及固定rho可行域已独审 | 依据真实结果保留或修订候选，明写该候选所保护的对象及判败反例，不将定义当成立 |
| 最小探针 | ab665a8的真实size50m五组单步/局部梯度/旁路证据；R4 d61 Linux51项与5ce相关19项 | 当前N1M完整生成、每分支身份/readonly/真实控制/终止契约与CPU独审；tiny或stub不代替实际 |
| 独立信号判别 | 旧reset共享七控制读出有有限q信息增量；D+K/D+S程序未建立增量且覆盖不足；R4尚运行 | 当前固定held程序的完整结果及适用限制；若阳性，只支持这一程序预测增量，不是信息真值或部署gate |
| gate与压缩 | r3实际非零保护、active KL、超额损失释放.3466%；旧局部R−B收益未建立 | 普通active位置上的同对象学习干预：实际rep作用→可访问任务信息→真实行为；不能只看KL或loss |
| 替代解释 | 固定N批次无法靠合法rho匹配强D已证明；源动作/模式/历史/时间和渲染边界已辨明 | 对所声称机制，逐一建立合法强度S、有效对应P、reward-only W等对照；无力的对照不算排除 |

证据位置见 [DL_VALIDATION_LOG.md](DL_VALIDATION_LOG.md)、[DL_PILOT_R3_ANALYSIS.md](DL_PILOT_R3_ANALYSIS.md)、[DL_RESET_JOINT_RESULTS.md](DL_RESET_JOINT_RESULTS.md)、[DL_R4_IDENTIFIABILITY.md](DL_R4_IDENTIFIABILITY.md)、[DL_R4_UTILITY_RESULTS.md](DL_R4_UTILITY_RESULTS.md)。旧已开24点不能重新命名为blind；N1M晚期对象也不能归因早期r3慢学或普通replay总体。

## R4裁决能允许什么

| 固定结果 | 合法解释和下一动作 |
| --- | --- |
| 源/身份/physics/退出失败 | 工程未验收，无科学符号结论；保存失败，只修已证工程错误，新SHA/new attempt，不拼接 |
| null超.0025或覆盖不足 | 测量/覆盖不确定，联报全部连续值；不增EP/root/H或换顺序求过门，不把未知当无用 |
| 资格、覆盖及主增量CI下界>0 | 仅支持单N1M/top2/H100、8cal/24held固定线性程序中e的探索性预测增量；可据此另立保护机制对象，不自动采用原乘性gate |
| 主CI跨0，或上界≤0 | 分别为增量不确定、当前固定程序未支持；不能挑anchor/U_act/prefix改主量或声称普遍无信息 |
| 任一当前效用为负 | 只描述actor与该路径参考的后果，不推出latent信息无价值或压缩有益 |

三臂的U_memory=U_obs−U_act是特定参考路径记忆效应，不识别action×memory交互。t只作为线性协变量，不能声称所有时间/阶段/轨迹混杂已排除。

R4独审的“独立”只覆盖：完整来源/hash/退出链、stored-applied官方CPU physics、F64/fsum与compact核对、memory/action/prevraw及静态MJB。它复用已测试统计公式和同一官方物理后端；不独立执行模型/head/D/K/NJ子key/render/训练，也不执行真实FromDM wrapper。单pair0 null不是全部MC渲染误差界。最终报告须保留这些限制。

## 正常压缩/保护的最小因果问题

以下是跨Agent独立源码审查后的**需求建议，尚未采用**，不得据此直接启动新训练。

1. **先确定要证明的三件不同事情**：当前actor如何用观测，由R4测；latent是否包含给定学习预算下可访问的任务信息，由更新后重拟合同容量/同训练预算的读出测；实际保护是否改善学习，由同对象真实更新与外部行为测。固定旧head在新坐标上失效，不能单独叫信息被删除；重拟合成功也不是信息论总量证明。
2. **最小配对B/N**：同完整初态含optimizer/moments/norm/counters、同普通replay数据、carry/keys/预算，B原loss对N实际选择性rep权重；先证明非reset且free-nats活跃位置确有作用，再测真实总gradient、optimizer位移、外部标签及行为。干预必须沿正式训练输入链，不用reset富集替代普通replay。
3. **按强主张追加有效对照**：S匹配独立标定集合上的实际活跃超额释放，回答是否只是损失数值总量；P在有效/活跃/D/K层内置换完整v且核真实移动，回答对应位置；W匹配强度reward-only，回答Dt是否必要。它们不能相互替代。第一局部B/N试验不必预先全开所有组，但宣称完整选择机制前必须有对应有效证据。匹配失败、singleton、权重全等或释放漂移失控都是对照无力。
4. **强度并非一个数**：相同标量loss释放不等于gradient范数/方向或Adam更新相同；若只匹配释放，结论仅排除该数值总量解释，不能泛称所有强度解释已排除。真实AGC/RMS/momentum使更新非线性；需记录相关模块的实际gradient与位移，不从loss权重反推出性能。
5. **不默认清旧momentum/冻结prior或actor**：同完整旧state已控制初始差异，旧moment不是两臂不同起点的混杂；它可介导后续差异，rep-stop后仍可位移。清moment或冻模块改变研究对象，只在进一步定位特定路径时另立干预，不是默认“修正”。同一旧state的函数级optimizer反事实可以诊断，但不能线性减掉动量位移当通用分解。
6. **分报绝对q/p**：若p任务效用提高而q不降，q−p/K下降可能是prior追上。若旧head变差而重新拟合恢复，支持坐标/利用问题。必须结合更新前后绝对q/p可访问任务效用及真实actor，不能单看KL或差值。

源码依据：`dreamerv3/rssm.py`的dyn=KL(sg(q)||p)、rep=KL(q||sg(p))与free-nats；`dreamerv3/agent.py`仍保留rec/con/reward/replay-value并只替换rep数值；`embodied/jax/opt.py`及Agent._make_opt的实际optimizer链。因此rep缩放不等于冻结posterior，全rep-stop也不是通用保护上界；它不应成为N必须战胜的虚构上界。

**普通replay可用性前置，2026-10-15 16:02:35北京时间，只读0GPU**：sv2原r3根
`/data/Policy_Discrepancy/runs/dl-pilot-r3-20261013-13ec72a/dmc_quadruped_walk/N-seed0-attempt01/replay`
仍有1280个NPZ/11,421,580,553字节（约10.64GiB），文件名形成16条无环无分叉链、各80文件、声明总帧1,001,008；
checkpoint/config/manifest/完成标记存在。只检查文件名与元数据，没有算chunk SHA、CRC或解码schema/content，
不能写成普通buffer已经验收。末端各有一个未落盘successor，与源码flush后新建空chunk相容，不能仅据此报损坏。
现有保存链没有当时sampler RNG/online queue/环境状态，故下一候选只能明确定义新的持久化buffer离线抽样对象，
不能无缝恢复原online流。实际采用前须版本化严格capture全部chunk、验schema/stepid/context与抽样后官方Consec前缀；
不能用reset富集、零carry或静默跳坏chunk替代。此处仅确认文件仍在，新B/N干预尚未采用或启动；
当前先完成R4 cal-null停止的[独立CPU核验](DL_R4_CALIBRATION_AUDIT.md)。

## 模式代理与实际抽样策略的额外辨识边界

**源码事实**：现有D比较q/p categorical mode所条件化的Normal，按scalar action dimension平均对称KL；不是对latent抽样积分后的边缘动作分布KL。将后者的数据处理界直接用于D是不成立的。raw K又可包含当前actor未读取的latent差异，因此“高D且高K”本身也不能保证当前任务依赖。

**已执行的有限接口构造，不是真实模型/任务证据**：本机CPU沿实际`outs.Agg(OneHot(logits,.01),1)`和`rep_probe.discrepancy`，BF16 logits、32×32 latent/12动作；只令首slot前两logits为(δ,−δ)，p交换，其余30类−20。构造共同actor只读取首slot0/1，Normal均值分别±.5并广播到全部12动作维、其它类0、std=.2（在原输出族允许范围）。它是人工输出接口，不是已训练Dreamer网络或RSSM未来。

在实数数学中，令Z=exp(δ)+exp(−δ)+30exp(−20)，a=.99exp(δ)/Z+.01/32、b=.99exp(−δ)/Z+.01/32。q/p首slot两概率a/b交换，TV=a−b，K_first=(a−b)log(a/b)。共同当前动作channel给出 KL(π_q^mix||π_p^mix)≤K_first；对同一仅依赖当前动作且值域[0,1]的奖励函数/条件kernel，两侧**期望奖励差**绝对值≤TV，不是单次实现差的界。δ→0+且严格非平局时这两界趋零，而mode条件D维持(.5−(−.5))²/(2×.2²)=12.5；δ=0则mode相同。D按动作维度平均，所给mixed界是joint KL；若同按12维平均，界还应除12。该不连续来自mode选择，不证明真实任务中它有多常见，也不判定mode代理作为启发式的设计非法。

CPU实际API从δ=1降到BF16 δ=.0009994507，D均约12.499999；最后first-slot K的F32 API约1.976616e−6、实数Decimal参照1.976577e−6，TV参照.0009894558。Decimal不是实际F32分布的严格数值界，极小δ在F32最终可舍入平局；上面的极限是实数论证，四个已执行δ均保持不同mode。再让另外31个actor未读slot的q/p logits为(5,−5)/(−5,5)，实际总raw K≈243.24994>free-nats1，而相同当前动作channel仍只依赖首slot。故K的“压缩活跃”与当前actor策略差异不是同一对象。

这只增加一种理论反例：mode边界及未用latent更新可以混淆D/K解释。**不能**把当前动作界外推R4的U_act/U_obs或100步回报、未来RSSM记忆、信息可学性或保护收益，也不能据此改正在运行的D定义。R4已保存mode/sample字段，任何真实偏差分析都要等完整认证后明确作为辅助分析，不换主比较。

小证据在 `analysis/outputs/dl-r4-utility/readiness-20261015/`：mode-boundary.json（源码SHA、4个δ、F32/70位Decimal分报，SHA256 `0d44349f544ae916f6b576dbc6a5c62b90ac77dd7fb1289a951ad380d6e450d6`）与mode-boundary-activeK.json（SHA256 `bbf807f7572816a858b96aa5c11200afe88b37161911479d16fa4fb9e696b6a5`）；两个外入口约1.49/1.46秒、只CPU/未加载Agent、checkpoint或环境/未读取r4数据。独立数学/源码审查确认期望差、全部12维、实数/F32区别及禁止U_act外推的边界；它没有再执行模型/构造。该有限构造不是新增探针或真实任务验收，完整目标保持active。
