# DL-r5 原 live 环境随机干预

版本 **DL-live-utility-r1**，2026-10-15。Agent在完整预验证授权内采用的新有限诊断，工程实施中，
**尚未执行真实模型/GPU对象**。保留native H100和原H1问题，不把固定控制库标签冒名替换为闭环效用。
原R4 cal-null失败和`.0025`停止记录不变；正式M2、原算法/训练gate不变。

## 问题与估计对象

原R4要求同物理state的两次闭环路径近似配对，真实RGB微差会在后续产生不同动作。
这并不证明native平均效应不可估。本对象在**一个原live环境/原EGL context中随机执行一个arm**，
此后连续native posterior H100，不恢复或复制live渲染状态。

问题：在冻结N1M模型正常访问的普通时刻，将当前q动作/q记忆换为p，平均真实H100回报有何变化；
在预定高D群体内，原H1奖励支持e相对D+K是否改善该平均效应的固定预测程序？
这不是每个state的双反事实真值，也不识别信息论最优策略、压缩损伤或保护收益。

Y为实际执行arm之后100个真实suite F64奖励之和/100；真terminal后按0吸收，分母固定。
qq用q动作/q当前latent记忆，pq用p动作/q记忆，pp用p动作/p记忆；
首步之后都用自己的真实后续RGB、正常q posterior及自己实际raw作为prev_action。
U_obs=E[Y_qq−Y_pp|X]为主，U_act=E[Y_qq−Y_pq|X]和特定参考memory差为辅；三臂不识别完整交互。

## 固定对象、随机化和时序

- 同已审N1M完整checkpoint/config（c6a7d92e…/877f8470…，来源见R4记录）、size50m、seed0、
  quadruped_walk、vision64/camera2/proprioFalse/repeat1/BF16；模型/optimizer/norm均不更新。
- 32个environment families，每family16个独立policy流，共512 unit；不是512训练seed。
  f=0..31、k=0..15、全局u=16f+k，env_seed=8192+f，eval/policy_seed=0x52350000+u。
  前8family/128unit仅cal，后24family/384unit为本新对象held。
- 每unit开始fresh官方suite/env和EGL context，同family也重新构造并reset，不能复用suite连续reset推进task RNG。
  每family独立fresh worker exec进程，先spawn纯CPU shadow，再加载同一N1M/编译；32family串行，
  family之间不延续Python/GL/CUDA用户态状态。这不证明底层驱动/硬件全独立。
- t事前均匀取原14普通位置64..896/步长64：
  `j=default_rng([20261030,0x54494D45,u]).integers(0,14); t=64+64*j`。
  t之前为原正常q行为。绝对policy counter从0到t+99连续，不在干预处换R4 branch key或归零。
- 当前q/p bundle、D/K、H1信号及处理前r_t全部封口后，才draw/reveal assignment：
  `a=default_rng([20261030,0x41524D53,u]).integers(0,4)`对应`qqA,qqB,pq,pp`，各概率1/4。
  两个qq标签执行完全相同first(qq)接口；标签不进入policy seed、NN输入、prefix或sham后的控制路径。
  source只能用单独time函数；统计模块的联合plan认证函数不能在封口前被producer调用以偷看label。
- 工程层证明实际写盘顺序、SHA绑定和model/input readonly；仅JSON中的逻辑event或bool不足以证明随机化。
  各unit紧凑统计字段的unit_id为局部k；helper/文件名中的全局unit为u，二者不能混用。
- cal最后unit完整后拟合并封口，再启动任何held worker/source。prefix/干预/后续关键位置真实public oracle，
  包括prefix0、当前q、最后正常q；临时eval counters/seed必须精确恢复，policy copy认证当前params。

## H1必须是处理前量

不能拿实际pq/pp执行后的nextreward计算e，再把它当处理前协变量预测该arm结果。
在t处，独立official CPU shadow在同完整physics/task snapshot和完整MJB SHA/长度/camera身份下，
执行已经生成的baseline-q would-be raw，取得共同reference-behavior的真实F64 nextreward。
shadow不render、不访问live对象，不加载模型/GPU；live before/after snapshot与静态模型hash必须一致。
request/response原bytes和SHA保存；qqA/B实际首reward必须与shadow F64逐字节一致。

原H1 mode kernel据此评分，latent/actor原subkeys共享，不额外消费nj.seed。e仍为原定义[0,1]信号，
不是所分配arm的奖励。该shadow是**离线诊断reference行为取证**，不是为部署增加环境oracle；
真实训练replay中原H1使用已记录行为动作/nextreward，不能把本诊断未经说明移植为在线gate。
处理前r_t另取live suite实际F64并核生产F32编码，不能误用shadow r_next作为baseline。

## 固定统计程序

令b(X)=已观测r_t，z=Y−r_t，qq合并A/B：

```
phi_obs = 2 I(qq) z − 4 I(pp) z
phi_act = 2 I(qq) z − 4 I(pq) z
phi_memory = 4 I(pq) z − 4 I(pp) z
phi_sham = 4 I(qqA) z − 4 I(qqB) z
```

因2P(qq)=4P(pp)=4P(pq)=1，处理前baseline项在条件期望抵消。
phi是有噪声的随机化伪结果，不能按其正负给state贴“有用/有害”真值标签。
r_t只可能减方差，不保证：两arm未来Y均0而r_t=1时原方差0、中心化方差6。
本对象固定该baseline，不事后择常数/不中心化/另拟合nuisance求阳性。

cal全部128unit等权（各family1/8、unit1/16），固定ridge .1、intercept不罚、标准化仅cal。
预测器为D+t+r、DK+t+r、DKe+t+r、e+t+r，t=position/1000、r=r_t；无lambda搜索。
cal D75=`quantile(D,.75,method=linear)`；held highD为D>=D75，保留ties并报告实际比例。
fit全部cal、主评highD是固定程序的cal-to-tail泛化检验，不是最优预测器或e总体信息性的检验。

主比较：held highD的phi_obs MSE(DK)−MSE(DKe)；同一phi使噪声平方项在风险差中抵消。
单位为平方平均reward，δ=.01只是描述性reward尺度，**不是主gain阈值**。
U_act、all-held、其它模型比较和memory均辅助，不替换主量。

固定辅助排除项：仅线性ridge的增量也可能来自重复列改变L2惩罚（如e=D），
或e=D²补入原模型缺失的形状，并不自动代表新信息。
增加cal-only固定基底B=[D,K,D²,D*K,K²,time,r_t]，cal128等权标准化，
含截距无罚OLS/SVD（numpy.linalg.lstsq，rcond=None）将原尺度e投影到B；
e_res为冻结投影的残差。cal最大绝对残差<=1e-10则禁用该列、extended直接复用base模型和预测，
不得读held e后重新激活。否则比较同ridge .1的B与B+e_res，沿用同highD/phi_obs/family draws风险差。
投影参数、rank/singular values、是否禁用全部在cal seal中冻结，不用Y选投影或新增搜索。
输出aux_nonredundant_basis_control，主比较不替换；未获辅助支持时重复列/有限形状解释仍未排除。
即使获支持，也只是在该固定二次基底外的预测增量，不证明对任意D/K函数的条件信息独立。
1e-10是数值冗余守卫，不是科研等效阈值。shadow e还可能共享H100中的首步1/100奖励，
因此合法处理前预测也不单独证明长期信息利用或保护收益。

全24 held family cluster bootstrap10000次/seed20261030，保留每family16unit及0-highD family；
每次用highD numerator/count的ratio-of-sums，不改成每family条件均值等权。
若有空分母则区间unresolved，不丢弃/补抽。固定cal模型/阈值，不bootstrap重拟合。
覆盖固定highD各label>=8、至少12个held family有highD，另报每label family数；这是有限支持，不是功效保证。
外部完整真实执行/来源审计、覆盖和主gain区间下界>0才可标`primary_support_pattern`。
区间跨0/全非正分别按不确定/该固定程序未支持；不追加unit/换t/调baseline或门求支持。

## Sham解释与识别边界

新RCT不需要两个独立qq结果近乎相等才能识别操作性平均效应；render/policy变动计入Y与family区间。
全局sham均值精度既非主risk识别的必要条件，也不足保证所有X条件偏差。
因此同highD sham 95%区间全入±.0025只报告预定precision flag，**不作主support硬门**；
不能把区间含0当通过，也不能据此改原R4单pair停止。

独立负控：held highD固定qq membership与z，重新iid公平抽qqA/B符号，
对|sum(sign*z)|作双侧随机化检验，10000/seed20261031、p=(1+extreme)/10001、alpha=.05。
拒绝标`implementation_concern`，不是bug/renderer根因证明；不拒绝也不证明接口一致。
主support pattern与concern分开联报；二者同时出现时禁止无保留声称canonical qq效用。

名义24-family区间条件于单cal fit/checkpoint，依赖适当family可交换性与一致性；
不是精确随机化CI、训练seed复现或全部隐藏GPU/GL干扰的证明。
若标签进入前处理、shadow污染live、实际操作不一致或跨unit干扰破坏识别，应拒绝工程/因果验收，
不能用声明字段、随机顺序或sham统计代替实际来源/代码/过程审计。
无论结果如何，本对象不单独证明正常压缩损伤、信息可读出、选择性保护或排除S/P/W替代解释。

## 资源、工程前置与后续

只计划sv3物理GPU4独占一A100，5–7不预占；启动前fresh XML+pmon及真实CUDA/EGL同UUID，
冲突等待在用户分配范围内明确安排，不能抢占/共享或杀他人作业。
完整父入口固定7200秒，含32次load/JIT、512 prefix/干预/H100、shadow/oracles、cal封口、统计/写盘/清理。
该单次上限为2张卡小时（1卡×2小时），不是预计必定完成或全部预验证总预算；
代码/统计/数据完整性和独立物理审计使用CPU，不为其占用GPU。
最大真实native交互512×996=509952步，另512 shadow物理步；不隐去32family初始化和实际I/O，
既有吞吐只是规划锚，可能超时，本对象不延时补跑/拼接。
实际完整来源/physics CPU独审另限1800秒/0GPU，须核全部单位/奖励/keys/过程与cal seal，
独立NN/render不在纯CPU审计范围，不能冒称所有后端独立执行。

需先完成新namespace/原kernel组合、qqA/B同操作、真实tiny public oracle、影子不污染live/真F64、
分配前封口/终止/父owned进程清理及统计反例的有限测试，再确定提交Linux验证和实际N1M入口。
代码/证据用DL命名，新目录保存失败，不改R4/正式M2的算法、门或原产物。
输出`/data/Policy_Discrepancy/runs/dl-r5-live-<date>-<sha>/attempt01/`；未完整退出/独审时不得升级为科研支持。
下一仍须正常active replay的压缩/保护干预与有效替代对照；普通历史buffer的内容取证见
[DL_REPLAY_INVENTORY.md](DL_REPLAY_INVENTORY.md)，不等于在线终点cache恢复。

## 工程验收记录（未实际N1M执行）

新增policy/shadow/family/stdlib父入口/CPU统计/独审及相应测试；不改原R4、Dreamer算法或M2。
本机Windows整组47项（35新有限契约＋12原R4统计回归）46pass/1 Linux进程组skip，15.896秒。
实际tiny BF16 policy/三点public oracle、真实官方CPU physics与stdio shadow已经有限测试；
独审单unit使用N1M形状BF16，但NN/H1/public oracle及RGB明确fake，不是真实N1M科学验收。
包含实际F64单ULP而F32不变、动作/carry/key/原r_t、封口先后/篡改、prefix终止/实际outcome早停、
128cal先封口再held、进程/资源拒绝和独立物理重演等反例。
Windows时钟实测GetTickCount64分辨率15.625ms，独审fixture仅Windows producer调用注入真实perf_counter；
Linux生产与审计的严格monotonic先后不放宽，source pins采用真实Git cat-file blobs而非archive转换字节。
完整512生产图、实际N1M/GL、Linux进程组和真实设备仍待确定提交后的验收；没有由工程测试产生科研结论。
