# DL 单次观测省略与正常反馈：候选协议

版本 `DL-feedback-r1`，2026-10-11。**状态：授权内 Agent 提出的候选，未冻结、未实现或执行；不是用户逐项确认的新方法。** 只论证正确生产动作接口上的有限控制机会，不修改 DtLatch/gate、D/e/K、alpha/rho、free-nats、训练损失或正式 M2 协议，不授权训练。旧 `DL-predictive-MC-r1` 仍按原门停止、裁决 `MC-inconclusive`；不能借新对象或新来源改写旧产物。

根 Agent 实际核查，新的 `DL-source-r1` attempt01 在 `bb78e8c625c141d862293d26fc32d5ca64a598cb` 因host BF16隐式传输检查失败且没有有效产物；只修显式device_put的 attempt02 已在确定提交 `f2d24da1831da2ffca22d313e7f534c96fc0d3eb` 于2026-10-11 13:00:42–13:08:22完成。source完整收据与独立artifact/八calibration CPU physics审计passed，见 [DL_PRODUCTION_SOURCE_RESULTS.md](DL_PRODUCTION_SOURCE_RESULTS.md)。本候选设计及交叉审查未读其新 source/blind 数组、不根据新收益修订。来源通过不能替代本候选实际policy/key/F64 reward语义验收，也不自动冻结或启动反馈实验。

## 1. 研究对象及目标边界

研究问题：**在生产 baseline 的正常状态与实际历史 carry 上，省略当前一次 RGB，之后双方都正常接收各自环境的观测，会损失多少有限窗口的真实控制收益？** 初始 clean posterior 与真实 prior 使用同 h，其后各自闭环；只看当前观测相对共同历史的局部可利用增量，不保持 prior 分支永久失明。

这是有理由的下一对象：旧共同记录续动作 H100 不会根据分支状态反馈纠错，也不是完整 policy；旧 reward mode/MC 评分不是控制价值；动作契约问题使旧 source 不能称生产自然状态。先在新来源检查正常未来观测补偿，能保留原任务 scope，避免先用外力制造新场景。合法有界物理挑战仍是替代，但须另立可见性、OOD、actor 能力与适用范围，不能以其阳性覆盖自然状态限制。

**工作假设**：当前观测提供历史之外、现有 actor 能利用的控制信息，且其影响在正常反馈下至少部分保留。可证伪反例包括后续 RGB 迅速补偿、prior 已足够、actor 未利用新增信息、raw policy 差异在物理 clip/mapping 后失去作用，或随机路径/模型表征不匹配主导收益。非正结果只限制这个对象；一次 omission 不等价于训练中的连续 KL 压缩。

用户的目标仍是用 Dt＋另一信号辨别需要保护的目标状态，并通过保护 posterior 提升模型。平均观测价值是机会检查，**不是目标子集辨别、第二信号增量或性能验收的替代**。

## 2. 正确来源是强制前置条件

只接受 `DL-source-r1` 的 `production_clip_unify_normalize_v1` 来源；先完成 source complete 和独立审计，再冻结本候选。来源修复及旧证据范围见 [DL_ACTION_CONTRACT.md](DL_ACTION_CONTRACT.md)。禁止接受 legacy raw-suite、旧20261012重建 history 或只有未来映射正确的旧物理 snapshot。

输入须绑定同一份 source plan/complete、32 artifact 元数据、独立审计、实际 Git/config/checkpoint/参数 hash、依赖及动作 bounds/dtypes。只打开 source plan 指定的八个 calibration NPZ，路径须位于来源目录内并先验字节 hash；完整 plan/complete 可核 blind 元数据，但不打开24个 blind 数组、carries 或 snapshots用于设计。可信项目 carry/snapshot pickle 只在 artifact 验证后读取。

固定 checkpoint 为原 baseline seed0 的 300000训练动作阶段：文件 SHA256 `2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9`，参数 digest `14859c4e129b51aa488c10eab09f16d56fd110c76239b2c9072f415829d111c2`，config SHA256 `11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445`。updates74493；checkpoint actions18769是policy calls，不是环境训练动作预算。

必要 source 证据是 actual DMC＋`main.wrap_env`、raw/applied/action-executed 独立记录、训练参数/counters冻结、suite＋FromDM `_done` snapshot、原16位置 actual pre-policy完整 carry/dtype/逐叶hash与实际eval key/counter收据，以及 source recorded-applied H100逐 transition reward/fullstate/task/counter/flags和生产policy/carry重演。source collection 的evalcounter合法推进与本诊断私有RNG隔离必须分开表述。任何来源不完整、不符或独立审计缺失，都只记前置条件未满足，不发布反馈收益裁决。

## 3. 候选数值与选择规则

以下数值仍待根 Agent 完成交叉审查和工程成本验收后冻结，不能当作已有启动配置。

| 项目 | 候选 |
| --- | --- |
| 来源 | 300k `DL-source-r1`，collection seed20261020，32完整episode/8 calibration＋24 blind；只用八个calibration |
| 状态 | 每条原16 outcome-independent H100位置取索引 `[0,5,10,15]`，共32状态；核对source存储位置与原sampling函数逐点相同 |
| history | source采集时actual pre-policy完整carry、previous raw action和真实snapshot；不另采历史、不更换来源carry |
| 策略重复 | 每状态4个事前命名的paired latent/actor RNG；private seed20261018 |
| 主窗 | 100次环境动作，唯一主对象是有符号累计真实reward；H1/10/50 prefix仅描述 |
| 统计 | episode等权，2000次cluster bootstrap，seed20261019；within-state RNG散布和完整signed分布联报 |
| 分支 | clean q吸收当前RGB，p只在初次用同h真实prior；之后各自正常posterior、sampled actor和生产物理映射 |
| 资源 | sv3物理GPU4单卡，3600秒父进程硬上限，实际启动前fresh占用与CUDA/EGL同卡核查 |

不按 source return、D/K/C/e、旧机会标签或本次 utility 选择状态、RNG、prefix、checkpoint。缺位置/不完整 episode 不替换、不丢弃后只分析剩余好点；只保留不完整证据并停止。24 blind不会成为本轮补样本。若数值不充分，不追加 H、n、RNG或1m阶段寻求通过。

## 4. 一次省略的模型与物理定义

每状态恢复同一完整 suite/wrapper snapshot，保留同一个 source实际t前carry。当前 clean RGB 由恢复后的实际环境以camera2/64×64**新render一次**，作为q当前输入；p忽略它。保存其与source当前RGB的hash、最大像素差和固定calibration视觉对照；不反复render选最接近或最有利图像。源码/配置/视角/geometry错误是semantic failure；像素差本身不自动宣称逐位重演，也不事后建立有利容差。

在原source dyn carry与previous raw action上按实际production cast/reset/core计算一次共同 h。q由当前encoder token得到posterior logits；p取 `_prior(h)`，不使用零/灰图像代替prior。用实际`_dist`采样初始 z_q/z_p，unimix恰好一次，双方共享该次categorical噪声；actor通过同h/各自z产生raw Gaussian样本并共享该次actor噪声。q current latent、posterior logits/tokens不得传入p的未来carry。

冻结架构的encoder/decoder carry为空，但仍保存完整carry；若实际状态/架构不符，停止而不自行丢字段。初步之后，每分支使用自己的dyn carry、previous raw action、真实RGB及正确flags正常posterior更新；不使用另一分支的图像、记录续行动或prior mode。`mode='eval'`仍sample，不冒称mean策略。posterior之外的生产cast/core/policy逻辑不重写成近似实现。

每步raw float32动作保存在model carry，实际环境沿`ClipAction→CheckSpaces→UnifyDtypes→NormalizeAction→DMC`执行；实际suite收到的float64 applied另存并与helper逐dtype结果核对。不能对applied二次normalize、以MuJoCo clamp代替映射或把applied当作RSSM previous action。正确环境状态与carry随各自分支推进，不保留多步陈旧carry。

## 5. 随机流、终止与奖励

每次policy调用的private root key由version/seed/episode/position/pair/绝对step事前命名，不含branch；latent/actor子key须按生产policy实际Ninjax派生顺序取得并加tag记录，不能独立发明tag key后只核边际分布。q/p共享同tag噪声，不同pair/step独立。p省略encoder观测时仍须保留与q相同的随机消费位置，不能少消费一次`nj.seed()`而使actor噪声错位；用真实production policy oracle核对这个派生契约。实际distribution的shape、unimix、Normal dtype和采样接口与production一致。某分支结束、插入日志或执行额外oracle检查不改变另一分支的key；不消费source/training/eval RNG或改变原counter。共同随机数保证所实现的正确边际，不保证任意非线性控制方差降低。

source预定位置保证原clock有至少100动作；100是新分支的有限时间窗，不从source未来轨迹读取控制或图像。真terminal必须按以下候选规则处理，而不删掉失败分支：

- `last && discount==0`：包括当步真实reward，然后停止该分支物理/action/policy调用；剩余H100使用显式absorbing zero reward。保留terminal step、action/observation/alive mask；padding不是实际执行、观测或logits，不跨reset。
- `last && discount==1`：须核clock/step_limit及task语义。早于第100动作是本对象semantic failure，整项停止，不把该分支删点；恰在H100末端的预期time-limit允许，计入最后reward。
- 异常discount、意外reset、无合法窗口或非有限物理/模型值：semantic failure。未到H100的预算/异常中断不是absorbing terminal，不能填零伪装完整结果。

**奖励dtype候选，须一并冻结**：主reward取suite实际返回的float64逐步值，生产wrapper的float32编码同时保存并核对；不是reward head预测。H100sum以float64计算。不能看到符号后选择F32/F64、sum/mean、正部或discount版本。absorbing zero仅用于已明确的真terminal，所有实际reward/action/state都有可核对mask。

## 6. 统计及停止解释

对episode j、其四个位置i、四个paired RNG r：

```text
U[j,i,r] = sum_s actual_reward_q[j,i,r,s]
           - sum_s actual_reward_p[j,i,r,s]       # H100，不含t处共同旧reward
U_episode[j] = mean_(i,r) U[j,i,r]
primary = mean_j U_episode[j]                    # 八episode等权
```

cluster bootstrap整条episode重采样，每draw八条、有放回；位置/RNG随整条episode一起保留，不能把128 paired重复当作128独立状态或训练seed。候选使用`numpy.random.default_rng(20261019)`生成2000draw，percentile两侧95%区间取2.5%/97.5%分位并固定`method='linear'`，不据结果选择区间算法。区间是固定calibration及四RNG ensemble上的探索估计，不是完整policy随机路径积分误差界、正式方法检验或跨训练seed复现。每episode、state/pair的有符号值、within-state std/range、每RNG总体mean、first-two/last-two及leave-one-RNG摘要同时发布；这些只描述有限随机误差，不挑RNG改变主结果或追采样。

候选主门和输出分类为：

| 完整语义验收后的证据 | 有限对象裁决 |
| --- | --- |
| primary lower95 > 0 | `exploratory_average_opportunity_supported`，仅允许论证新独立候选；不生成v或训练权限 |
| upper95 <= 0 | `exploratory_average_opportunity_not_supported`，停止这个自然来源/窗口/actor对象 |
| lower95 <= 0 < upper95 | `exploratory_average_opportunity_inconclusive`，停止有限对象，保留分布与有限RNG不确定性 |
| 来源/语义/复原不通过或预算中断 | `semantic_failure` / `incomplete`；不作机会判定 |

lower95>0只是**平均机会的探索充分门，不是选择性保护的必要门**。少数关键状态收益可以被其他状态损失抵消；平均正值也可能是prior/actor表征分布不匹配、路径抽样或非关键线索作用。四RNG不能提供精确逐状态positive/negative真值；不从它们构造新gate标签或报正式AUROC。非正/不确定停止此有限对象，不否定稀有目标信息、持续压缩学习收益或全部后验保护路线。

八个episode的bootstrap区间只有名义95%口径，不能保证有限样本覆盖率或罕见状态覆盖；四位置也不是该episode所有时刻的均匀积分。此检查仍只有一个baseline训练seed和一个checkpoint，不能据区间窄宣称跨模型复现。若分散大或少数episode主导，完整分布需显示这个限制，不靠删除episode或更换bootstrap求阳性。

## 7. 必要工程验收与硬预算

语义正确先于utility。CPU actual tiny模型先验：q=p同噪声退化、初始h共用、p初次不吃current token/latent、以后双方正常posterior、未来图像/动作不影响过去prefix、不同branch停止不移key、raw/applied不混淆、真terminal吸收与time-limit/reset拒绝、readonly模型/RNG/counter隔离。真实MuJoCo/production wrapper另验FromDM `_done`及完整恢复，不能以mock物理证明真实标签。

若为注入私有噪声拆开正常policy，须用未改生产B1 `model.policy` 作oracle，核actualcast/core/distribution/key/actor/carry。真实size50m在32个预定calibration源RGB/carry/source-eval-key上应核q正常路径与source保存raw/after-carry一致；诊断不得因BF16/JIT图差异自行放宽成“生产精确”。核失败先保留工程原因并停止，不能以大批utility或容差掩盖。诊断纯调用最终参数、optimizer/norm与原counters不变；oracle临时eval计数须独立副本或finally精确恢复。

每个已执行分支同时做两类独立复放：固定缓存RGB＋相同keys/carry的policy离线重演，raw动作/后续carry逐叶字节核对；固定stored-applied动作从同完整snapshot再执行physics，逐步integration/task/RNG/counter/terminal/wrapper状态按原1e−8要求核对，F32 reward编码精确。**F64主reward候选须逐步byte-exact复放**，不能以F32相等或state误差在1e−8内冒称F64标签重演；不精确则保存差异并停止语义验收，交审具体原因，不看到utility符号再放宽。积分state shape/dtype及finite先验，NaN不能通过max-error比较。必要时重render并保存pixel/hash差，只称实测反馈路径加条件复放；没有重新闭环逐位核对时不声称renderer或完整闭环bit-exact。

候选新增上界为32状态×4 RNG×2分支×100=25,600环境动作，完整physics动作复放另最多25,600；固定RGB的policy复放另最多25,600前向。source已验且独立审计绑定的H100参考可以复用，不无理由重复；新增必要source reference若不能复用，必须在冻结前计入真实工作量。模型加载/编译、全部oracle、render、复放、分析与写盘均计入**单卡3600秒父进程硬监督**，不是frame-loop cooperative check或预计ETA。超时只终止自己的worker、保留新attempt的不完整证据，不换卡、加预算、追加RNG或只发表已完成好点。

实际GPU4在source释放后须重新fresh核查，CUDA/EGL都绑定物理4；不得与source或未知作业共享。实际SHA、config/source/audit hash、环境、私有seeds、资源、退出码、实测时间/显存和每branch执行步数均落运行记录。上述资源/预算仍是反馈候选，root冻结与工程验收前不执行。

## 8. 产物及完成边界

新ignored运行目录候选 `/data/Policy_Discrepancy/runs/dl-feedback-<actualdate>-<sha>/cp300000-attemptNN/`。plan绑定source complete/audit及八个artifact字节hash、位置/实际carryhash、全部keys、dtype/contract/预算；逐episode保存fresh/source初始RGB对照、每branch真实RGB、h/latent/logits/raw/applied、float64/F32 reward、flags/masks、完整状态/复放摘要。完整数组及可信状态pickle不提交Git；本机仅镜像小plan/result/独立审计到ignored `analysis/outputs/dl-feedback/`。

独立CPU分析应不import生产分析器：从逐步reward/mask复算所有signed prefix、episode等权统计/bootstrap、终止/完整性与分类；核新来源、位置/keys与参数收据。它不能独立重演模型时须明说，产物一致性不是方法有效。

所有输出保持 `training_authorized=false`、`proceed_short_training=false`、`eligible_method_training=false`、`method_effectiveness_established=false`。即使平均机会阳性，仍需真正未见数据、独立目标子集标签、第二信号相对D的增量、背景/OOD/actor能力等对照、真实rep压缩和条件训练。真物理状态只作诊断证据，不作为vision64/propriofalse部署gate输入。

当前交付仅为候选定义及反例审查。首选下一步是等待正确source complete与独立audit，完成本候选接口/CPU/实际B1语义审查后由根Agent明确冻结；不自动实现、开跑或选择新gate。形成可正式实验方法仍未完成，也不保证本对象找到有效方法。
