# DL 单次观测省略与正常反馈：冻结协议

版本 `DL-feedback-r1`，2026-10-11。**当前状态：科学协议保持冻结；工程r1在确定提交 `bd0af3a47ac0861d932a4a08af2dd1ce84754dbe` 的27项server CPU检查与32个实际size50m源策略oracle通过，首个GPU尝试在反馈开始前因单帧数组容器接口失败，无utility产物。工程r2只修复该容器并增加反例测试，须确定提交server CPU复验后以新attempt继续。** 这是用户完整前验证授权内的Agent设计，不是用户逐项确认的新方法。只论证正确生产动作接口上的有限控制机会，不修改 DtLatch/gate、D/e/K、alpha/rho、free-nats、训练损失或正式 M2 协议，不授权训练。旧 `DL-predictive-MC-r1` 仍按原门停止、裁决 `MC-inconclusive`；不能借新对象或新来源改写旧产物。

根 Agent 实际核查，新的 `DL-source-r1` attempt01 在 `bb78e8c625c141d862293d26fc32d5ca64a598cb` 因host BF16隐式传输检查失败且没有有效产物；只修显式device_put的 attempt02 已在确定提交 `f2d24da1831da2ffca22d313e7f534c96fc0d3eb` 于2026-10-11 13:00:42–13:08:22完成。source完整收据与独立artifact/八calibration CPU physics审计passed，见 [DL_PRODUCTION_SOURCE_RESULTS.md](DL_PRODUCTION_SOURCE_RESULTS.md)。冻结时根 Agent 与本科学审查未查看新 source 收益、D/K/C/e/U，也未读取新 blind 数组；数值沿用原候选及内部审查，未按结果修订。来源通过不能替代反馈实际policy/key/F64 reward语义验收；科学协议冻结不等于工程或科研验收。

## 1. 研究对象及目标边界

研究问题：**在生产 baseline 的正常状态与实际历史 carry 上，省略当前一次 RGB，之后双方都正常接收各自环境的观测，会损失多少有限窗口的真实控制收益？** 初始 clean posterior 与真实 prior 使用同 h，其后各自闭环；只看当前观测相对共同历史的局部可利用增量，不保持 prior 分支永久失明。

这是有理由的下一对象：旧共同记录续动作 H100 不会根据分支状态反馈纠错，也不是完整 policy；旧 reward mode/MC 评分不是控制价值；动作契约问题使旧 source 不能称生产自然状态。先在新来源检查正常未来观测补偿，能保留原任务 scope，避免先用外力制造新场景。合法有界物理挑战仍是替代，但须另立可见性、OOD、actor 能力与适用范围，不能以其阳性覆盖自然状态限制。

**工作假设**：当前观测提供历史之外、现有 actor 能利用的控制信息，且其影响在正常反馈下至少部分保留。可证伪反例包括后续 RGB 迅速补偿、prior 已足够、actor 未利用新增信息、raw policy 差异在物理 clip/mapping 后失去作用，或随机路径/模型表征不匹配主导收益。非正结果只限制这个对象；一次 omission 不等价于训练中的连续 KL 压缩。

用户的目标仍是用 Dt＋另一信号辨别需要保护的目标状态，并通过保护 posterior 提升模型。平均观测价值是机会检查，**不是目标子集辨别、第二信号增量或性能验收的替代**。

## 2. 正确来源是强制前置条件

只接受 `DL-source-r1` 的 `production_clip_unify_normalize_v1` 来源。source complete 和独立审计前置条件已满足，才冻结本协议；来源修复及旧证据范围见 [DL_ACTION_CONTRACT.md](DL_ACTION_CONTRACT.md)。禁止接受 legacy raw-suite、旧20261012重建 history 或只有未来映射正确的旧物理 snapshot。

输入须绑定同一份 source plan/complete、32 artifact 元数据、独立审计、实际 Git/config/checkpoint/参数 hash、依赖及动作 bounds/dtypes。只打开 source plan 指定的八个 calibration NPZ，路径须位于来源目录内并先验字节 hash；完整 plan/complete 可核 blind 元数据，但不打开24个 blind 数组、carries 或 snapshots用于设计。可信项目 carry/snapshot pickle 只在 artifact 验证后读取。

固定 checkpoint 为原 baseline seed0 的 300000训练动作阶段：文件 SHA256 `2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9`，参数 digest `14859c4e129b51aa488c10eab09f16d56fd110c76239b2c9072f415829d111c2`，config SHA256 `11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445`。updates74493；checkpoint actions18769是policy calls，不是环境训练动作预算。

必要 source 证据是 actual DMC＋`main.wrap_env`、raw/applied/action-executed 独立记录、训练参数/counters冻结、suite＋FromDM `_done` snapshot、原16位置 actual pre-policy完整 carry/dtype/逐叶hash与实际eval key/counter收据，以及 source recorded-applied H100逐 transition reward/fullstate/task/counter/flags和生产policy/carry重演。source collection 的evalcounter合法推进与本诊断私有RNG隔离必须分开表述。任何来源不完整、不符或独立审计缺失，都只记前置条件未满足，不发布反馈收益裁决。

已核对的小收据绑定：source plan SHA256 `633efe28d551d5793ac8f0ed5f0e36e937e7f1af3c8edefc1d9f5fa278146fcc`；complete `d155f577d19d2bbc051ec56b95f15e26afc774c130c3c9728a7fe599f295f3cf`；独立artifact审计 `8f93788ca212118139fab135036fd47ccbad79f04a34139b5dfa7d2dd26f7516`；独立physics审计 `8ed3528fae18dc7a3043c767da87f873642a40a1a028ce86247f0521c3264722`。独立physics只打开八calibration，128个窗口/12800步、initial/restore/integration最大误差0；24 blind只hash。它直接执行official suite recorded-applied，未独立运行实际FromDM wrapper/policy/render。source reward证据是F32编码一致，**不能用它替代下文新反馈分支F64主reward逐步byte-exact验收**。

## 3. 冻结数值与选择规则

以下是本轮冻结的科学选择。工程实现、真实policy oracle、完整复放和资源guard须先验收；不以协议冻结宣称这些检查已经完成。

| 项目 | 冻结选择 |
| --- | --- |
| 来源 | 300k `DL-source-r1`，collection seed20261020，32完整episode/8 calibration＋24 blind；只用calibration IDs `[9,14,15,18,20,24,30,31]` |
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

source预定位置保证原clock有至少100动作；100是新分支的有限时间窗，不从source未来轨迹读取控制或图像。真terminal必须按以下冻结规则处理，而不删掉失败分支：

- `last && discount==0`：包括当步真实reward，然后停止该分支物理/action/policy调用；剩余H100使用显式absorbing zero reward。保留terminal step、action/observation/alive mask；padding不是实际执行、观测或logits，不跨reset。
- `last && discount==1`：须核clock/step_limit及task语义。早于第100动作是本对象semantic failure，整项停止，不把该分支删点；恰在H100末端的预期time-limit允许，计入最后reward。
- 异常discount、意外reset、无合法窗口或非有限物理/模型值：semantic failure。未到H100的预算/异常中断不是absorbing terminal，不能填零伪装完整结果。

**冻结奖励dtype**：主reward取suite实际返回的float64逐步值，生产wrapper的float32编码同时保存并核对；不是reward head预测。H100sum以float64计算。F64保留真实suite值，避免F32累计舍入成为边界utility符号的另一选择；它与训练/旧source的F32观测reward不同，后者仍须联存、精确核对。不能看到符号后选择F32/F64、sum/mean、正部或discount版本。absorbing zero仅用于已明确的真terminal，所有实际reward/action/state都有可核对mask。

## 6. 统计及停止解释

对episode j、其四个位置i、四个paired RNG r：

```text
U[j,i,r] = sum_s actual_reward_q[j,i,r,s]
           - sum_s actual_reward_p[j,i,r,s]       # H100，不含t处共同旧reward
U_episode[j] = mean_(i,r) U[j,i,r]
primary = mean_j U_episode[j]                    # 八episode等权
```

cluster bootstrap整条episode重采样，每draw八条、有放回；位置/RNG随整条episode一起保留，不能把128 paired重复当作128独立状态或训练seed。使用`numpy.random.default_rng(20261019)`生成2000draw，percentile两侧95%区间取2.5%/97.5%分位并固定`method='linear'`，不据结果选择区间算法。区间是固定calibration及四RNG ensemble上的探索估计，不是完整policy随机路径积分误差界、正式方法检验或跨训练seed复现。每episode、state/pair的有符号值、within-state RNG std（`ddof=1`）/range、每RNG总体mean、first-two/last-two及leave-one-RNG摘要同时发布；这些只描述有限随机误差，不挑RNG改变主结果或追采样。

冻结主门和输出分类为：

| 完整语义验收后的证据 | 有限对象裁决 |
| --- | --- |
| primary lower95 > 0 | `exploratory_average_opportunity_supported`，仅允许论证新独立候选；不生成v或训练权限 |
| upper95 <= 0 | `exploratory_average_opportunity_not_supported`，停止这个自然来源/窗口/actor对象 |
| lower95 <= 0 < upper95 | `exploratory_average_opportunity_inconclusive`，停止有限对象，保留分布与有限RNG不确定性 |
| 来源/语义/复原不通过或预算中断 | `semantic_failure` / `incomplete`；不作机会判定 |

lower95>0只是**平均机会的探索充分门，不是选择性保护的必要门**。少数关键状态收益可以被其他状态损失抵消；平均正值也可能是prior/actor表征分布不匹配、路径抽样或非关键线索作用。四RNG不能提供精确逐状态positive/negative真值；不从它们构造新gate标签或报正式AUROC。非正/不确定停止此有限对象，不否定稀有目标信息、持续压缩学习收益或全部后验保护路线。

八个episode的bootstrap区间只有名义95%口径，不能保证有限样本覆盖率或罕见状态覆盖；四位置也不是该episode所有时刻的均匀积分。此检查仍只有一个baseline训练seed和一个checkpoint，不能据区间窄宣称跨模型复现。若分散大或少数episode主导，完整分布需显示这个限制，不靠删除episode或更换bootstrap求阳性。

原sampling首位置为t=1，它在每episode的四个预定位置中保留，可能只有很短历史。固定position序号均值与完整分布联报；若起始位置贡献主导，只说明这个有限自然对象包含早期观测机会，不冒称成熟历史中的选择性保护。H1/10/50的sampled闭环prefix也不是旧r1固定动作/clippedmean H1探针的复测，两者不可互替。

统计入口契约是`analyze_utility(rewards, executed, episodes)`：F64 `rewards`和Bool `executed`形状均为`(8,4,4,2,100)`，轴序episode/固定position/paired RNG/branch(q=0,p=1)/step；八个episode为唯一整数IDs。每branch executed是至少一个True的连续前缀，其后padding reward严格为0，所有值finite；不按实际存活长度重新normalize。拒绝shape/dtype、mask洞、非零padding、duplicate/noninteger IDs。mask本身不证明true terminal，模型/环境层必须另核discount/last/clock。纯统计结果包含`primary`、`interval95`、`u_pair`、`u_episode`、`signed_prefix`（键`1/10/50/100`）、`within_state`（std/range）、`per_rng`、`first_two`、`last_two`、`leave_one_rng`及`status`；`proceed_short_training`固定False。必要NumPy反例测试只验证统计契约，不是source、闭环物理或方法验收。

## 7. 必要工程验收与硬预算

语义正确先于utility。CPU actual tiny模型先验：q=p同噪声退化、初始h共用、p初次不吃current token/latent、以后双方正常posterior、未来图像/动作不影响过去prefix、不同branch停止不移key、raw/applied不混淆、真terminal吸收与time-limit/reset拒绝、readonly模型/RNG/counter隔离。真实MuJoCo/production wrapper另验FromDM `_done`及完整恢复，不能以mock物理证明真实标签。

若为注入私有噪声拆开正常policy，须用未改生产B1 `model.policy` 作oracle，核actualcast/core/distribution/key/actor/carry。真实size50m在32个预定calibration源RGB/carry/source-eval-key上应核q正常路径与source保存raw/after-carry一致；诊断不得因BF16/JIT图差异自行放宽成“生产精确”。核失败先保留工程原因并停止，不能以大批utility或容差掩盖。诊断纯调用最终参数、optimizer/norm与原counters不变；oracle临时eval计数须独立副本或finally精确恢复。

每个已执行分支同时做两类独立复放：固定缓存RGB＋相同keys/carry的policy离线重演，raw动作/后续carry逐叶字节核对；固定stored-applied动作从同完整snapshot再执行physics，逐步integration/task/RNG/counter/terminal/wrapper状态按原1e−8要求核对，F32 reward编码精确。**F64主reward须逐步byte-exact复放**，不能以F32相等或state误差在1e−8内冒称F64标签重演；不精确则保存差异并停止语义验收，交审具体原因，不看到utility符号再放宽。积分state shape/dtype及finite先验，NaN不能通过max-error比较。必要时重render并保存pixel/hash差，只称实测反馈路径加条件复放；没有重新闭环逐位核对时不声称renderer或完整闭环bit-exact。

新增上界为32状态×4 RNG×2分支×100=25,600环境动作，完整physics动作复放另最多25,600；固定RGB的policy复放另最多25,600前向。source已验且独立审计绑定的H100参考复用，不无理由重复；源RGB实际carry/source-eval-key的32位置production oracle仍必做。模型冷加载/编译、全部oracle、render、主分支、两类复放、分析与写盘均计入**单卡3600秒父进程硬监督**，不是frame-loop cooperative check或预计ETA。超时只终止自己的worker、保留新attempt的不完整证据，不换卡、加预算、追加RNG或只发表已完成好点。

实际GPU4在source释放后须重新fresh核查，CUDA/EGL都绑定物理4；不得与source或未知作业共享。实际SHA、config/source/audit hash、环境、私有seeds、资源、退出码、实测时间/显存和每branch执行步数均落运行记录。GPU4/3600秒是本协议冻结的资源范围，不须额外GPU；实现仍须确定SHA/clean本机与服务器CPU验证及独立代码审查，任何工程gate未验前不启动GPU。optimizer更新为0，原模型/optimizer/norm与training/source counters冻结，只有明确命名的私有反馈流推进。

## 8. 产物及完成边界

新ignored运行目录 `/data/Policy_Discrepancy/runs/dl-feedback-<actualdate>-<sha>/cp300000-attemptNN/`。plan绑定source complete/audit及八个artifact字节hash、位置/实际carryhash、全部keys、dtype/contract/预算；逐episode保存fresh/source初始RGB对照、每branch真实RGB、h/latent/logits/raw/applied、float64/F32 reward、flags/masks、完整状态/复放摘要。完整数组及可信状态pickle不提交Git；本机仅镜像小plan/result/独立审计到ignored `analysis/outputs/dl-feedback/`。

独立CPU分析应不import生产分析器：从逐步reward/mask复算所有signed prefix、episode等权统计/bootstrap、终止/完整性与分类；核新来源、位置/keys与参数收据。它不能独立重演模型时须明说，产物一致性不是方法有效。

所有输出保持 `training_authorized=false`、`proceed_short_training=false`、`eligible_method_training=false`、`method_effectiveness_established=false`。即使平均机会阳性，仍需真正未见数据、独立目标子集标签、第二信号相对D的增量、背景/OOD/actor能力等对照、真实rep压缩和条件训练。真物理状态只作诊断证据，不作为vision64/propriofalse部署gate输入。

本轮交付为科学协议冻结、模型／物理／runner 实现及必要 CPU 反例测试；source complete与独立审计已完成。反馈工程验收尚未闭环，首个GPU尝试及必要容器修复见下节。首选继续当前完整授权任务：完成工程r2确定提交CPU复验、fresh资源核查、新attempt有界反馈诊断及独立产物审计；不另选gate或启动训练。形成可正式实验方法仍未完成，也不保证本对象找到有效方法。

## 9. DL-feedback-engineering-r1 实现与当前验收边界

本节较早CPU与额度中断记录保留为历史阶段；当前实际验收以第10节及验证日志为准。

入口为 `scripts/dl_feedback.py`，模型适配器 `scripts/dl_feedback_model.py`，物理记录／复放 `scripts/dl_feedback_physics.py`。模型只在受锁保护的 JIT trace 中替换首次 `obslogit` 为同 h 的实际 `_prior(h)`，保留生产 `_observe`、cast、unimix、latent／Normal 采样及原 NJ 消费顺序；退出时恢复 Python 方法。纯调用禁止状态创建／写入，原参数及全部 counters 冻结。模型额外 receipts 对真实 size50m 图的字节一致性仍未验证，tiny 通过不能替代32源位置 oracle。

工程 key 定义在看新 utility 之前固定：`numpy.random.default_rng([20261018,0x46445231,episode,position,pair,position+s]).integers(0,4294967295,size=2,dtype=uint32)`；s为0至99，最后合法绝对 clock 为999。high为exclusive，与实际eval生成约定相同；不含branch。保存全100 planned roots，包括terminal后的未执行keys；mask负责区分执行，未执行段不生成模型proof或图像。

runner 先完成全部32源RGB／carry／eval-key oracle，再生成反馈收益。每branch完成真实生产 step 后分别做 stored-applied physics 与 own-cached-RGB policy复放，初始h及共用 latent／actor key按实际receipt核对。F64/F32奖励、完整状态／task／RNG／wrapper、raw/after-carry/proof字节检查均不放宽；复放失败保留独立 `failed-replay-*.npz` 与日志，不生成完整utility裁决。父进程硬预算涵盖child全部工作，Linux只清理自己新建的process group，包括失败leader留下的descendants；Windows fixture只证明direct binary child，不声称树清理。

每episode产物八份、每份32records：numeric轴为position/pair/branch(q0,p1)/step；保存全keys、raw/applied、F64/F32 reward、discount、flags/action/observation masks、source/fresh初始RGB及trusted project record pickle。result/complete/supervisor分别绑定plan/hash、完整8episode/256branch、exit0/无超时；残留result文件不能单独证明完整验收。完整数组只在server runs，ignored独立审计器不提交Git。

本机 CPU 已验证：统计9项；实际tiny BF16模型2项；真实quadruped＋production wrappers物理6项（policy/RGB明确stub）；synthetic cached-policy replay4项；监督6项中Windows4项通过、Linux process-group2项skip。pure统计、缓存复放和监督fixture不是方法效果。真实 source／size50m／CUDA/EGL／renderer／本对象F64奖励／独立产物审计仍待GPU运行和完整审查。

独立runner审查与产物审计器的schema接线在代理额度限制时中断；已完成的模型／物理文件和统计反例保留。ignored `analysis/outputs/dl-feedback/independent_feedback_audit.py` 的五项synthetic自检通过，但 `audit_artifacts()` 仍明确 `NotImplementedError`，**不能称独立产物审计ready**。根Agent没有以自审代替这项独立验收，不启动GPU。后续先完成独立runner/hard-budget检查、补齐此审计器与篡改自检，再做确定提交server CPU验收；恢复工作不重复申请已经有效的sv3 GPU4–7授权。

**17:56恢复进展**：额度恢复后独立runner审查及replay反例复核完成，无已知阻断；审阅runner SHA256 `eb57fb0e00bf6f3d9befad85f09558dbdd383c556615e0111a2c39c5e500f71b`。首次decode前核source/runtime七项依赖版本；每adapter首次render前核source／实际F64动作bounds。replay隔离且拒绝carry/obs/key输入变更，并检查raw完整shape；新的四项synthetic tests独立通过4项/9subtests。Linux监督及真实size50m条件尚未验，artifact独立审计schema接线继续进行。GPU仍未启动，不把“审阅通过”当控制价值或方法通过。

## 10. 首次实际尝试与必要工程r2（18:07后）

实际执行和测试SHA为 `bd0af3a47ac0861d932a4a08af2dd1ce84754dbe`，已推送并独立核远端研究分支，未合入main；仅sv3新增clean detached worktree，canonical repo保持干净 `d9be5574878d2c3f117cc6667be844f9a527eb84`。server CPU统计9、actual tiny BF16模型2、真实quad/wrapper物理6、synthetic replay4、监督6，共27项全部通过，含Linux真实process-group超时和失败leader后代清理；pip check及source七依赖/32artifact hash/只解码8calibration检查通过。fixture/小模型通过不是实际收益或方法验收。

独立artifact审计器r1已完成，纯NumPy/stdlib，不import生产分析器或pickle解码；server CPU五项基础、三套完整synthetic产物及24类重新hash篡改自检通过。它核数值数组、源pins/keys、F64/F32编码、实际动作映射、完整性/监督退出及独立signed统计/CI分类；不独立执行真实model、renderer或physics。只接受完整exit0产物，不把残留result当完成。未对本次失败产物作科学审计。

GPU4 attempt01目录 `/data/Policy_Discrepancy/runs/dl-feedback-20261011-bd0af3a/cp300000-attempt01/`；18:06:29–18:07:30北京时间，父监督worker实测38.720526秒、exit1、非超时，PID1828665已终止并请求清理自身group；1秒采样37点峰值2501MiB。32个预定源RGB/carry/eval-key上未改公共policy及适配器q的raw/after-carry逐字节oracle全部通过；随后 `source_observation()` 的NumPy scalar reward/flags无法满足物理入口要求的0D ndarray，在首次fresh render和反馈分支前失败。无feedback NPZ、F64收益、result/complete，最终整诊断参数冻结检查未到达；不能称闭环或utility验收。完整日志/plan/budget/资源收据留server，本机小文件镜像 `analysis/outputs/dl-feedback/bd0af3a/cp300000-attempt01/`。

必要修复 `DL-feedback-engineering-r2`：单帧索引改为 `np.asarray(...).copy()`，保留全部dtype/shape/bytes，恢复生产wrapper的0D容器；物理精确检查、模型/RNG、state/key/carry、科学r1和预算不变。runner修复SHA256 `255bb4edfede73608636973902c79caf15c26ab83311665655207014F9A3F3A5`（十六进制大小写等价），两代理独立复核转换及实际模型 `_obs` B1字节相等；新增真实quad-wrapper源数组到初始化接口整合反例，全部7项物理CPU测试通过2.077秒。r2独立审计器只更新工程版本，原r1不可变副本保留；新SHA256 `b3a18990f3e9c29606e60216ea06fa5d022169ea1c8cae3f3bc39f39c47c6cbc`，同一五基础/三完整/24篡改自检通过5.07秒。仍须新SHA/server CPU及新attempt实际验收，不复用失败attempt作完成证明。

18:07:40北京时间fresh query/pmon确认GPU4–7各0MiB无compute作业，GPU0/2未知作业未干扰；GPU4已释放，这是时间快照。最低1张独占A100串行；本对象只用GPU4/3600秒，授权四卡无需占满。总方法验证GPU小时尚未冻结，过去2.5–6.5GiB峰值不能保证其他显卡/训练资源。
