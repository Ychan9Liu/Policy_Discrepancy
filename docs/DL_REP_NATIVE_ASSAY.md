# DL-rep-native-assay-r1：全窗口 rep 干预到原 actor 闭环回报

## 状态、目的与事实来源

**状态：在完整预验证授权内由Agent采用的有限诊断r1，尚未执行。** 不是用户逐项确认的正式方法、正式M2修改或方法效果结论。当前实现、确定SHA的CPU/Linux验收、实际source认证、fresh设备核查和完整执行封口仍待完成；本文不表示资源已预约。当前只写文档/工程适配并审查源码和小JSON元数据，不读取实际checkpoint、RGB、source数组或物理状态，不启动科学模型、GPU、渲染、训练或实验。执行版本、完整源码pins、实际命令及预算收据须在运行前封口。

本对象直接检验：从同一完整初始状态进行128次正常学习，停止整个 loss 窗口的新增 rep 梯度，是否改变**原 Dreamer actor 在正常后续 RGB/posterior 反馈下的真实任务回报**。正常 B128 为主对照，初始0只用于解释；不以七控制 F 的选择改善代替 actor 性能。A是因果诊断臂，不是拟部署方法，也不是选择性保护收益的上界。

必读：[AGENTS.md](../AGENTS.md)、[RESEARCH.md](RESEARCH.md) DL-30、[DL_RESET_JOINT_ASSAY.md](DL_RESET_JOINT_ASSAY.md)、[DL_RESET_JOINT_RESULTS.md](DL_RESET_JOINT_RESULTS.md)、[DL_RESET_SIGNAL_ERROR.md](DL_RESET_SIGNAL_ERROR.md)、[DL_FEEDBACK_RESULTS.md](DL_FEEDBACK_RESULTS.md)、[DL_PREVALIDATION.md](DL_PREVALIDATION.md)。源码接口为 `dreamerv3/rssm.py`、`dreamerv3/agent.py`、`embodied/jax/agent.py`、`scripts/dl_reset_learning.py`、`scripts/dl_reset_all_rep.py`、`scripts/dl_native_actor.py`、`scripts/dl_feedback_model.py`、`scripts/dl_feedback_physics.py`、`scripts/dl_production_source.py`、`scripts/dl_action_contract.py`、`scripts/dl_resources.py`。

**已有有限事实**：旧0/B/R对象确认初始q的共享F有限选择增量；局部R没有建立相对B的效用收益。R仅删除reset位置的当前rep梯度，未来rep递归、共享参数、其它损失和旧momentum仍保留。旧D+K/固定拟合S未通过其辨别门，不由新对象翻案。最新冻结输出分解显示cal LOEO的p参考动作变化，cal和blind的目标/拟合方向存在转移；这不证明它是全部根因，也不授权改旧target、重新拟合或调整阈值。

**工作假设**：整个窗口的当前rep梯度可能影响可供原actor利用的任务信息；同状态/同数据/同keys的A−B能识别其有限净作用。**尚无证据的推测**：未来rep、momentum、actor未利用、过拟合或读出转移中的某一个是阴性主因。本文不把这些解释写成结论。

## 已采用的有限对象与固定范围

下表是本有限诊断r1的固定规格；不按新结果选seed、H、更新数、动作策略或保留episode。采用本对象不等于正式方法就绪。

| 项目 | r1固定规格 |
| --- | --- |
| 任务和模型 | `dmc_quadruped_walk`；原baseline训练seed0、300k checkpoint、size50m；64×64 RGB、proprioFalse、camera实际2、action repeat1 |
| 学习臂 | A128，从原300k完整checkpoint开始，128次full `Agent.train`；整个64位loss窗口rep当前梯度停止 |
| 主对照 | 已审计的B128完整终态；认证复用，不冒称本次重新训练或独立optimizer重演 |
| 解释对照 | 初始0完整状态；没有额外学习臂、R臂、新gate或新信号回归 |
| 新评价 | 环境seed整数1024至1055，升序全部32个；各模型各seed一个完整正常episode |
| 动作和反馈 | 官方原native sampled policy；每一步接收自己实际环境的RGB、reward和flags，正常posterior/carry，禁止当前或后续prior替换 |
| episode长度 | 每条1000个真实suite transitions；repeat1；reset不计transition；最后一帧不再选/执行动作 |
| 主终点 | 32条episode等权的真实F64累计任务return，配对A128−B128；不取正部、不按存活步数或D加权 |
| 解释终点 | A128−0、B128−0，同一新评价程序；不能据解释对比改主终点 |
| F | 只作辅助解释，见下节；不生成新32个episode的七控制counterfactual标签 |
| 资源 | sv3/lyg0326物理GPU4，一张独占A100；完整父入口7200秒硬上限、最多2GPU小时；无吞吐充分保证 |
| 独立验收 | 另一个0GPU owned CPU父入口1200秒硬上限，全96,000个stored-applied官方physics复放及独立算术 |

评价共3模型×32episode×1000=**96,000个真实transition**，通常RGB为3×32×1001=96,096帧（包含reset与最终帧）。它是新的正常closed-loop任务评价，不是首动作＋zero尾、固定动作库、原H100一次posterior省略或持续prior运行。评价数据不回流任何optimizer或读出拟合。32个新环境seed不等于32个独立训练seed。

若提前last/terminal、1000步末缺少官方last、出现NaN/Inf、错误reset、丢失动作/状态或预算不足，整对象不完整。禁止padding、自动重置续接、替换episode、缩短长度、增加seed/H或发布部分样本的主裁决。末端`is_terminal`按真实官方discount记录，不把time-limit和真正terminal混同。

## 学习A及旧B认证复用

旧主目录：`/data/Policy_Discrepancy/runs/dl-reset-joint-20261012-c030e26/primary-attempt02/`，原producer SHA `c030e26f688ed227aa43866bd68dcf4352a073e3`；独审 sibling `independent-audit03/`，审计SHA `d987650c749b8917e2ea80a719d586232088f501`。本机小镜像位于 `analysis/outputs/dl-reset-joint/dl-reset-joint-20261012-c030e26/`。必须认证旧完整父退出/外CLI exit0、freeze、result、独审及其hash-chain，不只相信单个JSON声明。

| 原字节或状态证据 | 已审值，运行前仍需核原字节 |
| --- | --- |
| 原300k checkpoint | `/data/Policy_Discrepancy/runs/dl-prevalidation-bootstrap-20261010-ae8372e/sources/cp300000-agent.pkl`；SHA256 `2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9` |
| 原config | SHA256 `11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445`；实际文件路径须随采用命令记录，不用新默认config替代 |
| `B-checkpoint.pkl` | SHA256 `c878652b0fca1313ed9d1536eec838e2972e3dea611e4b309f94b32a63272670` |
| `B-complete.json` | SHA256 `b2596ff268873725e54a4008d760961e55a67c00c5318477b5ccc98ff280b8fb`；phase为B，不是不存在的`B128-complete.json` |
| `contexts.pkl` | SHA256 `c8a6a964455e2a579c4ed76f1de19cc38931955279b36b45445310414451e1ab` |
| 初始完整state digest | `216fe9be592190709fe4f22ad23ab89808baa9eb49b618046855861a11276c78`，原`dl_reset_learning.tree_digest`定义 |
| B最终完整state digest | `9e670873990510f97174f07b04a407ed290ad762057a33e8884818d5e2fd013d` |
| B最终params digest | `1a87e517f04d27705f7a6193ca96cad0f8e9271f9fff6f169bc23bed0d912e64`，同一tree digest定义，不混用另一`parameter_digest`算法 |
| 原batch digest | `3bfdef05a0dde48f04f37a67f51da86a236420e41f7cf9476a8ae9bcf67deeec` |

A从**原0状态**开始，不从B128或R128接着训练。复用固定cal IDs `[9,14,15,18,20,24,30,31]` 原完整1001帧、source八hash、原`contexts.pkl`和baseline annotation证据。每batch为八cal各重复两次：循环前一cal的真实terminal1000作为context，随后本cal reset0至63作为loss；原`DL-reset-learning-r1` layout/stepid、batch16、loss64、context1和128次每轮`init_train`均不改。它仍是离线episode人工串接、reset富集的重复数据诊断，**不是普通随机replay或原runner连续恢复**。

禁止为A改变stepid版本、重建新的context sample、重新排序/采样cal、增加训练数据、使用新评价轨迹或改正常seed counter。A initial/source-checkpoint/initial-params/calibration-arrays/context/batch/layout digests须与旧B逐项相同；128条`prepare_batch`真实uint32 seed及其counter必须与旧B逐项相同。参数分叉后实际latent或想象路径可以不同，不要求其输出强行相同。

A只在Agent/JIT构造前安装进程内loss包装：原一次RSSM.loss产生前向值后，training时整个`losses['rep']`替换为`stop_gradient(rep)`。保留free nats1、dyn/rep系数1/.1、其余loss、reward_grad/repval_grad=True、ac_grads=False、正常optimizer/moments/norm/slowvalue；Dt/M2/DtLatch overlays关闭。不冻结encoder/posterior/carry，不清momentum，不把rep loss数值设零，不将干预解释为所有历史rep影响清零。

官方optimizer scalar key为`opt/step/value`、dtype int32；A与旧B同样74493→74621，updates/batches各+128，其余原counters不变。每轮保存真实train metrics、reset及全窗口post-free-nats active/count/excess，非reset另报。日志须与loss分支隔离并通过确定版本的守卫/梯度工程验收；不能仅凭新metric存在宣称A严格匹配旧B。旧B没有全窗口压力测量，不伪造该值或将reset2048/2048替代全窗口计数。

完整输入payload先核SHA后同一字节反序列化；旧32source先全部hash认证再仅打开8cal，禁止解旧24source/RGB作为新的确认数据。保存A完整终态并reload，要求实际`agent.save()` typed tree digest等于A完成收据。B也必须在新评价加载后等于上述旧完整终态digest；0等于原标准load初始状态。不能只有actor权重/regex load，不能以JSON指标认证替代真正模型状态核对。

## 原native policy、当前参数与私有随机流

**源码事实**：`dreamerv3/agent.py:126` 的policy正常enc→dyn.observe→pol→sample；`mode='eval'`没有改为mode/mean。`embodied/jax/agent.py:255` 公共policy使用`policy_params`，且在当前动作完成后才应用`pending_sync`；旧训练时enable_policy=False的对象不能直接拿来调用公开policy。

每个评价模型使用新、独立的Agent，完整load对应0/B/A状态；科学私有评价**直接读取当前`agent.params`**，不经旧policy copy。为了公共production oracle，构造enable_policy=True的评价Agent并显式`sync_policy()`，检查`pending_sync is None`且所有`policy_keys`的policy copy与当前`agent.params`逐叶dtype/shape/bytes相同。之后不train、不同步新状态、不沿用旧pending copy。完整model/optimizer/norm/counters在私有评价前后必须不变；公共oracle临时改变的eval counters/seed须finally精确恢复，policy copy另作typed digest认证。

首选新薄适配`nj.pure(model.policy)`直接读取当前params，并复用已验只读/空间/随机数守卫；**没有prior omission**，不改actor/RSSM采样、unimix、BF16、bounded_normal或cast流程。私有pure调用`create=False`并track所有attempted create/modify，发现写入即拒绝。公共policy copy只用于已同步的production oracle，不作为科学私有读取来源；公共policy直接推进eval counter的行为不能混入科学私有流。

固定roots直接采用现有官方eval定义，config训练seed仍0：对环境seed`s∈[1024,1055]`和step`t∈[0,999]`，用`np.random.default_rng([0,0x4556414C,s,t]).integers(0,np.iinfo(np.uint32).max,(2,),dtype=np.uint32)`生成private root。相同s/t跨0/B/A相同；不同s/t按命名规则独立；不含arm/model/调用顺序/GPU/phase ID。它是新私有评价流，不复用源policy carry或声称source原q sample一致。初始化官方init_policy(1)采用相同episode的t0 root；该模型源码初始化为真实零carry，无采样消费。每一步actual nj.seed的latent/actor子key须按生产顺序记录，当前接口应恰好2次；不得另造子key，调用数变化即失败。共同随机数不保证非线性控制方差一定下降。

每episode新官方初始carry、previous raw action为真实官方初始0；reset观测`is_first=True`进入原mask，随后各模型保留自己的完整enc/dyn/dec/prevraw carry。不得逐步清carry、跨模型共享carry、用physical applied action替换prevraw，或用上一模型/旧source的图像。carry保存实际BF16和真实float32 raw action，不从F64 receipt回写policy。确认h/latent/raw/carry有限且raw/carry一致；这里不要求不同模型的h/prior或轨迹同字节。

固定每模型每episode步骤0/500/999做cached真实观测的公共production oracle：临时设置同eval seed/counter，调用公共policy，与private接口的raw/after-carry逐byte一致，finally恢复全部eval counters/seed。这些288次辅助模型调用不执行物理动作或新render，全部计入7200秒。工程tiny fixture只验接口；实际oracle才验该加载模型的production语义。oracle不能消费后续私有root或改变另一模型轨迹。

`NativeActor.initial(seed)`和`.step(carry,obs,seed,t)`采用官方model的B1 batched carry；它不同于公共wrapper的split/list表示，oracle仅作精确stack/split往返。`evaluate_episode(adapter,seed,initial_observation,transition,check=...,oracle=adapter.oracle)`的callback提供真实下一`observation/action_applied/reward_f64/discount_f64`；driver另保存1001 full snapshots、实际action stages/reset/render证据。返回observations字典1001帧、raw/applied1000×12、F64reward/discount1000、root/latent/actor keys1000×2、carry_hashes1001及三个oracle收据。helper不认证真实source、physics、renderer或资源，不能把其布尔字段当完整验收。

## 正常环境、渲染与完整记录

每模型每seed独立创建官方`quadruped/walk`，`task_kwargs.random=s`；执行真实DMC＋`main.wrap_env`链：`ClipAction→CheckSpaces→UnifyDtypes→NormalizeAction→DMC→ActionRepeat1→FromDM`。camera2（原config -1在quadruped解析为2）、64RGB、proprioFalse。不改XML、动力学、reset随机化、控制幅度或观察内容，不nearest-neighbor到七控制库；无zero尾、背景替换、cue遮挡或额外物理事件。

每seed三个arm的fresh reset完整integration/task RNG/clock/reset/FromDM done与初始RGB须相同；各自自然分叉后不强求相同状态。reset前后、每真实transition之后保存官方full snapshot，明确1001帧和1000个执行动作；末帧没有policy调用或可执行placeholder。首次及后续RGB由各自实际renderer产生，不能cached旧RGB续行。渲染前后核snapshot相同，证明读图未推进环境。

记录float32原sample、生产Clip/Unify/Normalize各stage及实际suite float64 applied action；套用已验production action合同并与实际wrapper输出byte一致。记录suite原F64 reward和供policy的原F32 reward，两者分别认证；主return只累加实际F64 transition reward，不重复加reset/过去reward。flags、discount、计数、task RNG、integration及FromDM done完整保存。任何new snapshot pickle先由外层原字节SHA认证再解，不能hash后重开未核payload。

**已采用设备范围**仅sv3物理4，UUID预期`GPU-9c070d5a-ac68-4c26-9522-a790ad874b23`；启动时须重新实测。按已验映射设置CUDA_VISIBLE_DEVICES为该UUID、进程内cuda0、MUJOCO_EGL_DEVICE_ID=4、CUDA_DEVICE_ORDER=PCI_BUS_ID、MUJOCO_GL/平台egl。先fresh核物理4占用，再在新worker第一次实际GPU计算/渲染后核compute与EGL实际UUID及PID映射，记录sourceSHA/probe/environment和时间；旧无render reset对象的proof不能当本轮实际渲染证明。物理4冲突则报告，不挤占、不杀其它作业、不自动改5–7或增加卡。本文件未做当前资源核查。

## 统计、验收与停止

固定计算每条`R[m,s]=sum_{t=1}^{1000}reward_f64[m,s,t]`；主配对`Δ[s]=R[A,s]−R[B,s]`，32episode等权。固定2000次whole-episode paired bootstrap，`default_rng(20261026)`一次生成2000×32共同draws，95% percentile `method='linear'`。0/B/A及解释差使用相同draws；不把1000步当独立样本、不更换区间方法、不事后挑prefix/seed。保存全部32条signed差、mean/std/range和原returns，显示少数episode主导情况。

有限主裁决：A−B lower95>0为本单训练seed/单checkpoint/固定128更新对象的闭环净收益支持；upper95≤0为本对象未支持正收益；其余为不确定。这不是等效性检验、跨seed方法复现或实用效应保证。尚未确定最小实用回报幅度，禁止把旧H20的δ=.2机械转为1000步性能阈值或事后选择。A−0/B−0仅解释；B相对0下降和A回升可加强损伤/恢复叙述，但不据此改主裁决，也不把其它loss的贡献归给rep。

**F辅助范围**：当前对象只引用已审旧0/B有限F结果作为背景，不新增A-F拟合或新32的七控制truth，不重fit第二信号、不换target。不因主actor结果不好再补F求阳性；旧F阳性不能替代本主终点，旧F结果也不抹去新真实actor结果。

独立CPU审计1200秒/0GPU，owned父预算包括所有imports/hash/加载/复放/算术/写盘/退出，不只step loop。另执行全部96,000个stored-applied官方suite步，不render、不call模型、不更新optimizer；复用已验complete-state/restore原语但对新H1000记录实现完整loop，不能直接调用只接受H20/七控制的旧validator。每步要求原F64 reward、flags、完整integration/task RNG/clock/reset/done byteexact；不把旧1e−8 source guard当新byteexact证明。记录同一模拟器复用；不称独立renderer、policy或optimizer重演。

独立`math.fsum`重算returns/配对统计/bootstrap，数值绝对差≤1e−10且CI零侧/分类exact；跨边界失败，不放宽。原records、模型/contexts/source/hash-chain、加载前后fullstate、copy/currentparams和key/oracle凭主运行typed收据认证。新auditor只hash模型/RGB，不解或调用；CPUphysics认证与GPU模型收据认证的边界必须分别报告。

完整GPU父入口7200秒包含认证原source/旧B链、checkpoint加载、A构造/编译/128full train、3×32新闭环评价、288次oracle、统计、hash、写盘和完整退出；父只kill本次owned进程组，helper继承deadline，末端late写盘不得发布可验收success。无完整退出、resource端点、external CLI exit0、全部96条episode或独审封口时不得宣布验收完成。保留每个失败attempt，工程修复不得改科学语义、采样流、阈值或预算后拼接结果。

新产物目录规则 `/data/Policy_Discrepancy/runs/dl-rep-native-<date>-<sha>/primary-attemptNN/`；独审sibling `independent-auditNN/`；本机ignored镜像 `analysis/outputs/dl-rep-native/`。保存plan/预算、完整配置与源码/环境pins、0/B/A状态及policy-copy digest、旧B认证reuse链、A128逐轮metrics/keys、freeze、96条新轨迹/物理/随机数收据、统计、supervisor/externalCLI/独审、resource start/end与峰值采样。checkpoint/source/RGB/回放不入Git。

2GPUh为**本r1单次attempt的一卡完整占用硬上限**，不是预计核时、整个科研剩余预算或A alone训练成本；CPU独审另计0GPU。历史baseline300k、source collection、旧B/R训练及旧审计成本不计入这个新增上限。既有6.47GiB采样峰不保证新渲染/训练显存或瞬时最大值；实施前列实际估计和硬预算，运行后报告实际全入口/worker时间与峰值及失败成本。超时按不完整停止，不能自动延长7200秒。

## 可证伪解释、下一分支及未完成目标

| 新有限结果 | 可以说什么、首选下一分支 | 不能说什么 |
| --- | --- | --- |
| A−B闭环净收益支持，状态/压力/接口验收通过 | 支持本重复cal学习对象中整个当前rep项对native回报的有限净抑制；下一任务才是冻结现实第二来源与bounded gate，比较强度/对应/reward-only | 不能称posterior信息保护已定位，不能称新双信号有效，也不能用A当gate上界 |
| 已有有限F可读证据，但A−B actor收益无支持 | 仍缺把已有接口信息转为学习/actor收益的证据；首选检查正常actor/value学习及任务需求，停止继续优化F/S回归求阳性 | 本对象没有新A-F证据，不能归因于A改善了F；不直接采用gate |
| actor收益支持 | 以真实主终点保留结果；旧F仅限制其readout family，机制归因仍需独立证据 | 不能断言收益来自特定cue，不能为补机制解释事后加入A-F |
| A−B未支持正收益或不确定 | 停止本预算对象，明确无有限净收益证据；结合0解释及已有根因记录决定是否撤回本任务的压缩损伤主张或另立有必要性的普通replay路线 | 不能宣称全rep有益/无效或所有selective gate无效，不能自动增加更新数/seed/H |
| 任何工程/physics/退出或输入认证失败 | 保存失败并仅修工程；新attempt同科学规则 | 不得发布部分科学裁决或拼接旧/new训练/评价 |

反例包括：A删除有益rep导致全局失稳而选择性gate本可有益；旧momentum或其它loss仍抑制信息；128重复cal改变policy但不是普通replay；actor改善来自通用正则变化而非新增cue；共享单模型偏差使内部value支持错误依赖；同样task seed不提供跨训练seed泛化。A−B是整个当前rep项的总学习干预，不定位encoder、posterior某一特征或完整历史之外新增信息。

**整体目标仍未完成**：本对象不含D+第二来源辨别、不含合法nuisance/错误依赖覆盖、不含选择性gate、压缩强度匹配/conditional-shuffle/reward-only/OOD对照、普通随机replay学习或多任务多训练seed。即使闭环A阳性，也只能把学习干预接到原actor性能，不能宣布可正式实验。首选继续当前完整授权chat，完成确定版本实现/旧B复用/新闭环接口/统计/Linux验收与资源核查后执行一次完整对象；不自动创建或发送其它chat。对象结束后的下一路线须按上述分支，不再把一串有限读出assay代替完整方法。
