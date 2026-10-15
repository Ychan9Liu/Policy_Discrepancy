# DL-NM1 C 实际执行与资源记录

2026-10-16，版本DL-NM1-cal-r1，状态：**attempt01/02/03工程失败已封存，尚无完整校准或独审**。
这是Agent在既有完整预验证授权内采用的128批cal-only，不是累计学习或正式实验；
后续T4096/整阶段6GPUh仍为建议，尚未采用。

**最新：attempt03实际工程失败**。actual/tested abefbb3，原handle85427已终态、实际SSHexit1/
3101.188秒；原parent exit1/3099.069495994秒，supervisor worker1/3099.005093803秒，均在原3600秒内。
128条batch_collected连续0..127后，首S控制的GPU coefficient被jax.jit作为闭包常量物化，
`_array_mlir_constant_handler → val._value`隐式device→host被原全局disallow拒绝。
局部host_to_device许可未涵盖这个真实CUDA编译路径；CPU41测试不证明CUDA编译常量传输通过。
无generation/final闭包或控制资格，不是方法阴性，不读取frozen结果、不拼接原128继续。

原parent538626/start410327282、worker538639/start410327396均已退场；
01:07:38–39真实XML/pmon物理5无进程/0MiB、4仅原feature worker542278 C/2492MiB、6/7无作业/0MiB。
01:07:40 /proc原身份已不存在，reference0原父子仍在/274个unit complete文件、无generation/终态。
下一仅修局部完整`jax.transfer_guard('allow')`及作用域恢复，不改原数值body、F32系数常量、keys、bins、科学门或预算；
须先确定提交CPU及生成数据的真实GPU5小回归，重现旧D2H故障、核修复与all-allow参考逐位相同及异常后guard恢复，
再在原GPU5/128/3600秒fresh完整新尝试，不能再以CPUgreen替代这项CUDA验收。

失败小镜像 `analysis/outputs/dl-nm1-cal-transfer-abefbb3/failed-attempt03/`，
root实际SSH收据和原log在其父目录。完整rawSHA：
producer terminal `ac22259b38d31e61447adc32b59f29e8ce9dcbf52ab0b32d3f3a80eeed199bdf`，
supervisor `9503689d1c7257b91ab5af652473a65f2d6f3dc8062c5544a244d8f5067b503f`，
worker log `b2b757da6812c5b218299d8c630d9d67fd75d0a8881310d708db8af4821f9388`，
root SSH receipt `be4407baf4e388520780b5fb33d6c398884cc9b084816af7a8cc615519879755`。
第二Agent独立核本机日志/终态，不冒称其再次查服务器；root上述/proc和GPU核查另有带时收据。

**2026-10-16独审准备补充，未启动且不得用于失败attempt03**：actual run/audit Git固定abefbb3，
外部controller38646B rawSHA `9313cb42e847adb6c7802adb3c6de9fc16625d99765f09799f3bcc16cb62ff55`，
本机包`analysis/outputs/dl-nm1-cal-audit-abefbb3-controller/`；29个原3f通用函数AST未变，
25个纯元数据拒绝反例及第二Agent只读审核通过。完整负scientific qualification不阻断integrity验收。
新增强制root实际C SSH退出receipt及rawSHA，保持原900秒、原auditor六CLI、无NN/反传/Adam/GPU重演。
只读原producer进程退场，不登记或清理其它作业。controller已实际stage至
`/data/Policy_Discrepancy/runs/dl-nm1-cal-audit-inputs-20261016-abefbb3/launch.py`，读回rawSHA全同、
实际abef代码clean；未执行。root收据/日志及后续小镜像放
`analysis/outputs/dl-nm1-cal-audit-abefbb3/`，fresh审计输出
`/data/Policy_Discrepancy/runs/dl-nm1-cal-audit-20261016-abefbb3/attempt03/`。
原handle85427真实SSHexit0/whole≤3600、128完整generation及原组gone后，
才把原SSH receipt bytes放新audit inputs/producer-outer-execution.json并提供两份实际pins启动。
审计自身还须真实外层SSHexit0/whole≤900；925秒root观察上限不延长成功预算，不用内部terminal自证。

实际运行/有限CPU测试Git为`8d63fff8b15a49c7e2e7bc84fd0f9c3481e6acb0`，
clean代码`/data/Policy_Discrepancy/dl-nm1-cal-cpu-code-20261015-8d63fff`，
本次报告文档不移动运行源码。该版本142项clean Linux有限CPU测试通过，见[工程证据](DL_NM1_ENGINEERING.md)。
执行规范[DL_NM1_TRIAL.md](DL_NM1_TRIAL.md)、产物/独审契约[DL_NM1_CAL_ARTIFACTS.md](DL_NM1_CAL_ARTIFACTS.md)保持原字节。

## 实际输入与固定配置

- 原N1M seed0/size50m/BF16，仅视觉64×64；checkpoint为原249493更新初态，不新增训练seed。
- 原checkpoint/config完整SHA及原state身份沿[已认证输入](DL_NM1_ENGINEERING.md)，不采用新模型。
- 原1280NPZ、16源链、1M动作的私有replay，完整迁移见[迁移记录](DL_REPLAY_STAGE.md)；
  stage SHA `00063544c10facde529f82fac07e759a2d44306aedd77924f30ff185e51519f8`。
- 固定128独立抽样批，每批16×65 context、16×64 loss；PCG64计划root20261101/tag0x4E4D4341。
  每批从相同完整初态执行B/B重复/rho0/N的真实scratch更新及B/N独立完整参考梯度；丢弃返回更新，
  committed optimizer/replay updates应均为0，无累计4096学习。
- normal-only A沿原H1m并排除current is_first；原raw K、e与f定义、alpha20/rho.5/free-nats1、
  原完整loss/梯度开关不改；S/W/P资格和128位置/8链/16批作用、1%/10%/50%门沿预先采用规范。
  不新增constant c或正式评价计划；S/W系数须依128完整真实张量计算后独审，不填默认数。

## 前驱完整审计及真实封存

原R5 producer1584完整512/32；成功physics auditor为
`277c18001d353d9101a3622959351d3994699adb`，actual exit0/655.220153340秒/1800秒。
report SHA `144c4490e69283ab14ad1a96fd8cbd9c47945875797a91ac6d420dade9e216cd`，
外层terminal SHA `d6164a94bdb7b9a95579454d539643d705f808d798d8f44043dad3d0101c8614`。
科学主检验未支持且sham concern，见[结果](DL_R5_R2_RESULTS.md)；C不以该科学阳性为前提，
也不因此取得保护或方法性能支持。

封存helper SHA `6436ce13847f7da5aa96530bc6a03c5101056829807ad3320f255749a6b9736c`，
两遍完整库存和两次原parent/worker/shadow/CPU auditor退役核验；不解码NPZ/pickle或读科学字段。
server完整15.583822330秒/900秒，真实SSHexit0/16.172秒，0GPU；execution success=true/failure=null。
最终retirement SHA `e8b76009e759c1d22d6770498a121abb7b7c0f22c469b87f2503fc60db168595`；
只有确认真实退出和完整execution收据后才把final SHA传C，不用pending自证退出。
本机收据`analysis/outputs/dl-nm1-cal-preflight-8d63fff/retirement-277c180/`。

## 资源、一次启动与最新快照

固定sv3/物理GPU5，NVIDIA A100-SXM4-80GB，UUID
`GPU-79704918-e096-92b4-252c-e3244497a8b3`；一张独占卡，授权4–7不等于四卡并占。
CUDA_VISIBLE_DEVICES设该UUID、CUDA_DEVICE_ORDER=PCI_BUS_ID、JAX_PLATFORMS=cuda；
无环境/渲染，MUJOCO_GL=disable，MUJOCO_EGL_DEVICE_ID/PYOPENGL_PLATFORM均移除。
原CUDA/EGL成对接口不用于此compute-only阶段，实际CUDA绑定仍由C收据检查。

22:28:20–24启动前完整XML/pmon，物理4–7均无进程/各0MiB，原C checkout仍8d63且clean。
冻结external launcher版本DL-NM1-launcher-r1，SHA
`53edf01e00a99bfe1c56ad99b7376e37c9e678eeded0ec0a9e7df6709536a30b`。
22:28:41.740151一次启动parent467533/PGID=SID467533/start409689202，outer467517；
全入口硬限3600秒，outer3610秒仅用于失败捕获/清理，不增加成功预算。独立CPU全入口另900秒。
launch SHA `71df2eba2e729c2e9b7d1d15bbfa660b04a7b2099adbfb6637ad34a6591dd740`。

运行目录`/data/Policy_Discrepancy/runs/dl-nm1-cal-20261015-8d63fff/attempt01/`，
外部权威/日志`/data/Policy_Discrepancy/runs/dl-nm1-cal-inputs-20261015-8d63fff/`，
本机观测`analysis/outputs/dl-nm1-cal-8d63fff/`，现有handle94939；不因观察返回session而重启。
22:31:13–17原parent仍活跃、正在CPU导入/核验，物理4–7无可见GPU进程/各0MiB；
GPU5仍为本入口保留，不能把未初始化CUDA的间隔当作任务释放。4/6/7未由本次C使用。
资源快照`analysis/outputs/dl-nm1-cal-preflight-8d63fff/resource-20261015T143129909781/`。

当前C峰值显存、实际吞吐、完整资格与系数均未测定/验收；旧诊断约2.53GiB不构成较小卡兼容保证。
已完成R5实际2.580220占卡小时；C新增上限1占卡小时，另CPU封存/独审不占GPU。
T的6GPUh只是未采用的整阶段建议，不是每臂6h，不是成功ETA或全部研究剩余预算。

## 当前验收边界与下一步

原8d63有限工程测试和前驱完整性已通过；真实C首次尝试失败，尚未证明真实N1M干预有效、对照匹配或方法有效。
当前chat保留首次失败、完成下述修复和确定提交Linux验收后，在相同物理5/原3600秒/原科学配置下
使用新的attempt标识从头重跑；不恢复或补齐原partial。完整外层真实退出/终态/清理和完整128产物后才启动900秒CPU独审；
不可仅凭parent returncode0或中间部分批次宣布通过。失败保留新attempt证据，不改门或延时兜底。
独审分别报告生产真实位移与独立参考梯度，不能把后者叫生产内部逐叶bit-gradient；
只验证参考S/W/P权重释放，不冒称各臂真实学习位移。T采用前须结合C实测与完整source/native工作量
核算整体预算，并满足[T解释及对照资格条件](DL_NM1_T_READINESS.md)。goal保持active。

## 22:34:08首次尝试工程失败及必要修复

原8d63/attempt01外层actual SSH exit1/328.578秒；实际parent exit1/326.670301574秒，
external_failure=null表示监督器没有额外故障，不表示worker成功。supervisor worker_returncode=1、
failure=`C worker failed/incomplete`，generation_complete不存在，原parent与owned组已结束/清理。
22:35:06–11 fresh完整XML/pmon：物理4–7均无进程/0MiB，GPU5已释放。
旧handle94939现已终态，不再对它续跑或重复启动。

worker已完成replay导入和计划生成、初始化真实size50m CUDA Agent；在第一个B的postload完整身份门失败，
尚未执行init_train/首批collect_batch或128校准，不能将此工程失败写成方法阴性。
两个独立源码审查与纯CPU最小重现确认一个充分的表示错误：
官方`Agent.save()`已device_get所有params并将五个counters显式转为Python int；
`host(agent.save())`再次将这些int变为零维ndarray。`tree_digest`有意区分两种类型，
所以数值完全相等、参数子树及parameter_digest完全相同仍会破坏完整typed hash。
同样错误也在末尾原态fence，且零维counter随后不能直接写JSON。

必要修复标记`DL-NM1-state-serialization-r1`：仅在两处完整save身份边界保留官方树及原标量类型，
共享精确身份核验；不改FULL_STATE/PARAMS/COUNTERS/STEP、共享host/tree_digest、数据、NN、配置或科学门。
旧aggregate检查在第一个条件短路，真实参数哈希等剩余条件尚未被该失败证据证明；
修复后须仍逐项严格认证，若再失败则另存证据，不能以已定位这一错误声称其余身份都已通过。
旧142项有限测试未走真实load_variants入口，新增回归必须覆盖官方tiny save/load身份及拒绝counter类型/值、参数修改。
修复本机测试、确定提交clean Linux补验及新的实际版本/启动尚待完成。

原失败完整目录保留；小镜像`analysis/outputs/dl-nm1-cal-8d63fff/failed-attempt01/`。
outer terminal20556字节，SHA `f98157c6247fd7c31d8b13c425209de83b8259fbc848d38781873f8fee33f49f`；
supervisor SHA `4a3cbc689648938b00b81742ede3cb7902130a9c1b8d6984ad85ffa428f8daca`；
worker.log SHA `a6b47a27d7d747aaf3ca473f5c782efaff2b7bb14414f701f7b41ca36b0fd090`；
main.log SHA `0bf6745dc082d88025599451631c156909939b85fbf92c40758401cd6e89fcec`。
实际runtime_before未产生，不能冒称其完整CUDA运行后核验已通过；真实模型初始化日志仅证明到达该加载路径。

## 22:57:27修复版新尝试

`DL-NM1-state-serialization-r1`实际/测试Git
`47db6265311bc04dbdcc79439842eb8b6e707097`；fresh clean Linux24模块174项中172pass、
2原optional skips，unittest275.478秒、完整279.206237810秒/600秒，SSHexit0/279.812秒。
254份实际源码和11份norm前后匹配Git blobs、C producer/independent auditor源码绑定一致，owned均为空。
前01fb的单个测试隔离失败及其日志保留；没有修改生产labels的纯CPU要求。
这是有限工程通过，不是真实C成功或科研阳性；确切证据见[工程记录](DL_NM1_ENGINEERING.md)。

22:56:47–51启动前物理4–7完整XML/pmon均无进程/0MiB，新47db代码clean。
新外部launcher rawSHA `4e2f82ad95f085fee43a9040e375b735184323e790a43b5d45cd159471571128`，
固定同物理5/UUID、原3600秒/3610失败捕获、全部输入/科学配置不改。
额外绑定原8d失败terminal/supervisor/worker log三pins并核原parent/worker两个session均gone；
旧R5 retirement final仍沿原e8b760pin及成功277c审计。外层仅更新Git/路径/attempt标识并加旧失败核验，
retired/source/discover/cleanup等原函数AST不变；独立只读审查未见剩余确定阻断。

22:57:27.038901实际启动parent496921/PGID=SID496921/start409861732，outer496907。
实际run `/data/Policy_Discrepancy/runs/dl-nm1-cal-20261015-47db626/attempt02/`；
外部权威 `/data/Policy_Discrepancy/runs/dl-nm1-cal-inputs-20261015-47db626/`，文件名均attempt02。
launch rawSHA `3781dc7f14f4d2a9433a99e8abec47d3ddc24503fca08b42cc0e97405d76cf97`。
本机观测`analysis/outputs/dl-nm1-cal-repair-47db626/`，handle22265。
**新尝试运行中，未完整生成/独审，未证明真实N1M干预/资格或方法有效**；不再将旧94939视为活跃。
同时另有纯CPU512当前任务标签入口，不共享或增加GPU，实际闭包待其完整退出。

23:03:25 `runtime_before`认证实际CUDA只在物理5，initial三实例B/rho0/N的完整state4e0d、params cffc、
五个counter和初始carry均完全匹配原checkpoint。`initial.json`114785B rawSHA
`f601f895a555e293a0ff94a540a7978a2797813fd7a5bb368453ad8d2f4b581b`；
`runtime_before.json`2371B rawSHA `bd33cc886a7d5855543357580e15e517fe4c5aac3697aaf4c4704e38793ed0d6`。
这证明先前保存类型错误已在真实加载边界修复，不证明128批或方法有效。
23:11:57原parent/worker身份仍匹配，16个batch record，0个最终batch complete、无generation/终态，
保持原3600秒和未读中间科研值；物理5采样23:09:43–47仅本worker C/8766MiB，4/6/7空闲/0MiB。
纯CPU标签已独立完整验收，见[结果](DL_NORMAL_LABEL_RESULTS.md)，不替代C审计或新增GPU预算。

## 23:49:32第二次尝试工程失败

原handle22265已真实终态，不再作为活跃进程观察：actual47db parent exit1，
parent完整3125.098555551842秒/3600秒，worker exit1/3124.9671284328215秒；
root实际SSH exit1/3126.812秒，external_failure=null仅表示外部未另报故障，不是成功。
原parent496921/start409861732与worker496934/start409861843均已退出，owned_groups_cleaned=true。
23:50:49–53完整XML/pmon物理4–7均无可见作业/各0MiB，GPU5已释放；该快照不代表持续空闲。

worker日志连续记录batch_collected 0..127，之后首个control_arrays在PRNGKey(0)的隐式
host→device scalar传输触发原JAX全局transfer_guard=disallow，退出码1。
官方Agent setup启用此guard，而旧独立controls fixture没有构造Agent，漏测该状态。
这解释本次工程故障；没有generation_complete、128最终complete或final模型/cache/source封口，
**不能以128采集日志宣布完整校准、控制资格、机制或方法阴性**；本轮未读取frozen科学数值。

最小修复待验：仅允许control_arrays诊断控制数组作用域的host→device传输，
保持key/F32/JIT/公式/系数/科学门及NN路径，退出及异常均恢复原guard。
原attempt02保持不动，不从128中间记录补拼最终结果；修后新完整attempt03沿原128/3600秒。
这是原授权内工程修复，不增加训练、样本或更改科学协议。

原始小镜像`analysis/outputs/dl-nm1-cal-repair-47db626/failed-attempt02/`：

| 文件 | bytes | rawSHA256 |
| --- | ---: | --- |
| 外部terminal-attempt02.json | 21469 | `581f69c75edddb855bc4090f9f6fb2d9cc1650229543e655925ef937bf9564b5` |
| supervisor_terminal.json | 1246 | `f273f291811f25205726237a09f0374570f15ce616a49a17464d8f13cc1e9948` |
| worker.log | 12814 | `13bced8720371148307dccf1b5ee72d6a0f02830a3afa55bc27362b34daf15e1` |
| main-attempt02.log | 1885 | `790a0afacafddc3e12f2ea129fd4c3ec3274f21cdb3a96e39ca1467a295fcb04` |

同本机父目录真实SSH收据794B SHA `bac876bac788c750d6ceb99fdd6ff42b39625b4fdab9a20641f6874034abb811`，
SSHlog1384B SHA `8f06dffa2f46e766d1a301df8b93a72e99654efaf63f58a4511ea46e12183241`。
第二Agent独立核四份rawpins与128连续日志/源码guard设置，未读中间科学值、未重跑NN或服务器步骤。

## 2026-10-16 00:15:02第三次完整尝试

`DL-NM1-control-transfer-r1`实际/测试Git
`abefbb3f1440cd9c95000a5d7f49c0a560a9a550`已通过fresh clean Linux四模块41项/无skip，
整入口98.490317038秒/300秒、rootSSHexit0/99.204秒/0GPU；原key/F32/JIT/公式/NN及配置不变，
仅局部诊断host→device许可，全部guard反例通过。[工程证据](DL_NM1_ENGINEERING.md)与失败两attempt分别保留。

Agent按原授权执行新的完整128批/3600秒、同物理GPU5/UUID797，未续拼attempt02。
新code `/data/Policy_Discrepancy/dl-nm1-transfer-code-20261015-abefbb3`，
run `/data/Policy_Discrepancy/runs/dl-nm1-cal-20261015-abefbb3/attempt03/`，
inputs `/data/Policy_Discrepancy/runs/dl-nm1-cal-inputs-20261015-abefbb3/`。
外部launcher `7ed44d26b427cbff10f34e73328852c28ef53a7145b39acd49b8920b73773241`，
保留原15函数AST及3600/3610/原R5 retirement/8d失败核验，额外绑定47db失败原三pins和真实SSHexit1收据，
仅更新Git/code/run/inputs/attempt03。两个Agent分别核原函数/main差异及metadata反例，无新增阻断。
新inputs中previous-outer-execution.json为真实原794B/bac876收据，不自造或修改原失败图。

00:13:27–28物理4–7无GPU作业/各0MiB，两个实际代码目录clean，四个旧C原PID均gone。
00:15:02.539327实际parent538626/PGID538626/start410327282（outer538610）启动，
实际worker538639/PGID538639/start410327396，原3600秒不改；本机原handle85427持续观察。
launch60636B rawSHA `2edecf6a7ef7213f754074d8a6092fcac9c53ddbf5f8a2a05da56231d64c802e`，
resource1905B `c5a86ddfc6d132185dda68e2a815f8188ab701b105abde69e66163190ec18b38`。
本机metadata `analysis/outputs/dl-prevalidation-launch-20261016/launch-metadata-20261015T161932/`。
00:18:19原parent/worker均在/proc核实存活及原身份一致，worker正初始化/读入，0record、无initial/generation/终态；
未读中间科学数值。物理5即使暂无可见CUDA进程仍为此运行保留，不共享。
另正常reference0在物理4独立执行，不改本C数据/预算；完整校准、控制资格与方法结论仍待实际退出和独审。

00:21:45原C parent/worker仍在/proc核实存活且身份匹配，initial已存在、0record，
无generation/终态；未读中间科研值，保持同handle85427/原3600秒。
