# DL-NM1 cal-only工程前置

## 2026-10-15：确定修复版clean Linux验收

actual/tested `47db6265311bc04dbdcc79439842eb8b6e707097`，新clean
`/data/Policy_Discrepancy/dl-nm1-repair-code-20261015-47db626`；24模块174项中172pass/2原optional skips，
无新增skip。unittest275.478秒、进程277.706325215秒、完整279.206237810秒/600秒，
真实SSHexit0/279.812秒、0GPU、owned空；254源码和11norm实际working bytes前后同Git blobs，
C source_binding/independent git_binding前后一致。原142测试未覆盖的save/load类型边界在新tiny真实路径拒绝反例中执行。
新labels末帧/真实task prefix与mock512隔离也通过，但尚无真实512数据或N1M校准科学结果。
第二Agent仅用本机镜像与真实Gitblob独立重核：R5清单254、NM1清单253、额外norm11，
并集264文件全部吻合；bundle/controller/helper、原日志和SSHexit链一致，未重跑测试或另扫服务器。
Linux实际alarm/reaped-descendant/symlink测试均执行；字节绑定不意味着所有NN图均被编译或真实N1M已验收。

证据 `/data/Policy_Discrepancy/runs/dl-nm1-repair-cpu-20261015-47db626/attempt01/`，
本机镜像`analysis/outputs/dl-nm1-repair-cpu-47db626/attempt01/`。
terminal SHA `347e3ca396d52202e3d8fd95e93b5796036c681b0d46efe0e4bd9ee214e56949`；
linux-tests.log SHA `84b3debc350aa627ce68ce2686ea57bed1003cd5bee6bf4fb66aa6ca0432ad55`；
source-binding SHA `6db3fd484b8a6844d4921e7a855e3602fa54390b699867989e2ebf3e0aef72a4`；
nm1-source-binding SHA `70e63e8c13ad7e0f6346120098edd98683799435e0d8d8668fe6418dd4ec372a`。
原01fb完整174组合实际exit1的单测试隔离失败保留，不能替代本次成功；生产标签限制不放宽。
真实C新attempt02实际版本同47db，见[实际执行](DL_NM1_CAL_RESULTS.md)；T未采用。

## 2026-10-15 22:34：首次真实C失败，完整save类型修复

原actual8d63首次真实C在postload typed hash拒绝，完整外层exit1/326.670301574秒，尚无首batch或校准。
原142项有限验收只覆盖tiny collect_batch，不覆盖完整真实load_variants身份路径。
两个独立审查与CPU最小重现确认：对已经host的官方Agent.save整树再次调用host将Python int counters
变为0维ndarray，破坏原精确typed hash；末尾原态fence同样受影响。
`DL-NM1-state-serialization-r1`仅修两处save边界，原完整state/params/counter门不放宽；
修后仍须核真实params等所有条件，不能由and第一条件失败推断其它条件已过。
失败证据/版本/资源释放及后续新attempt条件见[实际执行记录](DL_NM1_CAL_RESULTS.md)。

修复本机有限CPU：`python -m unittest tests.test_dl_nm1_cal -v`，10项9pass/1原Linux-session skip，
29.657秒/exit0；真实tiny官方save/load经过生产共用verified_state，原typed hash和JSON整数保留，
整树host反例、counter类型/值和参数改动均拒绝，原真实scratch/control/core tests保留。
实际working producer rawSHA `f27976afeb625ab58e6f273c30cb57b2574317f633025d567a800c4027753da9`，
test rawSHA `e4ea4236797fc1e280c80674fb9c5264984c6fb6ddca298563d2177cd29e6ad2`；
以上尚不是确定提交或真实N1M修复验收。下一新clean Linux整合24模块，采用600秒整体CPU工程上限，
依据旧20模块264秒及新增标签/身份回归工作量事前留出余量；不改变真实C3600秒/独审900秒。

## 2026-10-15：C 入口通过 CPU 验收，真实 C 未运行

`scripts/dl_nm1_cal.py` 与 `analysis/dl_nm1_cal_audit.py` 已实现，独立源码审查无剩余阻断，
actual/tested **8d63fff8b15a49c7e2e7bc84fd0f9c3481e6acb0** 的 clean Linux 验收通过；
尚未运行真实 N1M 的128批 C，不启动 T。
生产 stdlib 父进程监督单个 worker；原完整 B/B重复/rho0/N scratch、实际 context、
参考原 KL/梯度与真实更新、BF16原 bits、两遍固定系数/权重及全部批闭包分开保存。
独审在完整 raw 图/外层真实退出及原进程组结束认证后，独立恢复计划、65跨度、context、
种子/m/A、保留张量数值差、P Threefry 和 F64 资格，不重执行 NN/Adam。

审查发现并已修复两类工程阻断：原 PID-only group cleanup 改为原身份/start_ticks/SID/
全组归属核验；pickle 改为对实际解码的同一 bytes 先核已知 SHA。不能将登记表清空冒称进程已清空。
原 CUDA/EGL 成对资源接口不适用于无渲染 C，采用 C 自有 compute-only 认证，旧接口不改。
CUDA UUID/物理5/实际环境、fresh120秒、加载后真实 idle 和实际 runtime 独占核查保留；
MUJOCO_GL=disable，EGL环境键移除，EGL标为不适用而非已验证。

执行规范一句已由两个 Agent 同意在执行前澄清：S/W/P 是共同 N reference-forward 上的
实际 F32 权重收据，没有重跑其 NN 或证明其生产位移匹配。科学对象/门/预算不变。
完整剩余研究预算未定，一卡足够串行；只有 C 的3600秒采用，T 的6小时仍建议。
真实源迁移进度、e704 clean Linux114项与外部 source exit 证据见 [迁移记录](DL_REPLAY_STAGE.md)。

sv3 新隔离 `/data/Policy_Discrepancy/dl-nm1-cal-cpu-code-20261015-8d63fff`：
20模块142项，**140pass/2原optional skip**，无新增 skip；生产者与独审的真实 Linux
reaped-leader/descendant 清理测试均执行通过。unittest260.298秒，测试进程262.377387944秒，
完整入口264.391854106秒/300秒，真实 SSH exit0、0GPU/owned[]。
245源码与7规范前后真实 Git blobs 一致；原 R5 source_binding/audit source_manifest 一致，
另 C producer.source_binding 前后245项相同，C 独审 git_binding 原/审同8d63均通过，双方 owned{}。
另一个 Agent 对两套来源的并集251份实际 Git blobs、七norm及完整小型镜像独立重算全部一致，
未重新执行测试。Linux fixture 的 JAX/fork 警告保留于原日志且该 case 通过；实际父入口仅 stdlib。
这只是工程及有限数学验收：没有真实 N1M/cal/BF16 更新作用、独立信号增量或方法收益结论。

证据 `/data/Policy_Discrepancy/runs/dl-nm1-cal-cpu-20261015-8d63fff/attempt01/`，
本机 ignored `analysis/outputs/dl-nm1-cal-cpu-8d63fff/attempt01/`。
linux-tests.log SHA `e2108f62790d082e5f5d805ad8b21f32ba49d3a01d4e9cc990c4fea09d5c4cbd`，
nm1-source-binding.json SHA `0d8f76fe6bca6ae0f6a0db6875701c1ac9a5a7c4724b3f38cacc063da01e0f78`，
terminal.json SHA `efc7cdc146d8f97a5c0c0c9ec3bade5a7b87472650f3779bc92e582c2eb35879`。
bundle SHA `9cd817ee66fbf240cff4b8582e21c00b42edbb8b577ae6aa2adc1927c2baa746`，
controller SHA `e9f137158f8649b65cc3db051a921195f5737cd74eba85df4279ddd9f6d55dc2`。
C规范 actual blob SHA `0100d8f8c0df21669cfaed22c700aba364a8a4522efd9ee2e01427c7cc21f540`，
产物契约 SHA `e4ecad5dbe3fff263cf94d64fd2b4732419aa54c3cd2431a1faa61b0e0f4695c`。
后续文档提交不冒充该实际测试版本；只同步 sv3 新 CPU checkout，sv2 仅部署 e704 stage，
实际 R5 仍在原1584隔离目录运行。

## 2026-10-15：原 raw KL 出口与阴性资格算术

新增工程版本 DL-NM1-counterfactual-r2，源码提交
`98d92cdc40724ad0b87e229431a4c485bc934cbf`；不改变 C 科学定义/门/预算。
[产物与独审契约](DL_NM1_CAL_ARTIFACTS.md)明确完整128闭包、原始/实际计数、
BF16 uint16 原 bits、P 随机性独立复算和未校准 None 的语义。
完整生产入口与审计仍在实施中，本节不是实际 C 结果或新 GPU 启动。

`Agent.loss` 新增默认 False 的 `dl_reference`，只有独立参考 collector 显式开启：
从同一 RSSM loss 的原 `probe.rep_raw` 保存 K、其实际 F32 E、覆盖前 rep 和实际权重。
原 `_train` 不开启该参数，也不增加输出；原 `log_train_collect` 保持原调用。
一个独立只读源码审查确认默认 Python 分支不新增 RNG/状态/图，原 K 不从加权 floor loss 反推。
参考 JIT 新输出可能影响其融合，因此仍不把其梯度称生产 optimizer 内部 bit-gradient。

`analysis/dl_nm1_calibration.py` 使用存储的实际 F32 原值做 F64 乘积/全位置 fsum，
冻结 S/W 与实际 F32 P 边界，分别报告覆盖/同模块梯度及真实更新作用/对照剂量与置换效力。
零分母、F32 无实际保护或作用不足返回完整阴性，不填默认系数、不追加批次。
后续学习许可与方法有效性字段固定 False，C 门通过也不能自动采用 T。

本机 CPU 三模块19个不同测试全过：scratch3项30.900秒，normal-only6项与数学10项组合
24.857秒。真实 tiny F32 Agent 保留非零旧 moments，B 重复/rho0/公共训练全状态一致、
新 raw K/E/floor/实际 N 权重及只读边界通过；10个合成算术测试覆盖零分母、
解析正但 F32 无作用、不同路径不能拼作用、重复曝光不增覆盖、pooled P 比例和实际剂量失配。
一个测试首轮失败来自 fixture 的 deepcopy 保留重复对象别名；改为逐记录独立复制后通过，
未改科学公式或阈值。尚未进行新确定提交的 clean Linux 补证或真实 N1M 校准。

资源答复沿已采用规范：一张独占 A100 80GB 串行；C 3600秒最多1占卡小时，
当前 R5-r2 12600秒最多3.5占卡小时，都是各自入口硬限，非成功 ETA 或从现在剩余时间。
T 的6小时仍建议，完整研究总 GPU 小时未定；当前诊断约2.53GiB不能作为 C/学习显存保证。

2026-10-15，阶段科学版本DL-NM1-cal-r1，接口工程版本DL-NM1-counterfactual-r1。
Agent在完整预验证授权内采用[仅C阶段规范](DL_NM1_TRIAL.md)；不是用户逐项确认，
4096学习/6GPUh/后续性能门仍为建议。本记录阶段尚未运行真实N1M cal或新增GPU任务。

## 已实现接口及独立审查

- `dreamerv3/dt_latch.py`、`configs.yaml`：normal_only默认False且旧config缺键兼容，
  True只令干预A=原m且非当前is_first。原probe.m/v/f_D、H1有效性保留，
  另报m_intervention/v_intervention及dl/intervention_*；实际权重、释放校准、P层用A。
  reset的原rep及其它loss、free-nats1和全BT mean分母不变，不阻断reset原rep梯度。
- `scripts/dl_normal_replay.py`：新增read_batch，走官方assemble/annotation/Consec，
  不建立pending、不增加samples、不回写cache；普通batch→真实strict_update契约不变。
- `scripts/dl_nm1_counterfactual.py`：只读placement/显式counter seed，独立完整参考梯度JIT，
  以及原Agent._train核心的完整scratch状态单步。dona/allo逐叶device copy，不把原参数捐赠，
  不累计使用前次返回状态，不改Python计数/pending/cache；原完整状态末遍hash另核。
  参考梯度仅允许discarded normalizer scratch写入，禁止新叶/optimizer/target写。
  静态model/config改变会拒绝，B/rho0/N须各自构造正确编译入口，不能改已编译mode后复用。

生产核心保持原loss/梯度/AGC/RMS/momentum/slow-target链，不在其图内增加梯度输出；
独立参考梯度可能有不同BF16融合/舍入/抽样，不能叫生产optimizer内部逐叶bit梯度。
helper只提供接口，没有真实来源加载、128计划runner、设备授权、系数裁决或CPU独审入口。
还需实现这些部分，不能把本前置当C已经可执行或验收完成。

两个独立只读源码审查覆盖新mask及scratch/read_batch，未发现执行阻断。
审查指出endpoint参数cosine不能叫更新方向；已明确tensor_cosine，另实现
`neural_displacements(initial,B_after,N_after)`用F64实际差向量计算update_cosine。
ULP按参考F32 binade spacing归一化，不是整数浮点步数，跨binade可非整数；
零参考分别计数/报告nonzero-reference最大比值，不能用零附近巨大比值宣称科学强度。

## 本机有限证据及边界

normal-only新6项及原DT/logging组合共20项：18pass/2原有skip；整组19项67.257秒，
追加reset无效H1拒绝1项.556秒。skip为原2-device sharding及原opt-in full-agent snapshots，未关闭新门。
冻结旧overlay/calibration参考比较默认JAXPR（仅去Python callback对象地址）和全部数值，
真实categorical KL/stop-gradient/reset保留梯度及全BT mean、S/W/P/logging边界通过。
真实tiny active Agent缺键/False各两次完整更新，params/moments/norm/carry/seed/metrics字节一致。
初次图fixture漏原重复cal算子和callback地址的失败仅修fixture，不改科学数值或原检查门。

root scratch+replay第一组合8项全过11.575秒；后补rho0/N_nr分支后scratch两项全过30.405秒，
新增endpoint-vs-update方向反例单项通过。最终组合待确定提交的clean Linux验收。
真实tiny F32 vector/action2、B2/T4/context1、consec字段0，保留原全部loss/optimizer/free-nats1。
先一次公共更新形成非零moments/optstep1，再两次原共同初态scratch_B精确重复；原fullstate不变，
随后一次真实prepare/train/take的全部params/carry/outs/metrics逐字节等scratch_B，optstep2。
独立参考B/B梯度重复且非零；rho0参考/生产完整旁路等B；N_nr参考和生产有限且原state不变，
核实际reset原m可用而A不可保护。此tiny对象**不要求或证明N_nr有非零active作用/收益**。
静态配置误改拒绝通过；原同样的缓存source/输入哈希保持。

read_batch fixture跨3个chunk/完整65 span且含真实reset，重复读取原cache/原NPZ/samples/pending均不变；
不建立训练pending，因此直接回写被拒；有训练pending时也不能读取绕过，正常真实输出回写后新读取确见新context。
F32张量比较排除optimizer state；signed-zero字节差与数值非零分开。

以上没有真实N1M/BF16/16×65 cal验收、真实普通学习或方法结论。原R5实际代码仍1584，
运行中的sv3物理5隔离checkout不移动；新源码提交后仅部署新CPU验收隔离目录。

## 确定版本clean Linux证据

actual/tested **510339e990f0953445b3189eee4ad96febf7f89c**，新sv3 clean
`/data/Policy_Discrepancy/dl-nm1-cpu-code-20261015-510339e`。
16模块95项：93pass，2项原有optional skip（two-device sharding与FULL_AGENT_TEST opt-in），无新skip；
unittest215.083秒、测试进程217.137852906秒、完整入口218.324886414秒/300秒，exit0/0GPU/owned[]。
服务器终态19:53:31.942208；不同主机有钟差，不相减作耗时。
237源码真实Git blobs前后==父source_binding==audit.source_manifest，六份额外norm亦原blob一致；
C规范26950字节，SHA `611877a2f5d3f719abcbab10e22b3082dfcc7331b4e8a9b752281dc6163a16ad`。
第二Agent对本机镜像逐条重算真实blob，root再核小收据/日志raw SHA；不是重演真实N1M cal。

证据目录`/data/Policy_Discrepancy/runs/dl-nm1-cpu-20261015-510339e/attempt01/`；
本机ignored镜像`analysis/outputs/dl-nm1-cpu-510339e/attempt01/`。
linux-tests.log SHA `42438e9a7e22fc8c5fae364d9ccbafe876fd35ce2f45a977e67abe2860368ba2`，
source-binding.json SHA `324955539bbdfd08494bda93314f26a02a5bca4b5da141cf1ab62d373f35ba57`，
terminal.json SHA `8db61eb55a4fc9c75724a5f317c00131c73ed0070917c93f5fa124e7442e044e`。
原1584 GPU执行checkout未改，CPU测试未读N1M/R5科学值。后续报告提交只记文档，不能冒充此测试SHA。

**Git同步补证**：研究分支已push并ls-remote核验到b6f2ea9e0a233438225c6b5d09480ba25bd37915，
包括510339实际代码和仅文档结果；main仍595e982f1a3939775879728009e2105ee516e86d，未合并。
此前连接失败保留为历史，不再当当前未同步。后续文档收尾SHA另核，实际代码只部署sv3新CPU隔离目录，
没有更新正在运行的1584或canonical/sv1/sv2。

## cal应保留的前向差异与解释限制

独立源码审查后明确：C必须保存并比较B/N实际生产核心的完整replay outputs。
相同初态/keys下，active图也可能因BF16融合/舍入而改变pre-update h/stoch；
因此非零reference梯度与真实位移不能单独归为纯rep缩放。保持C_N已定数值资格不改，
但存在这些差异时，C先只证明该实现的operational总更新效应，更强的“仅rep压力”解释需定位或新隔离对照。
分报stoch离散变化、deter连续差值/ULP，不事后设容差宣布等价，也不自动否定候选。
反过来，replay entries相同未覆盖logits/probs、D/e、想象、actor/value/norm全部路径，不能当全forward一致证明。
这不是给所有未来操作性学习实验设置永久逐位门；T仍未采用，届时须明确estimand和必要对照。

## 下一必要交付

19:49:22发现的sv3原源路径缺口已通过[显式源迁移](DL_REPLAY_STAGE.md)解决，
完整C runner/900秒独审已在实际8d63代码实现并有限CPU验收（见本页前部），不再作为未实现项。
真实C仍须经严格cache解码/原schema和真实N1M运行验收；字节迁移和tiny测试不能代替。

C外部启动器已准备于本机ignored `analysis/outputs/dl-nm1-cal-preflight-8d63fff/launch.py`，
实际SHA256 `53edf01e00a99bfe1c56ad99b7376e37c9e678eeded0ec0a9e7df6709536a30b`，
独立静态复核/语法编译通过，尚未执行。它绑定实际8d63 clean源码、源raw pins、
3600/3610秒和 `DL-NM1-launcher-r1` 真实退出，不把外层3610当成功预算。
当前仅实现audited-complete前驱退役：必须绑定原R5成功内外终态、新独审report和
`DL-r5-audit-launcher-r1` CPU实际exit0/1800秒/owned全释放权威，不靠family间空闲放行。
失败封存路径保持拒绝，若实际发生须先审查完整封存schema，不猜测/拼接缺失对象。

R5权威终态与完整独审或失败封存完成后，fresh核物理5空闲，才按C独立3600秒上限串行执行。
失败/不足也是有限结果，不追加cal、降低原free-nats、放大rho或自动采用归一化e追阳性。
