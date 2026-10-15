# DL-NM1 cal-only工程前置

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

**输入位置缺口，19:49:22服务器只读核查**：sv3上原replay绝对路径与库存路径均不存在；
真实普通源在sv2，不能因为路径字符串相同就假定sv3已有原对象。
需要从sv2在新sv3输入目录暂存完整原字节及原inventory raw pin，并独立全SHA核验，
保留原hostname/源路径与实际暂存路径。当前loader严格要求实际路径等原inventory路径，
尚未实现显式迁移绑定；不得改写已pin的inventory、猴补路径检查或在sv3伪造旧run来通过。
此准备不需要GPU，也不是方法失败；源迁移接口及完整cal runner均仍未实现。

先确定提交/clean Linux有限组合与真实Git blob绑定。随后实现完整C的固定128计划、
原1M只读取批、完整初态反事实、reference F32逐位置记录/F64系数与F32实际释放、
encoder/posterior原张量证据、分项资格和900秒CPU全审。未实现runner不得启动真实C。
R5权威终态与完整独审或失败封存完成后，fresh核物理5空闲，才按C独立3600秒上限串行执行。
失败/不足也是有限结果，不追加cal、降低原free-nats、放大rho或自动采用归一化e追阳性。
