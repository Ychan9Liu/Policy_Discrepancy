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

## 下一必要交付

先确定提交/clean Linux有限组合与真实Git blob绑定。随后实现完整C的固定128计划、
原1M只读取批、完整初态反事实、reference F32逐位置记录/F64系数与F32实际释放、
encoder/posterior原张量证据、分项资格和900秒CPU全审。未实现runner不得启动真实C。
R5权威终态与完整独审或失败封存完成后，fresh核物理5空闲，才按C独立3600秒上限串行执行。
失败/不足也是有限结果，不追加cal、降低原free-nats、放大rho或自动采用归一化e追阳性。
