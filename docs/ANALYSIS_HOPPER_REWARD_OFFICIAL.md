# 05 · 从官方任务与奖励定义解释 Hopper 失败

日期：2026-10-20。状态：官方文献、dm_control **1.0.48**版本源码及sv2实际安装源码已核对；只分析，不修改任务奖励、冻结协议或训练实现，不启动训练/环境仿真。

## 最重要的结论

**hopper_hop的正奖励要求“身体达到高度门槛”和“向前运动”同时成立。没过高度门槛，哪怕在变好、在动，当前奖励仍可能一直为0。** 这比“学习得不好”更具体地指出了这次失败需要排查的物理条件，但没有证明是哪一个条件造成了全部零奖励，更没有单凭公式证明Dt为何单独失败。

官方论文将hopper描述为随机初始姿态的平面单腿系统；hop奖励同时取决于身体高度与向前速度。[官方论文第6页](https://arxiv.org/pdf/1801.00690)。本项目使用其纯视觉版本，位置/速度/touch没有直接作为agent输入，是已冻结的研究选择，不是发现配置误配。

## 官方奖励的准确含义

定义h为**躯干相对脚的高度差**，不是躯干离地高度；v为躯干向前的水平速度。将官方hopper.py和tolerance函数合并，可得：

```text
reward = 1[0.6 <= h <= 2.0] × clip(v / 2.0, 0, 1)
```

第一项来自不设margin的高度tolerance，越界立即为0；第二项来自速度阈值2、margin1、linear/value_at_margin0.5，表示在速度0到2之间连续增加。因此**不是必须速度达到2才有任何奖励**。[hopper版本源码](https://github.com/google-deepmind/dm_control/blob/1.0.48/dm_control/suite/hopper.py#L107-L122)、[tolerance版本源码](https://github.com/google-deepmind/dm_control/blob/1.0.48/dm_control/utils/rewards.py#L85-L124)。

| 身体比脚高h（米） | 向前速度v（米/秒） | 当步reward | 含义 |
| ---: | ---: | ---: | --- |
| 0.59 | 2.0 | 0 | 速度足够，高度不够仍无奖励 |
| 0.61 | 0 | 0 | 高度合格，站着不前进仍无奖励 |
| 0.61 | 0.1 | 0.05 | 慢速前进已有小奖励 |
| 0.61 | 1.0 | 0.5 | 速度收益连续增加 |
| 0.61 | 2.0 | 1 | 两项同时满足 |

高度0.59升到0.60时，在v=1的例子里reward会从0跳到0.5；低于门槛的许多改善动作获得同样的即时0奖励。向上跳得更高、动作更大、仅恢复站姿都不保证得到hop奖励。hop分支也没有单独的站立加分或努力加分；不能把另一任务hopper_stand的奖励当作hop自带的教学阶段。这里是**带硬高度门槛的连续速度奖励**，不把官方hop误标为纯0/1稀疏任务。

## 为什么不能照搬 quadruped 的直觉

官方quadruped walk/run使用连续的upright姿态奖励，乘以速度奖励。保持完全直立且速度为0时，速度项是0.5，因而总reward为0.5；hopper在高度合格但速度0时仍为0。quadruped的身体倾斜改善也通常能得到连续的反馈；hopper低于高度门槛时没有这种即时区分。[官方quadruped版本源码](https://github.com/google-deepmind/dm_control/blob/1.0.48/dm_control/suite/quadruped.py#L266-L324)。

这是两个任务奖励形状的具体差别，**不是仅凭公式宣布quadruped在所有条件下更简单**。它说明同一方法在walk中有收益，却可能在hopper的奖励入口处卡住，不能要求跨任务表现相同。

## 与这次原始数据如何连接

原始产物已验证：Dt真实训练动作99.7026%的reward为0，992个完整训练episode中799个零return；5万—10万动作只有52个正奖励步，baseline为9407。抽查原图像中Dt出现倒地/翻转。细节与输入hash见 `docs/ANALYSIS_HOPPER_M2_V1.md`，没有重算或更改正式评价结果。

**由官方公式作出的解释推断**：如果Dt长期没有同时获得合格高度和正向速度，大量0reward是官方函数的正常结果。倒地恢复过程可能在很多步里没有即时奖励；失败策略从数据中难以区分接近成功和远离成功的状态。原模型想象reward与优势信号弱，和这一现象一致，形成“迟迟没过奖励入口→奖励经验稀少→策略未改善”的解释。

这个解释比“奖励坏了”更符合目前证据：其他三组使用相同奖励能学会；但它**只解释零奖励为何会持续，不唯一解释为何本seed的Dt组进入该路径**。某些关键姿态/接触信息只在未来恢复/落地时有价值，而Dt探测当前actor响应，存在时间错位的可能；这是待验证方法假设，不能据此判定原gate误保护或漏保护。

倒地后也不自动以失败终止并重置。hopper没有任务特定的get_termination覆盖，默认环境以20秒/.02秒控制间隔产生1000动作截断；所以失败后可以继续经历许多零奖励步。[官方control环境](https://github.com/google-deepmind/dm_control/blob/1.0.48/dm_control/rl/control.py#L91-L117)、[官方hopper配置](https://github.com/google-deepmind/dm_control/blob/1.0.48/dm_control/suite/hopper.py#L30-L62)。这和本次所有完整episode长度1000、is_terminal总数0一致。不能把“倒地仍跑1000步”误诊为runner没终止，也不据此改为摔倒即reset。

## 还缺哪两个关键量

现在最需要的是把0reward分成：

1. **高度门槛没过**：h不在[0.6,2]。
2. **高度合格但没向前走**：h合格、v<=0。

正奖励只在h合格且v>0时出现。这次没有逐步physics height/speed，且输入只有图像，不能从reward=0本身唯一分解这两类原因；几帧倒地图像也不足以替代全程比例。最直接的下一项科学诊断是按训练阶段统计这两个条件、以及首次恢复到正奖励所需动作，并与D/w联合观察。具体采样、checkpoint、输入对齐与是否需要新增运行由02/用户制定；**不自行重跑、加站立奖励、改高度阈值、速度目标、alpha或任务**。

本轮读取到03已在实现/测试SHA `1ccde38c042d2264b5d5d1e6c93c658d170a9d88`完成mode诊断修复，文档交接为 `313532571f0a0e39759b9229264361c0fd9bd48f`。修复不改变原16条指标，也没有补出旧日志缺失的整向量mode或物理状态。当前首选交02制定奖励入口条件的辨别诊断，不重复要求03完成已完成的修复。

## 验证、版本与交付

- 官方源码固定为1.0.48标签，避免拿当前main替代实际版本；hopper/rewards/quadruped/control/base五文件的SHA256均与sv2实际安装一致。
- 仅抽取官方纯数值函数，在预设h/v/u网格上代入计算：hopper72组、quadruped32组全部与上述闭式公式一致。没有创建环境、采样动作、执行agent或进行训练；这是奖励公式核对，不是方法有效性或物理行为验收。
- 本机分析HEAD为 `313532571f0a0e39759b9229264361c0fd9bd48f`；具体离线核对脚本/源hash/网格/检查时刻/命令在 `analysis/outputs/hopper-official-reward-20261020/`。该脚本只在Git忽略的outputs中，脚本SHA256由verification.json保存；最终文档提交与分析HEAD只差本轮文档。最终SHA记本轮HANDOFF/回复。
- 本轮仅本机提交文档，未推送或同步服务器源码；没有训练源码或实验变更。原RESEARCH/STATUS未提交内容和docs/proposal保留，不混入本轮。只读访问sv2核对源码hash，没有核当前GPU占用，不作卡空闲/释放新声明。

可直接复制给现有02（最终交接SHA见HANDOFF/回复）：

```text
项目D:\program\PD\dmr3。交接取本轮最终SHA；本轮纯奖励分析HEAD313532571f0a0e39759b9229264361c0fd9bd48f。
必读AGENTS、STATUS、docs/ANALYSIS_HOPPER_REWARD_OFFICIAL.md、ANALYSIS_HOPPER_M2_V1.md、EXPERIMENTS与METHOD_SPEC。
已核官方1.0.48与sv2安装源一致：hopper reward=1[.6<=h_torso-foot<=2]*clip(v_forward/2,0,1)，quadruped直立静止已有.5奖励。Dt训练99.7026%动作零reward；当前无法分解高度没过与高度合格但速度非正的比例。
请制定奖励入口条件的辨别诊断：按阶段统计高度/速度两个条件、恢复到正reward的时间，并明确与D/w对齐。决定既有产物能否回答及必要新增证据；具体采样/checkpoint/seed/预算/资源待确认。mode修复已完成，不重复交03修复。
只读分析和协议文档维护；不改原16条指标、冻结v1、任务/奖励阈值/alpha/c，不启动训练或自行分配GPU。输出EXPERIMENTS/STATUS中明确的新诊断协议或缺项清单，完成按AGENTS给SHA、证据限制、下一步及prompt，不自动建chat或发消息。
```
