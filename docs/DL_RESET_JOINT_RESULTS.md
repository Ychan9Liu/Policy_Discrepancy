# DL reset 联合诊断结果与资源收据

更新：2026-10-12。冻结科学对象 [DL_RESET_JOINT_ASSAY.md](DL_RESET_JOINT_ASSAY.md) **DL-reset-joint-assay-r1** 已完成，并按既定规则停止。主执行、独立审计均验收完成；**整个预验证目标仍未完成，尚未形成可正式实验的方法**。本报告记录授权内 Agent 的有限诊断结果，不新增用户逐项认可的正式科研配置，不改正式 M2。

## 版本、来源与实际读取范围

- 主执行及 Linux 76 项全通过版本：`c030e26f688ed227aa43866bd68dcf4352a073e3`。
- 独立审计及 Linux 34 项全通过版本：`d987650c749b8917e2ea80a719d586232088f501`。相对主版本仅修改审计器、对应测试和验证日志；19 个主生产文件逐字节不变。
- 真实源版本：`f2d24da1831da2ffca22d313e7f534c96fc0d3eb`；源目录 `/data/Policy_Discrepancy/runs/dl-production-source-20261011-f2d24da/cp300000-attempt02/collection`。32 个源文件在开始、结束及独审时核原字节，无源文件改写。
- 模型 baseline seed0、300k checkpoint、size50m、RGB64/camera2/proprioFalse/action repeat1；任务 `quadruped_walk`。源 checkpoint SHA256 `2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9`；源 config SHA256 `11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445`。
- 固定 cal episode `[9,14,15,18,20,24,30,31]`，八个 episode 完整 1001 帧的 RGB/actions/flags/reward 用于末帧 context annotation 和学习；没有解 `before_carries`。各模型实际共同 h/prior、每 key 的 p sample 在全部32个 episode 间同字节；不同模型的 h/prior 允许变化。采用原 BF16 logits/samples，四个固定私有 keys，不消费原生产 Agent 随机流。
- 固定 cal、完整0/B/R状态、λ、F、信号回归器、D阈值全部写盘并由父入口封口后，才一次使用其余24个 episode。NPZ 的 image 字段会解完整1001帧，科学用途仅 reset0；没有使用其它 blind 时刻训练、选择或统计。这24个留出已开启，后续再次分析只能标为探索性。

运行目录 `/data/Policy_Discrepancy/runs/dl-reset-joint-20261012-c030e26/`。本机 ignored 小产物镜像 `analysis/outputs/dl-reset-joint/dl-reset-joint-20261012-c030e26/`；32原源、模型、回放和物理数组不提交 Git。

## 初始后验的有限信息增量

共享读出 F 以 h4096、z32×32 为输入，预测七个 action-centered H20 真实回报；q/p 等权、无分支标签。0 的八 cal 整 episode LOEO 在固定四值中选出唯一 λ0=.01，B/R 同容量重新拟合并复用该 λ。四次预测先平均，再 exact argmax；常数动作来自 cal，control ID5。

下表回报为同一七控制库的真实模拟器回报，不是 Dreamer 连续 actor 的任务评估。每项区间为固定模型、固定拟合和有限 keys 条件下，24 episode 等权、2000次整 episode 配对 bootstrap 的描述性95%区间；不是训练 seed 重复或总体功效证明。

| 模型 | q绝对回报均值 | p/常数均值 | q−p=q−常数均值 | 配对95%区间 | q/p闭δ失败数 |
| --- | ---: | ---: | ---: | --- | --- |
| 初始0 | 4.754023067 | 4.707622809 | +.046400259 | [.014348478, .080743645] | 2/4 |
| 正常B128 | 4.737006079 | 4.707622809 | +.029383270 | [−.001268155, .065092991] | 2/4 |
| 局部保护R128 | 4.736155492 | 4.707622809 | +.028532683 | [−.001856972, .064688447] | 2/4 |

δ=.2、regret>δ才计失败。三个模型的 p 在24个 episode 中均选择 control5，因此其真实效用与同一常数完全相同。本次两个q对比使用同一逐episode差值和同一区间，**不是两次独立重复证据**。初始0满足预设的 q−p、q−constant 两项正下界：**支持此共同 h/prior、固定任务及共享有限读出上的信息增量**。不能提升为所有观测、完整历史不可预测性、连续 actor 利用、正常 replay 或方法性能结论。

## 正常更新与局部保护

B/R各完成128次真实完整 `Agent.train` 更新；完整初始 optimizer/moments/norm/counters 保留，官方 int32 scalar `opt/step/value` 从74493到74621。batch16、loss64、context1；free nats1、dyn系数1、rep系数.1、reward_grad/repval_grad=True、ac_grads=False，原 M2/DtLatch overlay关闭。没有新设 alpha/c/shuffle gate。每次 init_train，固定八 cal 各重复两次；真实另一 cal 末帧作 context 接当前 reset0..63。它是**离线 episode 串接、reset 富集、重复数据的有限诊断**，不是普通随机 replay 或 runner 无缝续训。

B/R各2048个 reset rep 中2048个处于 post-free-nats active；reset excess sum分别为72736.67266845703、74537.0751953125。这建立了本对象确有 rep 梯度压力的前提，不能单凭它断言删除了任务信息。

R只停止 reset 位置 rep 的当前梯度，保留前向值、其它损失、非reset rep、完整旧 momentum。更新后用各自当前模型重新提取 q/p并重新拟合F，未用旧坐标读出替代。

| R−B指标 | 配对均值 | 描述性95%区间 | 有限解释 |
| --- | ---: | --- | --- |
| q真实回报 | −.000850587 | [−.006269759, .003717998] | 未建立局部保护收益，亦非等效性或稳定伤害证明 |
| p真实回报 | 0 | [0,0] | 本控制库内p选择未变 |
| q centered MSE | +.000116310 | [.000007016, .000215969] | R的有限读出预测误差更大，不等于连续策略表现更差 |
| p centered MSE | −.000006639 | [−.000010588, −.000002215] | 预测微改善没有转化为p控制效用变化 |

R−B真实选择差只来自 episode1（+.029743984）和28（−.050158075），其余22个为0。B/R相对0的q均值较低，但本对象没有预设B−0配对区间，不将此描述直接归因于rep压缩；这里p效用未提高，不能用“先验追上”解释真实差值减少。

**源码事实**：局部 R 没有冻结 posterior/encoder/carry；未来非reset rep可经RSSM递归和straight-through sample到reset状态，且各帧共享参数。未来 dyn、其它允许表示梯度的损失和旧 optimizer momentum也保留。`ac_grads=False` 停止 imagination 特征梯度，不能泛称 imagination actor/value直接更新encoder。**工作假设**：这些保留通路可能抵消局部干预；本对象未测量各通路的实际梯度或因果贡献，不能写成已经解释了阴性结果。

## 第二信号辨别

模型0上比较D-only、D+raw K、D+S、S-only，全部2slot线性架构上限、cal-only标准化、固定mean-ridge λ1；K为官方q.kl(p)原值，S为Fq在aq与ap之间的预测差值。实际cal设计rank依次为1、2、2、1，不能称有效自由度相同。原始逐样本结果、标准化、参数和有效自由度收据在运行产物中。

Δ=MSE(D-only)−MSE(候选)，正值表示增量。D-only MSE=.011065967。

| 候选 | 留出MSE | Δ均值 | 描述性95%区间 |
| --- | ---: | ---: | --- |
| D+K | .011501636 | −.000435668 | [−.000908282, −.000003426] |
| D+S | .013472877 | −.002406909 | [−.003409914, −.001394756] |
| S-only | .015776181 | −.004710213 | [−.007631644, −.001907678] |

**有限结果**：三候选均未改善此固定8cal线性程序的留出误差；不能据此断言K/S普遍无信息、所有非线性方法无效或条件独立不成立。

覆盖门预设每组至少4个U≥.2、4个U≤0，非正组另须至少2个aq≠ap且U<0。全24个只有2个实用正、8个非正（5个有限读出错误选择负例）；high-D阈值为cal中位数 .0490809865295887，high-D10个中为1个实用正、4个非正、2个有限读出错误选择负例。**全体及high-D均覆盖不足**。这些负例不等于已经验证原actor依赖了错误观测或nuisance。报告的 `eligible_covered_overall_increment` 是覆盖加全24区间，不是high-D条件区间，本对象没有后者。

全部S<δ=.2，最大 .118964645；固定S≥δ规则从未触发，零误报不能写成安全辨别。两个真实实用正例episode19、23均漏报，分别S=.066508690/U=.266378617、S=.073905754/U=.263278154。S来自argmax所以非负，仍有5个aq≠ap且U<0；不能把非负预测依赖当有用信息。**没有建立双信号辨别机制，未实现或验证线上选择性gate。**

### 全部留出选择，保留正、零及负例

以下只转录已审计结果，四舍五入展示不用于判据。p0/B/R和常数均为5；Bq/Rq列为控制ID。

| episode | q0 | S0 | U0 | Bq | Rq | R−B真实效用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2 | .034260877 | .017641020 | 5 | 5 | 0 |
| 1 | 4 | .018399928 | −.029743984 | 4 | 5 | .029743984 |
| 2 | 2 | .023984893 | .001700730 | 2 | 2 | 0 |
| 3 | 1 | .038508515 | .022165629 | 1 | 1 | 0 |
| 4 | 1 | .017583125 | .047628762 | 1 | 1 | 0 |
| 5 | 2 | .004132940 | −.023447280 | 2 | 2 | 0 |
| 6 | 2 | .046382024 | −.009668379 | 2 | 2 | 0 |
| 7 | 1 | .056325328 | .131173514 | 1 | 1 | 0 |
| 8 | 1 | .020893136 | .045061470 | 1 | 1 | 0 |
| 10 | 5 | 0 | 0 | 5 | 5 | 0 |
| 11 | 1 | .025846846 | .060988850 | 5 | 5 | 0 |
| 12 | 2 | .118964645 | .074192416 | 2 | 2 | 0 |
| 13 | 1 | .029305584 | −.115435167 | 1 | 1 | 0 |
| 16 | 4 | .049947238 | −.045175667 | 4 | 4 | 0 |
| 17 | 5 | 0 | 0 | 5 | 5 | 0 |
| 19 | 2 | .066508690 | .266378617 | 2 | 2 | 0 |
| 21 | 1 | .048819235 | .074451401 | 1 | 1 | 0 |
| 22 | 4 | .002580977 | .099525326 | 5 | 5 | 0 |
| 23 | 2 | .073905754 | .263278154 | 2 | 2 | 0 |
| 25 | 4 | .003290465 | .011119412 | 4 | 4 | 0 |
| 26 | 1 | .065246574 | .098270945 | 2 | 2 | 0 |
| 27 | 5 | 0 | 0 | 5 | 5 | 0 |
| 28 | 1 | .056729565 | .050158075 | 1 | 5 | −.050158075 |
| 29 | 2 | .006738978 | .073342366 | 5 | 5 | 0 |

## 验收依据、失败记录与独立性限制

主 `primary-attempt02` 九phase全部exit0、owned终止、父入口完整封口/退出/hash链；24个原seed fresh reset与原0→20参考480步、七控制主3360步和stored-applied复放3360步，共7200真实CPU physics transitions，无新render。八cal标签复用已完成自然reset审计。服务器Python3.11.16、JAX/jaxlib.4.33、Ninjax3.6.3、NumPy1.26.4、ml_dtypes.5.4、MuJoCo3.15.0、dm-control1.0.48；完整实际config、参数/state hash、环境、命令及设备证据在各phase收据中。

独立 `independent-audit03` 完整exit0/statuspassed、3360个stored-applied官方CPU复放，reward和完整state byte-exact；独立fsum累计回报最大差1.7763568394002505e−15，闭δ、U符号、选择、CI零侧均exact。独立重建35次共享头拟合、执行70次kernel solve，另验证四个固定线性信号头，132472数值比较最大差4.618527782440651e−14<原1e−10守卫，515 exact比较、78零侧比较通过，λ=.01一致。独立重建原sample predictions后，以已认证实际测量独立复算mean4/选择/S/U/MSE/配对bootstrap，没有将不同solver舍入替代主统计输入。另用fsum回报私有副本作边界敏感性复算，改变副本digest不冒充原产物hash认证。

独审复用同一官方模拟器/restore；没有独立重做fresh reset/480参考、模型前向或optimizer训练，没有解原NPZ/RGB/checkpoint。主链、原字节pins、typed数值精确提升、完整128更新收据是这些接口的认证依据；JSON不能独立恢复typed metrics digest，不能把记录认证称优化器独立重演。

保留两项工程失败：

1. `…/dl-reset-joint-20261012-348a5fc/primary-attempt01`：89.785934秒/exit1，baseline成功，B任何Agent.train前因人工key `opt/step`不符官方Variable叶`opt/step/value`拒绝；R/封口/blind未进入。0d83dc9仅修key并补真实tiny官方Optimizer及dtype/增量守卫测试；原科学预算/随机流不变，失败输出不拼接为成功。
2. 本c030目录 `independent-audit02`：.265011秒/exit1，在科学审计前拒绝外CLI schema。d987仅认证真实固定launcher字节、完整20argv/PID/正常terminal/3600秒/GPU UUID，未改主terminal、缺字段默认值、容差或科学结果。新attempt03通过，不重跑主对象。

主版本Linux76项全过，unittest日志内部54.734秒、含helper/pip的外计时55.857613秒、pip check0；审计修复版本Linux34项全过，unittest内部6.103秒、外计时7.187534秒、pip check0。tiny/fixture只验工程契约，真实任务的有限证据来自本主入口与独审；两者都不能验收最终方法效果。

## GPU资源：实际消耗与预算分开

实际只用 **sv3/lyg0326物理GPU4，一张独占A100-SXM4-80GB**；GPU5–7未用于本对象。UUID `GPU-9c070d5a-ac68-4c26-9522-a790ad874b23`，CUDA_VISIBLE_DEVICES使用该UUID、进程内cuda0、MUJOCO_EGL_DEVICE_ID=4，旧相同设备映射proof认证复用；本对象没有新renderer，不能称新EGL渲染验收。

| 口径 | 实测或硬限 | 含义 |
| --- | --- | --- |
| 成功主入口外CLI | 505.437680秒，8.424分钟 | 单卡占用的保守时间上界，约.140399GPU小时 |
| 成功父入口 | 503.609631秒 | 内部完整调度计时，与外CLI口径不同 |
| 六GPU worker合计 | 461.066102秒，约.128074GPU小时 | 包含加载/编译/学习/提取，不是GPU kernel活跃时长 |
| 两次主attempt外CLI合计 | 595.223615秒，约.165340GPU小时 | 成功加原工程失败；不是整个项目累计用量 |
| 独审03外CLI/父 | 12.483071/12.427242秒 | 0GPU，仅CPU |
| 本对象单attempt硬限 | 3600秒、≤1GPU小时 | 授权预算上限，非实测ETA；失败额外消耗如上单列 |
| 采样显存峰 | 6629MiB，约6.474GiB | 约2秒nvidia-smi采样，非瞬时最大值或最低足够显存保证 |

六GPU worker依次为baseline56.291129、B149.818778、R150.921420、eval0 34.662522、evalB34.661592、evalR34.710659秒；CPU freeze5.871482、truth20.240462、statistics1.615476秒。首次失败的2501MiB仅为加载峰，不能当训练显存。

以上GPU小时仅为本轮两次主attempt的新增占用/保守预留估计，不包含已复用的300k baseline训练、production32源采集、旧cal诊断或此前其它对象；不是整个科研工作的累计成本。CPU独审和Linux fixtures不新增GPU用量，不把历史来源成本记为零。

**安排建议**：复用现有checkpoint的当前规模诊断，一张独占A100串行已实测可完成；本对象可按单次≤1GPU小时预留。两张卡只在对象相互独立时缩短等待，无须同时占用4–7。新训练、正常replay、干扰/continuous actor/gate及正式多任务多seed的总GPU小时尚未冻结，不能由8.4分钟外推，也不能把6.47GiB作为其它模型、batch、分辨率的保证。

03:45:00.297989北京时间再次核查：4–7各0MiB，无compute/pmon作业；本次GPU4已释放、5–7未用。0–2仍有未知作业，未干扰；该快照不保证未来空闲。证据 `engineering/resource-end04.json`，此前03:33:53.466867的 `resource-end03.json` 保留。

## 文件、同步与证据pins

源产物入口：`primary-attempt02/result.json`、`freeze.json`、`supervisor_complete.json`；各phase complete/budget/log；`independent-audit03/audit.json`及supervisor；`engineering/primary-terminal02.json`、`audit-terminal03.json`、完整launcher命令、CPU验收、遥测和小镜像manifest。实际文件名/完整链以目录收据为准。

| 文件 | SHA256 |
| --- | --- |
| primary-attempt02/result.json | e8f56b834d85f94b22a293c96bb5c52923f23317f8eada3a914253302cfde037 |
| primary-attempt02/freeze.json | 3901903f41c8694bd9c9047e7764eb129706fb40473a2da8475fce3bb50e2afb |
| primary-attempt02/supervisor_complete.json | 12bed2fc2afb35e867289c7b13866f789491e8a5452f339910e7cb0a97f91edb |
| independent-audit03/audit.json | ba4c7d9c5d47baeb83f48eb2cde8eca5637186c1abe6ae42a961d13fe4fc8a22 |
| independent-audit03/supervisor_complete.json | 3496eadb986937e2d04c061863e5a065c08fca0d302b4a231b4c4fc4ba4e2732 |

执行版本仅同步sv3隔离clean detached checkouts `/data/Policy_Discrepancy/dl-worktrees/c030e26-reset-joint` 和 `d987650-reset-joint-audit`；未同步sv1/sv2。sv3 canonical仍clean `d9be5574878d2c3f117cc6667be844f9a527eb84`。本机canonical/main仍 `595e982f1a3939775879728009e2105ee516e86d`，原dirty RESEARCH/STATUS保持；研究分支未合main。03:19:14 GitHub push/ls-remote连接失败，不能确认远端收到；最终文档提交及补推状态以最终Git收据/回复为准。本文档归档不改变两个实际tested SHA或生产源码。

## 尚待决定及首选下一问题

**已验有限事实**是0的信息增量、局部R收益未建立、三第二信号程序无增量及覆盖不足；**工作假设**仍是任务有用观测可被正常压缩损害且可选择性保护；**无证据推测**包括把本结果归因于过拟合、线性容量不足、S只是标尺不准、旧momentum或future rep主导。不能把候选根因写成结论。

**Agent建议，尚未采用或执行**：首选继续现有完整授权chat，0GPU检查固定输出的误差来源，再讨论是否需要新因果对象。只用已审result的S/U/选择及四个predicted_U，分解固定动作对S−U，以及 `r=U−gD, v=g候选−gD` 下的 `ΔMSE=2mean(r·v)−mean(v²)`。训练侧比较须先声明仅读取新增已审readouts数值中的calOOF和冻结回归器；不重拟合、不调δ/λ、不新增特征/数据或重新称blind。完成标准是明确误差主要在哪个已测链条、哪些根因仍不可识别，并给一个预先固定的新对象及反例，而不是找到阳性阈值。

并行源码解释已核通路存在，但其数值和因果贡献未验证。一个后续建议是匹配完整状态/数据/keys/预算的全窗口rep当前梯度停止对照，与B及局部R重新提取并同容量拟合比较绝对q/p及q−constant；它不是部署方案，尚未采用/执行。已有24点再用只能探索；确认性结论需新冻结留出。即使有收益也不能直接拆分BPTT与参数共享，需要后续路径细分。

正式方法还需要普通replay时刻、合法nuisance/错误观测依赖、连续actor实际后果、选择性gate/匹配强度/conditional-shuffle对照和后续学习性能。证据不足时应保留路线局限或修改主张；本报告不授权扩大旧固定对象寻找阳性，也不改变正式M2协议。
