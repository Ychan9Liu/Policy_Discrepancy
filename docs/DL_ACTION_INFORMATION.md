# DL 有限第一动作与接口信息诊断

版本 **`DL-action-information-r1`**，2026-10-11。**冻结对象已实现、完成CPU实测与独立数值／源特征审计，结果见 [DL_ACTION_INFORMATION_RESULTS.md](DL_ACTION_INFORMATION_RESULTS.md)。** 外部动作选择依据成立，接口增量未支持、actor首动作不确定，按事前门不推进gate或训练。它不修改原M2协议，不扩大或翻案已停止的H100平均反馈对象，不授权方法训练或读取24blind。原结果见 [DL_FEEDBACK_RESULTS.md](DL_FEEDBACK_RESULTS.md)，前置语义及即时/学习路径区分见 [DL_TASK_SEMANTICS.md](DL_TASK_SEMANTICS.md)。

## 1. 必要性、对象与边界

现有cal32同时有接近正立且达到任务速度的状态和torso方向接近反转的状态。不能据此将整个actor判为无能力。下一步需要独立于D/K/e与q−p正号的有限控制成本，区分任务对动作的要求、latent接口中的可学习性和现actor利用。低reward、动作敏感性、微小winner翻转都不能定义关键状态。

本对象只问：**固定首动作库和共同normalized-zero续行下，状态是否要求有代价区别的动作；实际q接口是否比p接口更容易学到这种选择；现有q/p首动作在同一有限续行下有何后果？** 它不测正常闭环控制、全局最优动作、完整RGB/历史信息量或持续KL压缩。

选择共同normalized-zero续行，是为了不把另一路记录行为动作中携带的信息或后续视觉补偿混入第一动作成本。代价是它偏离正常policy续行，须限制结论。替代的源记录续行H10建议未采用；不得按结果切换续行或延长H。

## 2. 冻结输入、控制与资源

- 只使用已验production feedback engineering-r2 `0d49c589b525ba0c26f9f85bf52c23573dc19507`，原八calibration episode及positions `[1,300,600,900]`。全部32点保留，pair0/q完整初state；先核公布plan/result/complete hash、全部8artifact hash和每位置八分支初state一致。不读blind。
- 控制timestep `.02s`，H20共`.4s`。官方quadruped actuator为filter、τ`.1s`；4τ只为覆盖主要activation尾部提供依据，不保证覆盖全部任务后果。XML SHA256 `c6b0ade5f3160b155bbe673aec3e0430eed3273ae13b60880855647e82bf8334`，运行时核实际model/actuator名称、顺序、τ及timestep。
- 七个事前raw float32 normalized首动作，固定顺序为zero、all+、all−、lift-LR+、lift-LR−、extend-diag+、extend-diag−；幅度`.25`。FL/FR/BR/BL各自yaw/lift/extend；下表按实际名字构造，未知或不同model失败。
- 另取既有四paired RNG的q/p首个实际raw动作，共八个；顺序pair0q/p至pair3q/p。核first proof/raw/key/latent/carry字节来源，不新增模型、RNG或render。
- 每条序列仅第一transition使用上述首动作，之后19transition均为raw float32 normalized zero。沿已验生产Clip/Unify/Normalize映射，applied F64独存；zero的lift实际为`.05`，不是零力矩。保留原完整activation/integration/task/RNG/counter/reset，不清零。
- 主量为真实suite F64 signed H20累计reward，F32编码、逐步discount/flags联存。无按存活长度归一化；真terminal计reward后stop并零吸收，不跨reset。普通clock均有20步额度；异常early time-limit失败。
- 主执行最多`32×15×20=9600`真实动作，stored-applied完整复放另最多9600；零Dreamer optimizer/model/render调用。主执行与重复核完整state/task/RNG/clock、F64/F32 reward字节和动作mapping，维持原restore/state容差，不用数值容差改变reward/标签。
- **sv3 CPU、GPU0张**；单次对象父进程180秒硬限覆盖载入、主physics、复放、readout、分析和写盘。外部Linux监督加内部monotonic，只清理自身进程。线程预算固定1（BLAS/OMP），依赖与实际SHA另记。新attempt独立目录；失败/超时不发布部分点裁决、不自动扩大预算。

| 控制 | 非零normalized位置；未列位置为0 |
| --- | --- |
| all+ / all− | 全12维 +.25 / −.25 |
| lift-LR+ | lift FL/BL +.25，FR/BR −.25 |
| lift-LR− | 上一向量取负 |
| extend-diag+ | extend FL/BR +.25，FR/BL −.25 |
| extend-diag− | 上一向量取负 |

## 3. 外部控制依据

记七控制的真实累计reward为`R(s,a)`，始终越大越好；文中的控制成本描述以此reward及相对regret表达，不混用最小化符号。实用margin固定`δ=.2`（H20平均每action `.01`）；这是Agent事前选择的推进标尺，不是理论常数或功效保证。

先联报全部32成本向量、range、排序及`Aδ(s)={a: max_b R(s,b)−R(s,a)<=δ}`，边界包含等号。探索性充分门要求：

1. `max R−min R>=δ`的状态覆盖至少四个episode；以及
2. 存在来自不同episode的两个状态，其`Aδ`集合不相交。

第二项排除“成本整体缩放却同一控制总近优”以及“最好两项仅差1e−9而winner编号改变”的反例。未通过只说明这个有限控制/首脉冲对象未提供充分状态选择依据，不证明所有cue或动作集合无用。`1e−10`只作展示排序及预测argmax tie规则，不能充当实用margin、不改变物理reward字节或δ集合判断。

## 4. 同容量接口readout

主特征为已存first proof中的实际`[h,z_q]`及`[h,z_p]`，四paired sampled latent全部使用；不使用raw logits gauge、不用概率均值代替actor实际接收的单次stoch。同h逐样本验证，特征shape/dtype/来源须与实际配置绑定。

该检验只比较固定读出族中的**接口可获取性**。q网络也可能将h非线性重编码；q读出胜p不严格证明当前RGB增加完整历史之外的信息。首轮没有encoder tokens，不能声称已单独核图像可获取性。RGB/因果cue、完整历史、OOD和模型支持仍为后续条件。

- 八个leave-one-episode-out fold，留出的episode全部四state/四RNG一起隔离；每state四训练行重复同一七维标签，总权重相同，不能把128行称128独立状态。
- 标签先减每state七控制mean；每fold再作train-column centering，预测加回train-column mean，保留不正则化截距。所有预处理只用该fold训练数据。
- h与z各组分别减train mean并除以`RMS=sqrt(mean_i ||group_i−train_mean||²)`。RMS严格0时该组中心化全0；不设置事后epsilon或改尺度。拒绝非有限；保留退化计数。
- 多输出linear ridge，`λ=1`，目标为`mean_i ||Y_i−X_i W||²+λ||W||²_F`，两接口同维、同目标、同正则；明确使用相应`nλ`的Gram系统，禁止另行挑λ或feature。
- heldout每sample选预测最高控制，`1e−10`内tie选最低控制索引；用真实`R(s,a)`评价，四sample均值→state均值→episode均值。联报真实finite-library regret `max_a R−R_selected`、预测误差和全部选择。
- constant参照为该fold训练状态平均reward最高的固定控制，同tie规则。主对比为q读出相对p、q读出相对constant的真实selected reward差。whole-episode bootstrap2000、seed20261021、linear分位数；两项lower95均>0且外部门通过，才记`interface_increment_exploratory_supported`。

LOEO训练集重叠；bootstrap固定cross-fitted episode差只有**校准名义描述**，不含拟合不确定性，不是独立blind验收或跨训练seed结论。任一upper95<=0记该接口对象不支持；其余未达到双正门记inconclusive。负结果只限此函数类、有限数据与四样本，不证明latent或图像绝无信息。完整连续效应与δ尺度联报，不把微小正均值写成实用方法收益。

## 5. actor与学习路径的分叉

八个actor首动作在相同续行下产生实际成本。按原四paired RNG联报q−p signed差、within-state散布、全部episode均值、名义bootstrap；不按某个RNG正号贴关键标签，不用其差定义外部门。与七控制差距只称有限参照，不称全局regret。支持/不支持/不确定按原lower>0、upper<=0、跨0的有符号规则；该对象不同于原H100正常反馈裁决。

一次完整对象后停止，不加H、控制、位置、RNG或checkpoint求通过：

- 外部选择依据不足：停止这个有限能力对象，保留控制库/续行不足反例；不自动换模型。
- 外部门成立、接口增量不支持或不确定：保留有限函数类/样本限制，当前不推进gate或训练。
- 接口增量支持、actor也支持：仅允许论证新的独立信号及合法cue验证，不自动启用blind。
- 接口增量支持、actor未支持：不否定后续学习路径；优先另立正常shared-loss的有限压缩机制规范，不能凭纯posterior-head KL下降或固定readout漂移宣布信息被删除。

未来机制须正常loss、相同数据/优化器/RNG起点、各更新模型重建history与重新拟合同容量readout；步数、rep强度、S/P/W/reward-only/OOD及后续actor学习目前未冻结或授权由本对象启动。完整方法目标保留。

## 6. 实现与验收入口

实现见 `scripts/dl_action_information.py`、`scripts/dl_action_information_physics.py`、`scripts/dl_action_information_stats.py`；测试见 `tests/test_dl_action_information_runner.py`、`tests/test_dl_action_information_physics.py`、`tests/test_dl_action_information_stats.py`。科学r1不变，工程初版分别为runner/physics/stats-r1。执行及clean Linux测试SHA `f1ae4d6513e81ed680992b64ba8ebade7d256d52`，实际完整结果、源码／配置／产物hash及复现命令见结果文档。

本机Python3.12.14/NumPy1.26.4、单线程CPU共29项测试，27通过、两项Linux真实process-group验证跳过，1.592秒。统计包括nλ、train-only/整episode隔离、sample选择次序、closedδ与微小winner反例；官方suite物理包括真实H20/完整activation与state、F64一ULP且同F32篡改拒绝、terminal/clock及任务恢复。独立审查发现restore会覆盖task，已在主执行和复放restore后重新核实际task/model，反例通过；没有放宽科学门或容差。BF16接口测试用size50m形状的合成数据，不执行Dreamer，也不是真实模型验收。以上测试发生在未提交的新文件上，不将其称已部署提交。

sv3 clean工作区29项全部通过3.062秒（两Linux进程组测试实际执行）、依赖检查通过。真实cal32/480record完整主9600＋复放9600步，父33.910822秒／180秒硬限、exit0／GPU0；独立核完成链／BF16特征与actor raw来源、成本／δ集合／selected reward／bootstrap／分类和eigh readout通过，最大算术差7.1054e−15。复放完整state最大误差0，不称新增独立physics。产物位于 `/data/Policy_Discrepancy/runs/dl-action-information-20261011-f1ae4d6/cp300000-attempt01/`；只运行该完整对象一次，没有扩展H／λ／库／位置／checkpoint。

下一步继续当前完整授权目标，先完成合法可见任务cue与完整历史之外增量的理论规范及必要性审查；不直接启动尚未定义的physical challenge、blind、gate或压缩训练。只读固定OOF截距／特征／constant代数解释已完成，不能以MSE或收缩几何更改停止门。完整方法目标未完成；按AGENTS报告实际执行SHA与最终文档SHA的差异及同步范围。
