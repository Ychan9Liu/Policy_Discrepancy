# DL 有限控制预测技能诊断

版本 `DL-predictive-r1`，2026-10-10，**授权内Agent决定的诊断，未采用为训练方法**。起因见 [DL_OPPORTUNITY_RESULTS.md](DL_OPPORTUNITY_RESULTS.md)：已有持续控制收益，但H1保护近随机；剥夺损害子集上e和fD可同时增大。先判断奖励头是否能预测这些信息相关的后续效果，不能只换gate尺度求通过。

## 冻结问题与对象

问题：共享干预前历史、同一真实状态上，在不依赖当前actor选择的短控制下，clean posterior是否比prior预测后续奖励更准确；外观/纹理剥夺是否降低该优势？工作假设是H1局部奖励掩盖信息差异；反例是H10仍没有技能、模式世界模型误差迅速累积、cue剥夺偶然改善预测、只在行为动作下有优势或背景同样改善预测。预测技能不等于控制收益或posterior因果正确。

只读取H100新数据的**8 calibration episode**，每episode沿用16个H100预定位置（`sample_positions(frames,16,100)`），不读取blind episode数组、不拟合gate、不挑位置、不改已看H100对象。两个既有checkpoint/source/config hash保持；诊断历史seed20261012与原H100一致。这是校准诊断，不能宣称新的独立盲测。

每位置固定6条H10控制：实际记录的10动作序列；零动作10步；每维+.25；每维−.25；正负交替pattern；反pattern，各固定序列10步。后五条与actor/图像/D/e/K无关，均匀报告，不按收益挑控制。实际首动作加共同续行的旧H100标签与这些控制是不同对象。

**工程语义澄清（DL-predictive-engineering-r2）**：源采集直接保存`Agent.policy`返回、传入dm_control的raw Gaussian样本；actor mean经过tanh不表示sample被限制在[-1,1]。八条300k calibration源文件均hash核验，raw范围约[-5.10,5.33]、全部有限。最初入口错误新增“记录动作须在[-1,1]”检查，GPU4 attempt01因此exit1、尚无预测结果。修复保留原raw float32动作及现有精确复原条件，不裁剪/重写source；固定五控制仍±.25不变。CPU真实续行使用跨界raw控制另验编码精确和完整复原；这是输入契约修复，不改变科学对象或接受容差。

真实环境从原完整snapshot复原，各序列重复；逐步reward、终态/task/RNG/counter保持原1e-8要求。记录序列的10个reward须float32编码精确匹配原采集。未满足窗口/终止/复原即停止，不补齐/放宽。最大2阶段×8episode×16位置×6控制×10步×2重复=30720物理步，另计历史前向；不新增采集。

模型路径共享h，四观测变体q与共同p均取categorical mode，从各自初态按相同控制序列确定性core→prior→mode→reward向前10步，不读未来图像或奖励、不新增参数/抽样、所有checkpoint状态不变。未来奖励仅用于事后评分；不能流入t处预测。保存每变体/控制/步的q/p原始logits、bins、预测reward和真实float64序列。

## 评分与停止

主要量为H10平均TwoHot CE及有符号`mean(CE_p−CE_q)`；同时报告clean/cue配对的`CE_cue−CE_clean`。H1/3只作预定描述，不按最好结果选择H；区分记录行为控制和五条固定控制。另报每步绝对兼容KL、mean-CE构造的有界支持作为未采用描述性代理；不生成实际v，不校准S/P/W或改alpha/rho。

控制敏感性由真实五条固定控制的H10累计reward跨度报告；跨度≥.1至少16位置/4episode才有此校准必要证据。累计.1沿用原H10平均.01的效果尺度，16/128位置及4/8episode是事前校准诊断选择，不是正式功效保证。未满足时不把接近零的泄漏效应称充分排除动作泄漏；即使CE技能方向成立也不得判必要技能充分。模型预测的控制排序仅是有限集合、模式近似诊断，不证明全局最优或完整policy价值。背景变体真实state/reward相同，prediction变化仍需独立审计。

统计单位为episode，bootstrap2000/seed20261016。校准集方向和区间是探索证据，不是正式效果检验。若clean对prior、clean对cue在固定控制H10都没有正方向及区间支持，停止采用多步reward代理这条候选；若技能存在，只允许论证一个修订，随后先冻结新gate/尺度/对照/独立新数据/停止条件。不得把校准技能、增加数值幅度或H100结果直接称可训练/正式准备就绪。

本诊断不能修复gray-body保留姿态/影子的局限，也不能排除mode错误、共同模型错误或有限动作集合偏差。真正关键运动学干预及closed-loop价值须另立对象。

## 工程与资源

先验收CPU真实tiny模型：q=p对照、无seed的确定性前向、H1与原paired one_step匹配、未来reward不参与预测、输入/参数状态不变、轴顺序及非法控制/窗口拒绝；现有真实H10/H100复原证据复用物理helper，新增常量控制续行需CPU实测。源码只在本机提交，服务器独立clean worktree运行，实际SHA另记。

按已授权sv3 GPU4–7，先GPU4只做300k校准诊断，单卡/size50m/vision64/repeat1，硬预算3600秒；GPU7 logging审计独立。300k结果是否具备必要预测技能决定是否继续同冻结对象1m阶段，不事后调整控制/H或看blind。每次fresh占用及实际CUDA/EGL绑定，峰值/退出码留记录。结果在 `/data/Policy_Discrepancy/runs/dl-predictive-20261010-<sha>/`，小镜像ignored `analysis/outputs/dl-predictive/`。所有输出固定 `training_authorized=false`，不更新optimizer、不启动短训练。
