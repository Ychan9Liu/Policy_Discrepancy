# DL 正常任务分布读出定义

版本DL-normal-task-readout-r1，2026-10-15。**Agent在完整预验证授权内采用的有限分析定义**；
不是用户逐项确认的算法，也不采用任何学习组数、更新预算或正式实验协议。
两次独立源码/数学审查后采用下列固定特征与读出，不以运行中R5的D/e/Y或held结果选择。

## 要回答及不能回答的问题

在共同真实RGB/实际raw完整历史上，更新后的模型分布是否仍能被同一固定线性读出族用于预测独立任务状态量？
主标签为官方quadruped Physics的局部前向速度torso_velocity()[0]（m/s，不clip），upright为辅助。
二者是任务状态量，不是“正确动作”“应保护”真值；不以H1 head、自身e或actor效用定义这些标签。

每个模型从真正reset沿同一完整RGB/raw历史重新推断。当前q/p共享该模型自己的h；
不复用旧cached latent/旧carry，不喂更新模型自己新采样的raw来改变共同历史。
分别提取h、h+q_prob、h+p_prob并各自重新拟合：q/p概率由
`jax.nn.softmax(model.dyn._dist(logits).output.dist.logits, -1)`取得，包含官方unimix；
禁止用mode、单次sample、exp(logits)或CPU手写softmax替代该确定特征。
生产层须认证h4096、prob32×32及实际模型/原始dtype来源，取host原值后显式转float64作分析。

整个概率分布比actor的一次sample更丰富，读出不能称actor已经利用这些信息。
p本身是h的模型函数；h+p在线性读出中改善可能仅来自额外非线性基底，q−p不是条件互信息。
读出改善也不证明性能收益；完整更新Agent的独立native行为必须另测。

## 唯一固定拟合程序

实现为`analysis/dl_normal_readout.py`，纯NumPy、无文件/模型/环境/GPU调用或科研通过判定。
各模型、q/p分支均独立重拟合，不交叉移植旧head。每组h与prob分别使用**仅拟合集**均值中心化，
除以group RMS=`sqrt(mean_rows(sum_coordinates((x-mean)^2)))`；不用逐维std。
拟合RMS为零时该组在拟合与评估都映射为零，报告退化，不根据held方差重新激活。

固定ridge lambda=.1，最小化`mean_rows ||y-X beta-intercept||^2 + .1 ||beta||^2`，截距不罚。
目标保持原物理单位，两列矩阵解等价分别拟合两个任务量；不在held选lambda/特征/标签/阈值。
dual kernel为XX'/n，各非恒定特征组trace1；hq/hp最多trace2，惩罚斜率effective df<=20，
h-only最多10，包含截距的总df还需加1。联报实际RMS、完整kernel谱、数值rank/阈值、df和原单位绝对风险。
这控制特征维数带来的总能量，不保证所有信息可线性访问，也不消除任意坐标变换的读出限制。

## 独立单位与比较

实际参考来源、完整输入身份、fit/held隔离须由生产入口先认证，纯数学模块本身不认证来源。
若复用完整独审R5，沿其8fit/24held family与每family16个unit；
若因R5完整性失败而采用独立normal-reference，则按[其固定来源](DL_NORMAL_REFERENCE.md)每family14状态。
每family内各状态等权，先求family均值，再对24 held family等权；不把448/512帧当独立重复。
配对比较须同source/frame/order/target身份，仅counts相同不足以证明同状态。

固定10000次paired family bootstrap、seed20261040、线性2.5%/97.5%分位，
差值为reference risk−alternative risk，正值有利于alternative。所有对齐比较共用同一resample。
返回pointwise区间，不自动作多重性校正或方法裁决；区间条件于固定fit，未传播8个fit family重拟合不确定性，
也不估计训练seed间稳定性。真实科研的主次、合取判据和对照资格由另一机制协议事前冻结。

必须分报初态0、B/N及相应对照的绝对hq/hp/h风险与q相对p优势：
B相对0绝对q风险上升、N相对B保留且native方向相容，只支持该有限对象的保护相容链，不能识别唯一信息中介。
B未损坏而N更好称促进学习；p追上且q未降不是posterior损伤；旧head失效而refit恢复是坐标/利用解释。

## 有限工程证据与缺口

本机四项数学测试通过（.078秒）：dual对独立primal/SVD、重复坐标下group能量不变、零组、
family不等帧数的等权反例、差值/区间方向、相同mode不同概率及坏输入拒绝。
第二Agent只读核公式、官方概率API和统计方向，无阻断；这不是实际任务数据/模型读出或方法效果验收。
下一确定提交后Linux同项验收；真实特征/任务标签与native仍未执行，完整goal未完成。
