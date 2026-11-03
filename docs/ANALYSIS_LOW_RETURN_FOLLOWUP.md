# 两条低回报运行的判断（2026-11-03）

针对用户询问pendulum_swingup-baseline与hopper_hop-PPD-KL是否训练明显有问题：**两条seed0都有明确的学习失败/未充分学会迹象；没有足够证据将其归为训练实现故障，不能用产物验收通过否定学习问题。**

## Pendulum baseline seed0

原始21点评价中，0至70万仅5万点有1.4，其余为0；75万187.1、80万165.8、85万0、90万0、95万251.1、100万90.5。AUC32.53、末段101.48，对照PPD-KL为691.47/787.80。它没有建立持续的倒立控制，不是几个评价点轻微波动。指定97.6万—99.2万训练episode的独立reward和为0，但单段视频不代表所有训练episode。

原30条收据/配置/评价审计无缺失，baseline实际权重为1；尚未像hopper那样完成该任务全训练replay reward、模型预测、优势/策略熵的专项复核。奖励入口窄可以解释长期低回报的困难，但“探索不足、奖励模型漏学、策略优化信号太弱或实现问题”尚不能裁决。不能从这条失败baseline的大差值直接宣传PPD-KL有普遍收益；也不能因低分擅自删掉该baseline。

来源：docs/ANALYSIS_DMC15_V1.md、docs/ANALYSIS_DMC_FOUR_GIFS.md及analysis/outputs/dmc15-analysis-20261102-final/learning_points.json。修正此前“前50万评价为0”的简写：5万点为1.4，其余至70万为0。

## Hopper PPD-KL seed0及补seed

seed0 PPD-KL AUC约0.86、末段1.01；baseline为253.83/316.75。先前全replay复核发现992个完整训练episode中799个零return，99.7026%的动作reward为0。已有证据支持早期持续正奖励未建立，模型想象奖励/优势长期弱且策略高噪声；尚非唯一因果解释。没有观察到先学会再崩溃。详见docs/ANALYSIS_HOPPER_M2_V1.md。

本次只读sv3已accepted的seed1/2/3六条原始evaluations，核对21点×10完整episode后重新算两指标。复用03既有run验收，不声称本次重新全量审计checkpoint/收据。仅回答是否重复近零失败，不并入原seed0正式总体，也不报告跨seed显著性。

|seed|baseline AUC|PPD-KL AUC|baseline末段|PPD-KL末段|
|---|---:|---:|---:|---:|
|0（历史批次）|253.83|0.86|316.75|1.01|
|1|265.17|323.30|316.26|457.71|
|2|257.07|247.71|386.68|291.81|
|3|245.75|191.45|331.94|272.37|

补seed三条PPD-KL均学到了持续非零回报，只有seed1主次均高于baseline，seed2/3均低于baseline。seed0的近零失败未在这三个补seed中重复，不能把seed0视频泛化为PPD-KL必然无法学习hopper，也不能据此删除seed0或宣告方法稳定。

seed0训练SHA48d71f1541721208322442f5a1d5977124f3e888；补seed训练SHA743089f40662b031a5cf2b6bf988decb52ba0b28，两批存在版本、设备/执行环境差异。旧mode诊断修复已有损失/梯度/RNG不变回归，但不能把跨批差异全部归因于seed。原输入微小EGL像素差异和潜在学习路径分叉限制保持。

## 下一步及交接

首选继续现有05对pendulum两组执行与hopper对称的只读专项：统计完整训练replay正奖励/零return、首次持续奖励时段；对照reward预测、imagined reward/advantage、策略熵及梯度有限性，定位失败环节。完成标准是区分原始经验、模型预测与策略优化的证据，并列不能裁决的根因。可立即做CPU分析；新实验/seed/干预由02和用户确认，03仅承接明确工程故障或授权实验。

本次证据在analysis/outputs/hopper-pendulum-followup-20261103/，含实际remote_source.py、各运行原路径/评价hash/21点及重算指标；只读sv3文件，不训练、不改配置，不访问GPU，不写W&B或同步服务器源码。无训练代码变更。分析起点SHA6008be7f15294ba1a27b6cf80b3184ac41f84e7d，实际读取脚本保存于产物，最终提交仅文档；版本/校验见DELIVERY.json。

可复制prompt：

```text
继续现有05，项目D:\program\PD\dmr3。交接版本见analysis/outputs/hopper-pendulum-followup-20261103/DELIVERY.json。
必读AGENTS.md、docs/ANALYSIS_LOW_RETURN_FOLLOWUP.md、docs/ANALYSIS_DMC15_V1.md、docs/ANALYSIS_HOPPER_M2_V1.md。
两条seed0属于学习失败迹象，尚未证明实现故障；hopper补seed三条PPD-KL均非近零失败，但2/3主次低于baseline。不要删失败seed，不混算新总体或宣称跨seed显著性。
下一任务仅只读CPU排查pendulum两组训练replay奖励覆盖、模型奖励/优势及策略熵，区分已验证失败环节与根因假设。不训练、不调参、不覆盖原产物。输出analysis/outputs新目录及专项报告，保存来源hash/版本/命令；验收原始reward计数与episode边界、日志对齐及限制，完成后报告版本并按证据交02/03，不自动发送消息。
```
