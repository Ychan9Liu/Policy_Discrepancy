# DL 固定控制预测诊断结果

版本 `DL-predictive-results-r1`，2026-10-10。**冻结mode路径的后验特异预测优势未获支持，不采用该多步reward代理。** 不据此判整个reward head无技能、后验没有关键信息或所有候选无效；形成可正式实验方法的目标仍未完成。

协议见 [DL_PREDICTIVE_DIAGNOSTIC.md](DL_PREDICTIVE_DIAGNOSTIC.md)。同一训练seed0的300k checkpoint，仅原8 calibration episode×16预定位置，共128点。六H10控制为原raw float32记录动作及五种策略无关固定控制，四q观测变体/共同p从同h开始mode rollout。没有新采集、blind数组读取、gate拟合或optimizer更新。

实际执行SHA `be9e32aa018d8b61c0b2931fe44f72a6f480d4b5`，科学版本DL-predictive-r1、输入契约修复DL-predictive-engineering-r2。sv3物理GPU4，UUID `GPU-9c070d5a-ac68-4c26-9522-a790ad874b23`，进程内CUDA0；probe核验EGL物理4，实际复用图像不render。配置、Python环境、源hash及device proof在run plan。

目录 `/data/Policy_Discrepancy/runs/dl-predictive-20261010-be9e32a/cp300000-attempt02/`，21:45:40–21:46:48北京时间，exit0；含初始化总68秒，plan后诊断38.517秒，1秒采样显存峰值2511MiB。8产物hash及prior变体精确一致、全部真实重复最大误差0、checkpoint状态未变。597a3a6 attempt01错误输入域检查exit1/2499MiB保留，不重写为通过。

本机ignored镜像 `analysis/outputs/dl-predictive/be9e32a/` 含8原始logits/真实reward数组及operational证据。根独立NumPy审计不导入生产分析器，以searchsorted重建TwoHot并逐episode重算bootstrap2000/seed20261016；CE/区间最大绝对差5.7602e-10。小证据 [dl_predictive_independent_summary.json](dl_predictive_independent_summary.json)。这是同校准集的数值审计，不是新科学复现。

| 原定主H10指标 | 均值 | episode bootstrap 95%区间 |
| --- | --- | --- |
| 固定五控制clean−prior log-score | −0.002496 | [−0.012202,0.007979] |
| 固定五控制clean相对gray-cue log-score | −0.030599 | [−0.083873,0.009507] |
| 记录行为控制clean−prior（描述） | 0.005359 | [−0.010022,0.023506] |
| 记录行为控制clean相对gray-cue（描述） | −0.001656 | [−0.017438,0.015664] |
| 真实固定控制累计return跨度 | 0.713652 | [0.177421,1.253567] |

固定控制CE_q/CE_p为1.169487/1.166991；真实跨度≥.1覆盖68/128位置、8/8episode，原16点/4episode校准敏感性必要条件满足。两主技能正下界未成立，`necessary_skill_supported_on_calibration=false`。背景a/b CE差为.013041/.009591、区间均跨0。H1/3和行为控制保留，不能挑局部改善替代主H10/fixed。

**Agent授权内决定**：按原停止规则，不启动同冻结对象1m，不采用多步mode代理、不生成新v/改alpha/rho、不运行18条短训练。不能把主技能未支持归结为完全没有控制效应。

**仍未排除**：mode与模型边际预测不同、未来latent mode误差、共同模型误差、gray-body保留姿态/影子、开环与正常接收未来观测的闭环差异。这些解释尚非已证实原因。

**独立Agent建议，尚未采用协议**：一次有限R-M校准排查，保持同128点/6控制/H10/真实reward，初始及未来latent按实际unimix采样，先混合TwoHot概率再评分，独立MC RNG及预算/可靠性/停止规则先冻结。不读blind、不直接训练；若数值稳定仍无必要技能，停止当前reward支持候选，真实key/closed-loop另立对象。不能长期禁止p接收未来观测制造差距。即使MC技能成立，原D仍是modal条件Gaussian sym-KL/actiondim，不能声称完整策略mixture/variance控制价值或正式方法通过。
