# DL 冻结信号误差诊断

版本 **DL-reset-signal-error-r1**，2026-10-12。完整授权内Agent采用的CPU探索性诊断，不是新确认性实验或正式方法配置。前置为 [DL_RESET_JOINT_RESULTS.md](DL_RESET_JOINT_RESULTS.md) 联合诊断及独审已通过；其固定对象已停止，所有24留出已开启。

## 唯一对象与读取范围

只读取c030主目录中已由d987独审认证的 result.json、freeze.json、readouts.pkl、blind-numeric.pkl，以及独审audit.json；先核固定JSON SHA，再核两个派生pickle与freeze/result及audit双重绑定。开始/结束原字节不变。无原source NPZ/RGB/checkpoint/replay读取、模型调用、optimizer更新、physics或GPU；CPU硬限120秒、新输出目录，不覆盖旧产物。

pickle反序列化会载入包内既有cal和blind派生特征，包括其它模型的数值。本分析只科学使用模型0的D/K、F的S/U/choices、四冻结信号头的参数/原预测和八cal七控制回报；不使用h/z拟合、选状态或重新测量。导出上述数值白名单供独立算术核对，不能声称pickle只解了白名单字段。

cal S/U来自F的整episode LOEO，但λ由全八cal选取，不能称完全嵌套无偏crossfit；四信号头的cal预测为同八点训练内预测，不是第二层OOF验证。留出S使用全八cal F。比较时必须保留这个测量程序差别，不能把cal→留出差异单独归因于环境分布。

## 预先固定的输出，不拟合或调门

1. 全八cal和全24已开留出分别报告D/K/S/U范围、均值、总体方差；S与U的均值误差、MSE、协方差/相关。完整episode联报，固定δ=.2的数量只沿用原定义。
2. 四冻结头：`MSE=bias²+Var(g)+Var(U)−2Cov(g,U)`，说明均值误差与预测变化/方向，不将恒等式当根因证明。
3. 对每候选和D-only，`r=U−gD, v=g候选−gD`：`Δ=2mean(r v)−mean(v²)`；另拆均值与中心化部分。验证与原主增量一致，绝对1e−12内；不重新算CI/阈值、选λ或加basis。
4. D/K/S留出位于cal样本min/max外的计数，只作支持范围描述，不称OOD或不可泛化证明。报告冻结系数/rank/effective df，不改变它们。
5. 检查cal各LOEO的p控制是否与冻结全cal常数不同。仅用已有八cal七控制标签和固定q选择分解 `U原=U固定全cal常数参考+(Y常数−Y原p选择)`；全量保留、不得替换原训练target或重拟合。该恒等分解是发现参考行为差别的探索性检查，不是一个新信号或公允无偏验证。

独立Agent以math.fsum从原result检查留出分解，并对数值白名单核cal结果；不能复用producer aggregate就称独审。没有新训练重复、独立留出或额外作用证据。

## 可证伪解释与后续决策

若追加位移与真实残差反向，不支持“只是同向位移过大”；若方差项改善仍误差增加，不支持“只因预测方差增大”。冻结S系数方向与留出原S关系相反、训练/留出均值差或参考动作变化都只能定位失配现象，不能单独证明过拟合、表示容量不足、信号普遍有效或阈值调整有效。

该诊断结束后，依据实际根因可识别范围选择一个推进独立信号或压缩机制的真实对象。全窗口rep对照与连续actor效用属于候选，尚未采用；不能用又一项算术通过宣布方法可正式实验。普通replay、合法错误观测/干扰、选择性gate及学习性能证据仍需建立，正式M2协议不变。

代码 `analysis/dl_reset_signal_error.py`；运行目录 `/data/Policy_Discrepancy/runs/dl-reset-signal-error-20261012-<sha>/`；本机ignored镜像 `analysis/outputs/dl-reset-signal-error/`。实际版本、输入/输出hash、CPU时间、独立算术和结果解释追加本文件及RESEARCH/STATUS/验证日志。
