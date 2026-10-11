# DL 有限路径MC诊断结果

**2026-10-11 证据范围更正**：本次沿用的 source/固定控制/真实 reward 均属于 `legacy_raw_suite`，绕过生产 Clip/Normalize。原 MC-inconclusive、epsilon/区间/hash/key/概率混合审计及停止门保留，不能裁决生产模型的预测技能。接口差异也不使 MC 变成阳性。见 [DL_ACTION_CONTRACT.md](DL_ACTION_CONTRACT.md)、RESEARCH/DL-14。

版本 `DL-predictive-MC-results-r1`，2026-10-10。**冻结MC-r1裁决为 `MC-inconclusive`，不采用为新gate，不进入短训练。** 原mode诊断未支持必要技能的结果保留；本次不确定也不能写成“边际奖励模型完全没有技能”或“后验保护无效”。形成可正式实验方法的整体目标仍未完成。

协议 [DL_PREDICTIVE_MC.md](DL_PREDICTIVE_MC.md)，实际执行提交 `4fb7e8947f71dce8faf96096e700ab2b1987f8e3`。这是Agent在用户总体授权内选定的有限排除诊断，不是用户逐项确认的新训练算法。仅300k同一训练seed0 checkpoint、原8校准episode/128位置、原六H10控制和已验证真实reward；初始及全部未来latent按实际unimix分布抽样，私有MC seed20261017、四独立block×32路径，先混合概率再求TwoHot CE。历史seed20261012、统计seed20261016、chunk8固定；零新physics/render/optimizer，不读blind数组或原物理snapshot。

## 实际执行与资源

sv3物理GPU4、UUID `GPU-9c070d5a-ac68-4c26-9522-a790ad874b23`、进程内CUDA0，启动前新零动作CUDA/EGL probe核验物理4。模型直接按冻结生产spaces构造，未调用环境factory。checkpoint/config字节哈希在加载前验证，实际加载参数digest及updates74493与源相等；末尾参数和全部counters不变。独立审计只核这些执行收据，没有另重演模型或checkpoint更新。

运行 `/data/Policy_Discrepancy/runs/dl-predictive-mc-20261010-4fb7e89/cp300000-attempt01/`，22:32:50–22:35:37北京时间、exit0，167秒；worker含加载/编译/评分164.365秒，1秒采样显存峰值2519MiB。2056个原始NPZ压缩后220,539,681字节（约210.32MiB），scores目录du约218MiB；未从未压缩体积推定实写成本。GPU5–7本项未使用。22:42:35 fresh完整query/pmon确认GPU4–7均0MiB且无本项目作业，GPU0/2未知作业未触碰；快照不代表持续空闲。

执行提交已推送并核验远端DL分支，未合入main。只部署sv3隔离worktree `/data/Policy_Discrepancy/dl-worktrees/4fb7e89/`，原repo保持干净d9be557；sv1/sv2不部署本诊断。结果归档之后的文档提交不是此次运行SHA。

## 数值与科学裁决

主对象是五条策略无关固定控制的H10平均边际log-score；六条中的原行为控制及H1/3仅描述。

| 必要方向 | pooled128均值 | episode bootstrap 95%区间 | 有限MC epsilon |
| --- | --- | --- | --- |
| clean相对prior | −0.00537181 | [−0.01545785, 0.00417213] | 0.00140737 |
| clean相对gray-body | −0.01277616 | [−0.04218397, 0.00899335] | 0.00125471 |

两个epsilon都超过事前0.001 nats稳定门；两个均值为负、区间均跨0，正方向裕量也未成立。epsilon由32/64/128嵌套差、两独立half差及四block的有限t指标最大值构成，**不是严格积分误差上界或新的正式显著性检验**。主裁决必须保留MC-inconclusive，不能称稳定负结论或通过。

原真实固定控制敏感性68点/8episode满足原校准16点/4episode覆盖条件，不能把失败笼统归因于没有控制效应。q/prior平均CE为1.146848/1.141476；即使平均CE较mode降低，也不等于后验新增控制信息或第二信号有效。背景a/b及控制敏感子集、逐位置half差保留为描述，不改主对象或挑子集通过。

128点原modal函数重演全部logits字节精确，最大差0；本次未观察到旧mode历史重建/重演差异，不外推其他硬件或完全积分了历史latent。实际MC初态仍条件于原一次历史抽样，未对整个历史做边际化。

## 独立证据及限制

sv3确定提交clean CPU5项15.831秒、pip check和真实来源校验通过；本机5项8.16秒。CPU用实际vision64小Agent验抽样/私有key/q=p/未来prefix/状态隔离/来源拒绝和统计规则，不替代size50m或科学结果。

独立审计器只用NumPy/stdlib，不import生产分析器、不加载模型/环境/GPU；流式核2048 rawchunk+8 aggregate全部哈希、8 source/modal来源、128位置及sample8×16规则。1,081,344个命名key用独立Threefry精确重建；由原logits先softmax混合概率，再独立TwoHot和episode等权bootstrap、epsilon/停止布尔值复算。最大概率重构差2.4869e-14，统计重构差9.7145e-17；科学门不变，无舍入边界歧义。独立审计37.483秒、passed=true，这是产物一致性验收，**不是方法有效**。

小证据 [dl_predictive_mc_independent_summary.json](dl_predictive_mc_independent_summary.json)；完整独立报告SHA256 `eed5199844ac35d61aeb7868f3b7e61adbf7ec97929b6ab092dab115216ecb0b`，原result SHA256 `28788c2e1303439f59245b73348b8bcac6ca281b2fe22ec284f41f2d71ea8ff8`。服务器原始chunk保留；本机ignored收据镜像 `analysis/outputs/dl-predictive-mc/4fb7e89/attempt01/`。不把同一校准集的数值复核当独立科学复现或多个训练seed。

**授权内Agent决定**：按冻结停止门结束本MC分支，不提高L/追加key/改H/控制/阈值，不运行1m、不生成新v或18条短训练。模型正确性、灰body关键线索泄漏、正常feedback下的真实观测增量、第二信号辨别和长期压缩收益仍无充分证据。

**首选下一研究建议，尚未冻结或执行**：先论证“一次当前图像省略，之后双方均接收各自正常观测”的真实闭环信息价值，而不再延长共同记录动作。相同t前carry/真实snapshot，q吸收当前RGB、p用同h真实prior；之后各自正常posterior与真实环境，使用事前共同私有latent/动作噪声，比较有符号真实return及恢复轨迹。该对象能检验局部当前观测是否有可利用增量，不能等同连续KL压缩或保证找到正例。

初步规模建议为原8校准episode、每条16位置中事前固定4个等间隔索引、每状态4 paired policy RNG、两分支100真动作（32状态、25,600动作，另计历史重建）；这些数值是建议，不是启动配置。先需明确定义一次omission、恢复/随机流/边界和停止条件并做语义验收；位置/RNG不是独立训练seed，统计单位优先episode，不能用32点替代旧128点方法验收。进一步方法验证须真正未见数据、与D/K/e独立构造的外部标签及有效对照。合法有限物理挑战是备选、存在OOD/视觉可辨识风险；raw状态KL可作廉价第二信号候选，尚未采用，且与D不统计独立。不得给vision64/propriofalse模型泄露真物理状态作为部署gate输入。

当前验证最少1张独占卡即可串行，2张可并行独立诊断/工程；既有sv3 GPU4–7足够。正式训练预算未冻结，旧20.4GPU小时外推不代表修订方法费用。
