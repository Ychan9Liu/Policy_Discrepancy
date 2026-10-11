# DL 正常 reset 信息验证

版本 **`DL-natural-reset-calibration-r1`**，2026-10-12。状态：授权内Agent已交叉审查并采用下面的**有限CPU物理校准对象**，工程实现中，尚未实际执行。它是下一条证据链的前置，不是新gate算法、方法有效或正式实验准备就绪。原B、A、H20/H100的对象与停止规则保留；正式M2不变。

## 研究问题与选择理由

要保护posterior，需要分别证明可利用的任务信息、第二来源的辨别能力、正常rep压缩的影响及选择性保护的效果。现有A只支持已见calibration上的有限接口读出；B在原MC门下不确定，不能建立这些因果联系。继续扩大旧readout或MC不会自动解决来源问题。

原始目标是q相对共同h/prior保留可学习决策信息；“相对完整真实历史不可预知”是更强命题，不是所有机制路径的必要前置。正常reset可同时提供一个更强且合法的检验对象：官方quadruped_walk初始化随机姿态，RSSM在is_first时清零旧deter/stoch/prevaction，再构建h/prior。模型encoder排除reward和flags、proprio=False，当前输入是实际RGB。**同h/prior和图像可获取性仍须实际验证，源码推导不能写成已通过事实**；训练后的prior不能假定uniform。

选择正常reset避免新增torque、teleport和任意改变物理参数，使用已有正确生产来源中的真实初态。reset结果只覆盖初始观测，不能外推中途事件、延迟cue、完整控制或训练性能。已有pos1成本差仅是邻近可行性线索，不能替代pos0实测。

## 本轮采用的物理对象

- 只用正确production来源 `f2d24da1831da2ffca22d313e7f534c96fc0d3eb` 的八calibration episode `[9,14,15,18,20,24,30,31]`，每条仅**第0帧正常reset**；原24 blind不打开数组、不作科学统计。
- 数据目录 `/data/Policy_Discrepancy/runs/dl-production-source-20261011-f2d24da/cp300000-attempt02/collection/`。源plan SHA `633efe28d551d5793ac8f0ed5f0e36e937e7f1af3c8edefc1d9f5fa278146fcc`，complete SHA `d155f577d19d2bbc051ec56b95f15e26afc774c130c3c9728a7fe599f295f3cf`；父目录源artifact/physics审计SHA分别 `8f93788ca212118139fab135036fd47ccbad79f04a34139b5dfa7d2dd26f7516`、`8ed3528fae18dc7a3043c767da87f873642a40a1a028ce86247f0521c3264722`。完整32文件只核原字节hash，科学读取仅八cal，禁止路径逃逸及未pin产物。
- 固定baseline seed0/300k模型身份只用于source归属核验，本CPU对象不加载checkpoint、不执行模型。原checkpoint/config/params pins、source终点退出链、原完整state/RNG/clock及action mapping工程证据须绑定；source文件及本轮相关源码前后不变。
- 采用 [DL_ACTION_INFORMATION.md](DL_ACTION_INFORMATION.md) 的原七normalized F32首控制及原顺序：zero、all+、all−、lift-LR+、lift-LR−、extend-diag+、extend-diag−；幅度`.25`，之后19步normalized zero。沿 `production_clip_unify_normalize_v1` 映射为实际F64控制；zero的lift applied为`.05`，不能称零力矩或清空activation。
- 原H20/`.02s`控制步、filter tau`.1s`、原XML/model/actuator顺序/上下界不变。F64累计reward越大越好，`δ=.2`不变；不能按reset结果加控制、延H、增幅、换checkpoint/seed或筛姿态。
- 八条先全部验证reset0的真实is_first、clock、完整task/RNG/activation及官方seed reset来源。先完成原记录raw→applied的0→20参考复放，共160步；F32reward/flags/raw-applied与source exact，完整integration沿原`1e-8`守卫并报告实际误差。source只存F32reward，**不声称有source F64 reward oracle**。
- 参考全部通过后执行 `8×7×20=1120`新主物理步，保存每步F64/F32 reward、discount/flags、raw/applied及完整状态；stored-applied再复放1120步。新主/复放F64reward、flags、完整integration/task/RNG/clock/reset/done字节exact。reset首20步意外terminal/time-limit或reset必须失败，不能把吸收padding当实际步。完整主对象最多**2400实际步**，没有render、JAX、actor、optimizer或GPU调用。

## 唯一有限判据与反例

令 `Y_i(a)=sum_{t=1}^{20} reward_f64(i,a,t)`，`regret_i(a)=max_b Y_i(b)-Y_i(a)`；联报全部八个成本向量、range、regret及closed集合 `Aδ(i)={a:regret_i(a)<=.2}`。集合边界包含等号，不加展示tie容差。

本校准仅检查一个存在性条件：至少一对**不同episode**的closed Aδ集合不相交，且两个状态的range均至少δ。它严格说明“七个首控制＋共同zero尾中，没有一个控制同时对这两个初态达到δ近优”。这不是总体覆盖、统计功效、RGB可辨、q可学、actor有益、高D关键或压缩机制验收；不能把一对状态当作所有五项目标通过。

全部28对均报告，不只展示成功对；另外联报 `min_a max(regret_i(a),regret_j(a))−δ` 描述边界余量，**不新增阈值或按它筛状态**。任一必要工程条件失败不发布科学complete；有限存在性不足则停止此对象并保留控制库/续行/自然初始化范围限制。门成立也先停止本校准，再审定下一完整模型/读出对象，不自动读取blind、启gate或更新Dreamer。

主要反例：状态reward整体变动但同一控制始终近优；cost range大但Aδ仍重叠；微小winner编号变化；仅改变速度而RGB不可辨；q响应错误、actor无利用、共同prior并非真实同字节；reset天然可辨却不代表正常中途；normalized-zero尾偏离实际policy。两信号数值大不能替代外部成本或修复这些反例。

## 实现、资源与独立验收

实现入口为 `scripts/dl_natural_reset.py`，测试 `tests/test_dl_natural_reset.py`。独立审计 `analysis/dl_natural_reset_audit.py` 与其测试另作源/数值/官方CPU物理复核；审计不复用producer统计作为oracle。原restore原语及同一官方模拟器的复用须明确，不声称独立模拟器。

本对象仅sv3/lyg0326 Linux CPU，**0 GPU**，BLAS/OMP单线程、CUDA不可见、MUJOCO_GL=disable。stdlib owned-parent全入口**180秒**硬限，覆盖参数/import/来源/Git helper/全部physics/统计/写盘/验链；只清理自身新建进程组。新attempt输出不覆盖旧证据。只有确定提交上的fixture/clean Linux验收通过，才一次执行完整校准；失败/超时保留证据，不扩大预算求通过。

独立审计另为CPU0GPU/180秒有限对象：全链/源码/source输入前后hash、完整退出/八episode覆盖/2400步收据；独立math.fsum复算所有H20累计值和28对，连续数值absolute≤`1e-10`，集合/整数/分类及余量零侧exact，不用容差改变δ成员。不同加法顺序若跨科学边界，审计失败而非放宽门。再独立执行1120步stored-applied官方CPU复放，与新主F64reward/flags/完整状态字节exact比较；不重演模型、renderer或生产policy。

运行产物服务器 `/data/Policy_Discrepancy/runs/dl-natural-reset-<date>-<sha>/` 独立attempt；本机ignored镜像 `analysis/outputs/dl-natural-reset/`。计划、实际SHA/config/依赖、源pins、reference、主/复放原始数值与状态、结果、完整退出及审计记录均保存；参数/replay/原始大数组不入Git。实际命令和时间/版本以运行收据为准，不以本稿基准 `06448d687526395cb231eff86b3c2af1a4550e97` 冒充执行SHA。

## 后续建议，尚未采用或执行

如果有限外部条件成立，另立完整reset模型/读出规范：实际同h/prior proof、跨全部reset共用四private latent root使p samples也相同、当前64×64 RGB可获取性、唯一训练/留出程序及有实用意义的真实选择增量。原封存24条是否一次启用须按新冻结规范明示；本对象不授权打开它们。原A两侧独立readout的阳性不能直接证明一个新共享F有效。

一个第二来源候选是共享、q/p合并等权训练且无branch-ID的action-centered有限控制读出F。候选 `S_decision=F_q(argmax F_q)-F_q(argmax F_p)` 不需在线真实反事实Y，解释为忽略posterior的预测有限库regret；S非负来自argmax，**不是q正确的证据**。独立真实U、同容量D-only/D+K/D+S及S-only、低值/有害覆盖、合法sky配对和state-additive reward不变性都仍待协议和证据。rawK可能在特定分布中提供预测增量，但同D/K可配不同真实控制后果，不能普遍定义应保护什么。

正常压缩机制须原full Agent.train/shared loss、同完整optimizer/moments/norm/counter及有限共同更新，先比较正常B与停止新增rep梯度R，再在独立规范中验证候选/强度/条件置换/第二信号-only。R保留历史momentum，不能称所有rep影响清零。各模型更新后重建history、重新拟合同容量读出，联报absolute q/p及q−p，排除prior赶上和坐标漂移；不能以KL下降或局部梯度恒等式代替信息损伤。B净改善但R更好也可支持有限rep抑制，不要求B必须绝对下降。

**reset训练接口陷阱**：原first65+replay_context1会把第0帧当context，loss只覆盖1–64，因此不能直接验reset rep。合法候选是真实前一episode terminal作context＋新reset0–63作loss，保留context1/reset mask/真实action-reward；terminal action占位未执行需显式记录。此拼接须工程实测，不能用dummy帧或context0偷换后声称原配置。本轮没有训练。

首选继续当前完整授权chat，完成本CPU对象及独审，再据实裁决下一规范。整个目标还缺第二来源独立辨别、正常压缩/保护、有效替代解释对照及后续学习/性能，不能因新增文档或有限物理门宣布可正式实验。若需其他chat接手由用户手动操作，不自动创建或发送消息。
