# DL reset 联合诊断

科学版本 **DL-reset-joint-assay-r1**，2026-10-12。授权内 Agent 交叉审查后采用的有限诊断设计，实施中；不是用户逐项确认的正式方法或已经获得的效果。原 A/B、自然 reset CPU 对象均已停止，本对象不追加或重判它们；正式 M2 不变。此次全目标仍包括独立信号、正常压缩、选择性保护和排除替代解释，不能用本有限读出代替整个目标。

## 问题、假设与证据边界

工作假设：当前正常 reset RGB 在实际共同 h/prior 下，可以通过同一个有限容量读出，提供真实控制选择增量；正常 rep 的新增梯度可能抑制这种可利用信息，停止这些新增梯度可能缓解它。物理前置已经支持八个校准初态中存在不能共用同一 δ 近优控制的一对，但 RGB、q 信息和以上作用尚无证据。

独立第二来源的候选为 raw K 和诊断读出的 S。K 只是更新幅度，不能普遍判断任务效用；S 的非负来自 argmax，错误 q 也可能有高 S。对两者只能用独立真实控制回报验证。七控制读出是有真实模拟器标签的离线探针，**不能称标准 Dreamer 内部免费信号、连续 actor 已经受益或可全时刻部署的方法**。此对象不新增线上 gate。

反例包括：RGB 虽可辨却 q 不保留可利用信息；q/p 均无法胜过简单常数动作；先验追上而 q−p 减小；旧坐标读出失效但重新拟合恢复；停止 rep 新梯度只改变 optimizer 轨迹而没有更高真实效用；S 高却真实 U≤0；高 D 条件缺少正/负覆盖。负结果只能否定这个有限容量/预算/任务对象，不代表所有图像或非线性方法无信息。

## 唯一来源与封口顺序

来源为正确 production `f2d24da1831da2ffca22d313e7f534c96fc0d3eb`，目录 `/data/Policy_Discrepancy/runs/dl-production-source-20261011-f2d24da/cp300000-attempt02/collection`。沿用 [DL_NATURAL_RESET.md](DL_NATURAL_RESET.md) 的完整 plan/complete、两个来源审计、原退出链、32个原始文件 hash 和 baseline checkpoint/config/params pins。所有32文件先核原字节，再解八 cal；源 JSON 和所有原字节在终点再次核验。

校准 episode 固定 `[9,14,15,18,20,24,30,31]`，模型 baseline seed0/300k、size50m、RGB64/camera2/propriofalse/repeat1；原始 flags、reward 和 raw previous action，不解 `before_carries`。八 cal 为学习和末帧真实 context 重建，允许解其完整1001帧 RGB/action/flags/reward，必须报告这个读取范围，不能声称只解了 RGB0。

原24 blind 在以下内容全部写盘、hash 绑定前不解任何数组：科学计划；模型0/B128/R128实际完整 state；共同四 keys；唯一 λ0；三个 F 的预处理/参数；常数动作；校准 OOF 信号及四个信号回归器；D 中位数；统计及覆盖门。之后一次全部24联评，模型选择、控制、样本、λ、更新次数、阈值均不再改变。blind 源如用 np.load 解 image 字段将解完整1001帧，须如实记录，科学用途只有 reset0；不使用其它帧选择、学习或统计。

## 外部标签与实际模型接口

真实七控制、顺序、F32首控制幅度 .25、19步 normalized-zero、H20、原 production 映射及 δ=.2 完全沿用。normalized zero 的 lift applied=.05，不能称零力矩。八 cal 复用 `aa9fb7f777a28e8a82d57c31f9e749b5912af316` 下已完整审计的 F64 reward 标签及完整退出/hash链。

新24 blind 各执行官方原 seed fresh reset、原0→20参考（480步）、七控制主（3360步）和 stored-applied 复放（3360步）。源仅 F32 reward exact、完整 integration 原1e−8；新 F64 reward/flags/完整状态复放 byte-exact。无 early terminal、reset、time-limit 或 padding；不渲染、不改变 XML/动力学/seed。新物理合计7200实际 transition，不复用校准上的一对作为留出。

对每个模型，用官方 init_policy、encoder、dyn.observe、prior/dist 的只读私有函数处理合法 cached RGB0。四 roots 为 seed `20261026`、固定 tag `0x52535231` 与 sample ID；不含 episode、模型或 q/p 分支。q/p 使用同一实际 latent sample key。跨该模型全部32 episode 的 h、prior logits、每 key p sample 必须字节相同；四 h 也须相同。0/B/R 之间 h/prior 可以改变。保存原 BF16 logits/latent，不替换为 mean 或 uniform。mode Dt 按现行代表性策略对称 KL，raw K 按官方 q.kl(p)，和 sampled 接口分开记录。所有只读函数拒绝 attempted create/modify，全 state/counters 前后不变；不读陈旧 policy copy、不声称新 q sample 等于原源 q sample。

## 正常 B 与 reset 保护 R

固定 **128个完整 Agent.train 更新/臂**，只比较0、B128和R128，不设看结果后追加的更新档。模型、完整 optimizer/moments/norm、损失系数、free nats、梯度开关、slow value 等来自同一 hash-authenticated checkpoint；使用标准 Agent.load，实际 n_batches 恢复为 checkpoint updates，不能声称恢复原 prefetched batches。每臂记录实际 counters/seed及完整 state hash。

batch16、loss length64、replay_context1保持原设置。每批每个 cal episode 出现两次，输入为真实另一 cal episode 的 time-limit 末帧作 context，再接该 episode reset0..63；terminal action 的源占位0未执行，随 reset 屏蔽，明确标注。源 episode 是独立环境采集，拼接是明确的 **离线 episode 串接、reset 富集诊断**，不是原 runner 的真实邻接或自然随机 replay 分布。每次从 init_train 开始；不声称环境/replay 无缝恢复。固定八个 episode 和重复批次可产生过拟合，不能外推正式训练。

context 使用模型0对八个真实完整1001帧 source obs/raw prevaction、合法 flags、固定独立 private key 重建的末帧 replay entries，0/B/R 共用这份冻结 annotation。这是新构建的真实观测 annotation、标准 stale replay context，不冒充 source 原缓存 carry。构造训练批时沿 Replay 的采样前缀规则仅把 context 列 is_first 标为 True，并明确记录为 annotation；原 source bytes 不变，下一真实reset仍True、其余loss位置保留原 flags。terminal BF16 entries精确提升到实际replay F32 spaces并核往返字节；后64列不用的replay entry slots保持标准zeros模板，只有真实context列用于truncate，不当作实际已缓存的64个state。context 参数因下一 reset 被屏蔽，不改变 context 为0，也不补 dummy 帧；工程验收须核实际 reset 在 loss 第0位、实际 old carry/prevaction 屏蔽及真实 reward/action alignment。

B 使用原完整 loss/optimizer。两臂的进程内 RSSM.loss 包装只增加 reset_count、post-free-nats rep>1 的 active_count和 reset excess_sum 三个只读指标；这些指标不能修改 loss/state/梯度。R 另外仅在构造 Agent/JIT 前把 reset 位置 rep 换成 stop_gradient(rep)：前向值、free nats、dyn/其余 loss和非 reset rep 不变。历史 rep momentum 保留，不称所有 rep 影响清零。不修改默认 Dreamer 或正式 M2 路径。两臂同数据/prepare seed counter/更新预算，真实参数随干预分叉后中间随机输出可以不同。记录每次 update/真实 train metrics，更新后用当前模型重新提取 reset 表征、重新拟合同容量 F，避免旧坐标探针伪伤害。

## 共享 F、真实选择及统计

每模型64 cal rows=8episode×q/p×4sample，共用同一 F、intercept 和预处理，无 branch/episode/sample ID、D/K、物理状态输入。输入 h4096+flatten(z32×32)，输出七 action-centered 回报；数值上先 `d=Y-Y[:,0:1]`，再 `d-mean_a(d)`，等价于 `Y-mean_a(Y)`。此先减共同anchor的实现避免整体offset引起中心化近tie漂移；只在输入pairwise差未被offset舍入改变时保证相应不变，不能声称任意大offset的F64都精确。F64 pooled cal mean、h/z各自总 RMS、零 RMS 置零；ridge mean objective、无正则 intercept，不截特征/特征值。复用已验 numerical primitive，但新选择程序另测。

初始0在全部八 cal 上 whole-episode LOEO，固定 λ `(1,.1,.01,.001)`，按等权 q/p 的真实选择 utility 最大选取；全局1e−10 tie内按 pooled MSE最小、再最大λ。得到唯一 λ0，0/B/R分别重新预处理、拟合，全部用 λ0。常数动作来自八 cal 的平均真实 Y，三模型共同使用。

唯一动作程序：每 branch 的四个 F 预测先 F64平均，再精确 argmax；exact tie最低 control ID。**不使用先每 sample argmax 再平均效用的旧程序**。`U=Y(aq)-Y(ap)`，另报 `Y(aq)-Y(a_const)`；`S=Fq(aq)-Fq(ap)` 不需要线上真实 Y，但 S≥0 不证明正确。24 episode 等权，2000次 whole-episode paired bootstrap、seed20261026、共同draws、linear percentile95。四 keys 是有限采样对象，不算独立 episode。

模型0的 q−p 和 q−constant 两项 lower95均>0才支持此有限读出信息增量；跨零为不确定，任一稳定负为本对象不支持。全部 centered MSE、absolute q/p/constant utility、regret、closed δ近优失败率、每episode选择及 U正/零/负完整报告。B/R只用主128点，同样联报 absolute q、p、q−p和 R−B 的 paired差；p改善导致差变小不能写 q受损。R−B更高也只支持这一 reset富集预算下的有限新增rep抑制，不能证明全时刻正常训练保护有效。

## 第二来源的有限比较与不能通过的条件

仅模型0为信号辨别主对象。D-only `[D,0]`、D+K `[D,K]`、D+S `[D,S]`、S-only `[S,0]`，相同2slot线性架构上限、cal-only各slot mean/RMS、零RMS置零、固定 mean-ridge λ1、intercept无正则；不加basis、不选阈值/λ。实际 rank/effective df必须报告，不称同有效自由度或普遍条件独立。cal信号S和目标U由固定 λ0 的 LOEO F产生，blind使用全cal F。校准target退化必须报告。

联报 blind squared error/MSE及相对D-only的 paired增量和95区间。raw K 保留官方原值，沿数值守卫≥−1e−5，不截零；低于此守卫停止对象。raw K只有在本程序上有留出增量才得到有限支持；本对象无法证明任意非线性D条件独立或 nuisance/OOD辨别。high-D阈值为八cal D中位数，预先冻结。

最低外部覆盖：24 blind 至少4个 U≥.2 和4个 U≤0 的不同episode；非正组至少2个 aq≠ap且U<0。谈 high-D辨别还须 high-D 内分别至少4个实用正/非正，保留上述非恒等负例。不足判 coverage不足而非改门、补位置或筛点；这是有限可检最低，不是充分功效保证。若 S/指标退化、coverage不足或对D无稳定增量，不能宣布双信号机制成立。

报告字段 `eligible_covered_overall_increment` 仅表示覆盖门满足且全部24 episode 的相对 D-only 增量区间下界为正；不能解释成高 D 子集内的区间优势。`high_D_mean_increment` 仅为描述性均值，本对象没有预设高 D 条件区间判据。

## 资源、执行与后续

首选 sv3/lyg0326 一张独占 A100，在已分配物理4–7范围内；优先物理4，新检查占用/UUID，不分享、不干扰未知任务。各 GPU worker 顺序运行，CUDA/EGL绑定完整记录；本对象 cached RGB/CPU physics 没有新 renderer，不把旧映射proof称新渲染验收。已有2.45GiB诊断峰值不作训练显存保证。完整联合父入口 **3600秒硬上限，最多1 GPU小时**，包含加载/编译/128×2更新/physics/拟合/联评/写盘/退出链；是当前对象上限，不是预计耗时或整个剩余目标预算。独立CPU审计另300秒/0GPU，校验源码/source/封口/完整更新与key链、独立算术、stored-applied 3360步复放；复用同一官方模拟器须明示。

先完成确定提交上的CPU fixtures/只读tiny模型/学习mask/Linux owned进程验收，才fresh核资源并执行一次完整对象。失败/超时保存新attempt，工程修复不能改变科学语义、放宽守卫或追加更新求阳性。独立数值审计分两层：用直接kernel solve独立重建全部cal LOEO/λ0/共享F和signal regressors，与原实际sample predictions比较absolute≤1e−10并要求选择exact；再从这些已认证、hash绑定的原实际sample测量值独立复算mean4/choices/S/U/MSE/2000次fsum bootstrap及CI零侧exact。不同solver的舍入测量不替代原统计输入；仅复用原aggregate不算独审，fsum若跨CI零侧仍严格失败。全部calOOF及blind sample predictions保存，参数/源/模型/optimizer认证另靠完整运行链，不称独立模型或优化器重演。产物 `/data/Policy_Discrepancy/runs/dl-reset-joint-<date>-<sha>/`，本机ignored镜像 `analysis/outputs/dl-reset-joint/`；checkpoint/source/replay不入Git。完整源码SHA、实际配置、环境/设备/counters、raw outputs、freeze hashes、退出及独审写入运行收据。

无论结果如何，该对象结束后停止，按证据决定后续。仍缺普通 replay state、合法干扰/错误观测依赖、连续 actor的实际后果、选择性gate/匹配强度/conditional-shuffle对照及后续学习性能。只有这些适用证据建立后才形成可正式实验方法；若有限q信息或机制不足，保留明确局限，不把fixture、读出或文档完备当方法完成。
