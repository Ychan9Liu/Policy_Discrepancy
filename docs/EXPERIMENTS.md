# 实验协议与结果索引

更新日期：2026-10-09。

**2026-10-09 最新baseline执行声明**：用户已明确sv2 GPU0–3四条正式logging baseline使用prealloc=False、JSONL+W&B、关闭scope，W&B project `PD_1`。配置缺项已解除，按已授权范围准备验证日志并启动；冻结科学协议不变，其他三组尚未授权。本段替代下方历史“等待声明”，状态/完整记录见 `docs/FORMAL_BASELINES.md` 和STATUS当前段。

**2026-10-09 正式baseline任务（已授权，未启动）**：用户分配sv2物理GPU0–3，授权四任务logging-only全预算baseline、逐任务正式c及原始return指标，范围不含其余三组。设备/依赖/确定版本核查通过，当前等待prealloc/logger执行设置声明；科学协议不变，启动授权不重复请求。详见 `docs/FORMAL_BASELINES.md`、STATUS当前段与analysis/outputs中的完整配置备选，未运行训练或产生正式c/指标。

**2026-10-09 正式准备索引（未启动）**：`docs/M2_V1_FORMAL_PREPARATION.md` 固定完整共同执行配置、sv2四任务CUDA UUID/EGL映射、16条新独立目录及baseline→正式c→其余组依赖。最终计划索引在 `analysis/outputs/formal-m2-v1-preparation-final/planned-run-index.json`，不是结果索引；全未授权/未启动，四份constant的c未绑定，正式指标为空。此次只有CPU配置解析和无训练单帧设备探针；不增加正式训练授权，不改变下述冻结科学协议或旧证据。

**2026-10-09 04执行核验索引**：旧16条真实size50m工程产物/统计已独立复核，新增四任务schema v2 CUDA报告与状态隔离、固定输入/恢复/去重补验见 `docs/SERVER_RETURN_CUDA_AUDIT.md` 和 STATUS当前段。工程表的正式AUC/tail仍为空，原始return冻结口径未变；新增工程c不得正式复用。CUDA与EGL设备须同时绑定、现场核查。此次有限工程完成不新增正式百万步运行授权。

**2026-10-09 指标口径补充（已确认，当前）**：继续沿用已冻结的原始 return 指标协议，不重新确认全部 v1，也不引入归一化 score。episode return、评价点均值、按 seed 独立 AUC、任务等权汇总、指定末段五点和缺失标记按下文 C 节执行。score 讨论不作为工程验收或正式实验准备的阻塞项；本次只澄清指标，不表示工程验收已通过、不新增正式实验启动授权。已有四任务 size50m 的16个缩小预算工程主运行及后续修复证据分别见 `docs/SERVER_INTEGRATION_SIZE50M.md`、`docs/CODE_SIZE50M_REPAIR.md`、`docs/STATUS.md`；工程 c/return 不混入正式实验，正式百万步比较未启动。

状态：M2 实验协议 v1 已由用户于 2026-10-08 确认，作为 03 的明确实施交接版本。匹配窗口、统计单位、预算、评价网格和指标均已冻结；完整工程配置、随机流映射和运行链路仍须落实、验收。正式比较尚未启动，本轮不启动实验。方法定义及算法验收见 `docs/METHOD_SPEC.md`，既有工程验收证据见 `docs/CODE_PHASE1.md`、`docs/CODE_PHASE2.md`。

当前研究参数以本文件已确认的 v1 为准，不再将这些参数列为待用户确认。此前闭区间窗口、D* 反算 alpha、目标权重、独立校准训练 seed 或三个训练 seed 的建议不适用于当前 v1。带日期的历史讨论和工程短跑记录继续保留，不作为正式 M2 数据。工程待落实不表示研究协议仍未决定，也不构成本轮启动实验的授权。

## M2 v1 已确认设置

- 主问题为 M2：Dt 是否提供优于统一减弱 rep KL 的逐时刻选择信号。
- 首版使用参考定义，只支持本仓库标准 categorical RSSM 与连续 bounded_normal actor。
- 四组为 baseline、Dt、固定 constant、global shuffle；条件打乱不在 v1 四组内，更强 M2 再另定协议。基础四组不自动排除活跃性/KL 大小等混杂。
- mode 漏检、早期不敏感、干扰误敏感与延迟线索列为局限并诊断，首版暂不叠加修复。
- 四任务已确认：`dmc_hopper_hop`、`dmc_quadruped_run`、`dmc_quadruped_walk`、`dmc_reacher_hard`；四者均 `image=True`、`proprio=False`、clean 环境。训练和评价均不加入背景/动力学干扰。
- 四任务统一 `size50m`；free-nats=1，dyn/rep 外层 scale=1/0.1；`ac_grads=false`、`reward_grad=true`、`repval_grad=true`，四组一致。其余完整配置与正式提交仍须冻结，不能叠加 `debug` 覆盖规模。
- Dt 与 shuffle 的 alpha 固定为 20，不用 D* 或目标权重反算，也不根据作用强弱自动调整。baseline logging 的候选源权重 w 用同一个 20 执行已确认的 c 匹配；constant 的固定 c 按任务计算后冻结，目前尚无实测数值。
- 只有训练 seed0，即 `seed=0`，各组共用；不另设独立校准训练 seed。环境、训练动作/更新/replay、评估与 shuffle 的具体随机流映射是待冻结的工程字段，不能将“同 seed 配置”写成“同一随机流/同一轨迹”。
- M2 仍是待检验研究问题，当前没有正式运行或效果结论。单训练 seed 的结果只能作当前 seed 与这些任务下的探索性比较，不能声称跨 seed 稳定有效。

## M2 v1 已确认协议（03 交接）

下表与后续统计、计步、评价规则均已由用户确认。03 按此实现和验证，不重新等待窗口、预算、指标或效果阈值的选择。工程派生细节见后文，不得改变已冻结口径。

| 字段 | v1 冻结口径 | 状态 |
| --- | --- | --- |
| baseline 与匹配来源 | 单训练 seed=0；每任务一次 logging-only baseline，实际 rep 权重为 1；同时作为正式 baseline 与 c 匹配来源，不另设校准训练 | 已确认 |
| 匹配窗口 | 半开区间 `[100000,300000)` 训练动作步，按优化更新开始时的累计动作步归属 | 已确认 |
| 匹配统计量 | 每次成功优化更新中的有效 loss 位置，raw rep KL>1 时累计 `sum(active*w)` 与 `sum(active)`；四任务各计算一个 c | 已确认 |
| 重复记录 | replay 重采样保留计数；仅恢复或日志重写产生的同一 update ID 重复记录去重 | 已确认 |
| 启动与冻结 | constant 启动前冻结本任务 c；四组从头训练 seed0，不从 baseline 检查点分叉 | 已确认 |
| 预算与 repeat | 每组 1000000 训练动作步、repeat=1；包含 replay 预填充，排除 reset 和评估 | 已确认 |
| 评价网格 | `0,50000,...,1000000` 共 21 点；每点恰好 10 个完整 episode，固定参数快照、采样动作、独立评价随机流 | 已确认 |
| 主/末段指标 | 梯形 return AUC/1000000；800000、850000、900000、950000、1000000 五点评价 return 的算术平均 | 已确认 |
| 差值与整体结果 | 每任务报告 Dt−baseline、Dt−constant、Dt−shuffle；四任务原始 return 指标等权汇总，并保留逐任务结果 | 已确认 |
| 解释与失败规则 | 单 seed 描述性探索，不设最小增益/等效阈值，不作跨 seed 稳定性或统计等效结论；无活跃位置/非法值停止报告，不填 c=1、不自动修改 alpha | 已确认 |

### A baseline 复用、c 匹配与冻结

baseline 配置为 `agent.rep_probe.mode=logging`、`alpha=20`，实际 rep 权重 `v=1`。logging 的候选源权重 w 与实际权重 v 必须分开；不能用 baseline 的实际活跃权重 1 去计算 c。constant 的实际 rep 权重始终为冻结的 c；其只读源权重诊断沿用 alpha=20，不进行额外 alpha 选择。

令 k 为一次成功完成的优化更新，s_k 为开始该更新时已完成的累计训练动作步。仅当 `100000<=s_k<300000` 时纳入匹配窗口：起点包含，终点不包含。s_k 在更新开始时记录，S/N 仅在该次优化更新成功后纳入；不以更新结束步、batch 内 transition 采集时间、日志写出时间或 optimizer step 替代。

对该任务窗口内每次成功优化更新的实际训练 batch、replay context 裁剪后的有效 rep loss 位置 i：

```text
R_ki = baseline raw rep KL (unimix 后、因子求和后、free-nats 前)
A_ki = 1[R_ki > 1]
w_ki = 1/(1 + 20*D_ki)
S = sum_k sum_i A_ki*w_ki
N = sum_k sum_i A_ki
c_task = S/N
```

当前 baseline 无 padding loss mask，有效位置是原 loss 的完整 `[B,T]`；不因 is_first/is_terminal 增加排除规则。reset 不计动作预算，不等于将 episode 起点从 rep loss 中删除。统计单位是每次成功优化更新的有效 loss 位置，而非唯一 transition；replay 再次抽到同一数据是正常训练样本，每次成功更新均保留计数，不按 transition ID 或数据内容去重。失败/未提交的优化更新、只读 report/评估、prefetch 未消费批次、编译/初始化 fixture 不计入。

累计原始统计和与计数，不能从 D_mean/p90 反推 c，不能对日志里的 active_weight_mean 简单平均。多个 shard 对实际有效位置全局求和，复制的全局结果只累计一次。以本运行唯一 update ID 标识每次成功更新，仅去除恢复或日志重写造成的同一 ID 重复记录；同 ID 的 S/N 或 s_k 不一致应报工程错误，不能静默挑选一条。窗口更新逐一归属；宿主累计建议用 float64 的 S 和整数 N，并记录每次更新的来源和精度。

若某次成功更新没有活跃位置，贡献 S=N=0；整个预定窗口 N=0 时停止该任务匹配并报告，不启动该任务 constant。出现非有限 D/w/raw KL/c 或非法 c（不满足 `0<c<=1`），应报告工程错误并停止，不偷偷丢弃坏位置、裁剪 c、填 c=1 或自动修改 alpha。空活跃集合的均值标为不可用，不将其 NaN 与非法 D 混淆。

每任务 c 的冻结记录至少含：task、baseline 运行 ID/提交/完整配置哈希、窗口端点及动作计步口径、纳入 update 清单、S、N、c、alpha=20、阈值 1、无效值统计、生成工具版本、冻结时间与产物校验值。先冻结，再启动 constant。baseline 继续完成其正式预算，不额外训练校准模型；匹配窗口统计已经取得不等于 baseline 比较已经完成。

不得按 return、组间结果或作用不足更改窗口、alpha 或 c。四任务匹配规则相同、c 独立。此规则只匹配本任务 baseline 窗口内的活跃位置平均候选系数，不保证全训练过程或不同组之间的惩罚值、梯度向量与参数更新相等。

### B 训练动作预算与随机流

动作步 s 定义为所有训练环境中已执行、且确实推进环境的非 reset 动作总数，包含 replay 预填充阶段产生的动作步；不是 optimizer update 数、frame 数、policy 调用数或含 reset 的 driver 返回次数。每组预算固定为 B=1000000，repeat=1；reset 与评估动作排除于 B，并另记 reset 次数、eval 动作步、optimizer 更新数和训练墙钟。

四组从头创建模型、优化器、replay、环境及训练计数器，均用训练 seed0；不恢复 baseline 参数、replay、计数器或窗口检查点。组间数据轨迹可因策略不同而分叉。以同提交完整配置冻结 batch/replay/更新频率、优化器、观测尺寸/相机/episode horizon、环境数、JAX dtype/设备拓扑等；只允许 task、权重模式、本任务 c、独立产物路径等已声明差异。评估运行和出日志不能隐式改变更新预算或 RNG。

记录每条流的 root seed、命名空间/派生函数、task、环境 index、episode/评价点/update counter、恢复策略及实际 seed/key：训练初始化/更新、训练 policy 动作、训练环境、replay 抽样、评估环境、评估动作/latent 采样，以及 shuffle。shuffle 独立 key 与 optimizer step 对应；评估与 shuffle 不消耗原训练随机流。

固定评估映射建议以 `(task, eval_index, episode_index)` 标识，各组共用同一映射与并行布局；是否每次评价复用还是更换初始情形在映射中明确。可使用 seed0 派生的独立评估命名空间，这不增加训练 seed；具体派生数值待工程方案冻结。不得只写“seed=0”而省略各流映射。

### C 评价网格与指标

固定网格为 `s_j=50000*j, j=0,...,20`，共 21 点，每点恰好 10 个完整 episode。s=0 是初始化完成、训练交互（含 replay 预填充）与更新开始前的策略，不能用预填充后的检查点代替。每点固定不可变参数快照，记录实际动作步和 update 编号；训练结束后完成 s=1000000 的终点评估，不能因循环退出漏掉终点。

单个 episode 的 return 是完整 episode 内原始环境 reward 的累计和：不折扣、不除以 episode 长度、不使用模型预测奖励或训练时变换后的奖励。环境接口已将 action repeat 内的 reward 累计，统计端每次接口返回只累加一次，不再乘 repeat 或重复展开奖励；reset 返回不产生动作 reward，不能把下一个 episode 的奖励拼入本 episode。

每个评价点对该点全部预定完整评价 episode 的 return 求算术平均，v1 恰好10条，即 `r_j=(1/10)*sum_e return_je`。不按 episode 长度加权，不把不完整 episode 当成完整样本，也不因 return 高低删选 episode。评估沿用采样动作，不能部分组用 mean/mode。并行评估预先固定10个 episode ID，保留全部 return/长度/随机性记录。评估不得写入训练 replay、推进训练 RNG/计数器、更新训练 normalization/模型/优化器，也不得读取仍在变化的参数。

每个训练 seed 在每任务/组内独立计算 mean return 曲线的梯形积分，再除以固定训练动作预算 B=1000000。横坐标 x 是评价快照对应的实际累计训练动作步（含 replay 预填充、排除 reset/评估），记录实际步数而非日志写出步或评价动作数；冻结网格要求 x_j=50000*j。原始 return AUC 只按固定预算除一次，不是 score 归一化。

```text
AUC_task,group,seed = sum_i ((x_(i+1)-x_i) * (r_(i+1)+r_i)/2) / B
Tail_task,group,seed = (r_at_800000+r_at_850000+r_at_900000+r_at_950000+r_at_1000000)/5
M_task,group = mean_over_training_seeds(AUC_task,group,seed)
T_task,group = mean_over_training_seeds(Tail_task,group,seed)
Delta_task,ref(M) = M_task,Dt - M_task,ref
Delta_task,ref(T) = T_task,Dt - T_task,ref
ref in {baseline, constant, shuffle}
Overall_X_group = (1/4)*sum_over_four_tasks X_task,group, for X in {M,T}
Overall_Delta_ref(X) = (1/4)*sum_over_four_tasks Delta_task,ref(X)
```

v1 的训练 seed 集合仍只有 `{0}`，上述 seed 平均当前退化为该 seed 的指标；公式说明统计层级，不授权新增训练 seed。若以后经单独协议增加 seed，先每个 seed 独立积分，再在任务内对 seed 的 AUC 求算术平均，最后四任务等权平均。不能将不同任务或不同 seed 的 episode 池合并后直接作为独立训练重复，不根据结果改变任务权重。

末段固定为 800000、850000、900000、950000、1000000 五个评价点 mean return 的算术平均，不直接取日志最后五条。必须按实际训练动作步和快照核对与预定点的对应关系；日志乱序、重写或额外工程记录不改变该集合。四任务按原始 episodic return 的 M/T 与差值等权汇总，不换成归一化 score、相对百分比、baseline 比例或 min-max 标准化。保留每任务/组/seed 的曲线和指标、任务均值、四任务总体值及三项 Dt 差值，不让整体均值掩盖负向任务。

缺失 AUC 初始/终点、任何冻结网格评价点，或缺失指定末段五点时，分别将受影响指标标为不完整，并记录缺失点和原因；短预算退出、episode 数不足或快照步号错位也不能伪装成完整指标。保留可用原始记录供审计，不自行缩短积分区间、改变 B、替换末段点、前向填充或跨缺点补插评价。某指标不完整不应自动删除另一项仍完整的指标；但完整四任务/seed 总体指标需声明预定组成是否齐全，不能默默忽略缺失项后重加权。03 必须落实精确动作步网格，无法生成对应快照时报告工程阻碍并修复，不自行放宽。当前只作单 seed 描述性探索，不设最小增益或等效阈值，不作跨 seed 稳定性或统计等效结论，也不根据结果追选判据。

### C.1 字段语义、结果表与 03/04 验收（2026-10-09 已确认口径）

现有字段叫 `score/scores` 不意味着归一化：若实际为累计原始环境 reward，应在字段映射和产物 schema 中明确 `score := episode_return_raw`，`mean := mean_episode_return_raw`。允许保留兼容字段名，不要求为改名而重跑；03/04 要核查实际奖励来源、累计方式和完整 episode，再记录语义，不能只凭名称认定正确。后续引入归一化 score 需单独记录协议变更及适用范围，本轮暂缓该讨论，不作为工程验收或正式准备的阻塞项。

统计代码和结果表至少保留以下信息（字段名可按兼容性映射，语义不可改变）：

| 层级 | 必须可追溯的字段/内容 |
| --- | --- |
| episode | task/group/训练 seed、run ID、评价点 ID/目标动作步/实际快照动作步、snapshot/update ID、评价 episode ID、环境/动作随机流、原始 return、长度、完整性与奖励来源语义 |
| 评价点 | 目标/实际训练动作步、全部10条完整 episode 的 return、episode 数、mean return、完整性/偏离原因；每个点只能对应一个有效固定快照，冲突记录报告而非任选 |
| 每任务/组/训练 seed | 固定 B、使用的动作步列表、return AUC/B、指定五点评价均值、各指标 complete 标记和缺失/冲突列表、源提交/配置/文件映射 |
| 任务与四任务汇总 | 每任务先对预定训练 seed 指标平均、四任务等权平均、三项 Dt 差值；保留各任务与 seed 原值及组成完整性，不报告跨 seed 稳定性或统计等效 |

03 应落实/核对统计代码和结果表，并以手算可核验 fixture 覆盖：原始奖励含不同 episode 长度、非零最终奖励、repeat 已累计；评价点均值；非等间距实际 x 的梯形公式与固定 B；逐 seed 积分→任务均值→四任务等权（多 seed 仅作统计 fixture，不新增正式训练）；乱序/额外记录仍按指定五点选取；边界/指定点/完整 episode 缺失时明确不完整，不换分母、不替点；旧 scores 字段的 return 映射。04 用既有工程产物核查提交、快照、实际动作步、episode 与字段语义，工程缩小网格不能冒充正式21点或正式百万步 AUC。新增运行需符合另行授权，不能为本次指标说明自动启动训练。

统计实现与必要测试保存在仓库适当代码/测试目录，生成表格与审计产物放 `analysis/outputs/` 或既有服务器运行目录的独立分析子目录，大产物不提交。字段映射/缺失点/配置与源文件索引随结果保存，验收记录更新 `docs/CODE_PROTOCOL_V1.md`、`docs/STATUS.md`，结果索引更新本文件。完成交接按 AGENTS 提供完整最终/实际测试 SHA、差异、同步范围、已通过和未通过项及下一步 prompt，不能把工程示例表写成正式 M2 结果。

本轮源码只读核对基准为 `696fa9e0ded3a8da0864beadfb850af05668e1c3`：`embodied/run/protocol_v1.py` 的 evaluate 累计 `tran['reward']` 到 scores，record_evaluation 对 scores 求 mean；`embodied/envs/from_dm.py` 的 reset reward 为0、动作 reward 来自 DM 环境，DMC 的 `ActionRepeat` 已累计 reward，main 的通用包装未作奖励变换。当前 `scripts/analyze_size50m_integration.py` 是工程/资源审计，未提供完整冻结网格的正式 AUC/指定五点/跨任务指标汇总。这些是源码事实，不是本轮运行验收；上述统计 fixture 与结果表仍由03/04落实核验。本次仅更新文档，不改代码、不启动实验，不宣称工程通过或 M2 有效。

### D 作用诊断与解释边界

报告精确 `D=0` 比例、可选的近零比例及其固定阈值；raw rep KL>1 的活跃比例；源 w 与实际 v 在活跃位置的平均值；实际 rep_before/after、加权超额 KL 及减少量。若用相对超额减少量，分母为 `sum((R-1)_+)`，为零时标记不可用；不能把 free-nats 常数减少解释为梯度压缩减少。logging 实际 v=1，源 w 仍可小于 1；shuffle 要报告置换后实际权重和原始源权重，不以源活跃均值代替实际值。

`D=0 => w=1` 是正常定义行为，不作错误或自动调 alpha 的依据。若实际作用很弱，保留 alpha=20 并报告其比例与阶段变化，说明当前配置的机制检验力度；非有限 D/w、非法 c 或 loss/梯度属于工程错误，不能混成科学失败结果。

一个训练 seed 不能估计训练 seed 间变异。评价 episode 和时间点均不能充当独立训练重复；四任务也不是同任务的四个训练 seed。可报告 episode 内描述性离散程度，但不能将其变成跨 seed 的置信区间或“稳定有效”证明。M2 保留为问题，结果仅支持当前 seed、任务、配置和对照下的描述性探索，不作跨 seed 稳定性或统计等效结论；新增训练 seed 的确认另定后续协议。

clean v1 不提供干扰鲁棒性证据；任务级干扰误敏感和延迟线索干预未纳入 v1 的 16 条训练（四任务×四组）。mode/早期作用先用日志诊断，其余局限如未完成干预应明确保留。全局 shuffle 仍可能改变活跃性和 KL 大小关联，更强选择性解释需条件 shuffle。

## 03 尚需落实的工程字段（不是未决科研选择）

已确认的 v1 研究口径不再等待批准。03 需提供同提交的完整实际配置、环境/训练/replay/评估/shuffle 的随机流派生与恢复映射、更新成功与 update ID 定义、S/N 收集和冻结产物、精确计步与参数快照，以及续跑/失败/缺失评价的处理记录。04 在运行前核实服务器/GPU、依赖和提交一致性；这些技术字段必须符合本协议，不能改变窗口、预算、网格、alpha 或解释边界。

c 的实测数值尚未产生，是按确定规则计算的运行产物，不是留给 03 选择的超参数。效果阈值并非“待定”：v1 已决定不设最小增益/等效阈值，只作描述性探索。具体 RNG 派生数值和完整配置仍待工程冻结，研究结论仍须后续数据支持。

03 的统计验收至少覆盖：s_k=100000 纳入、s_k=300000 排除；更新结束时越界仍按开始时归属；失败更新不累计；同一 replay 数据在两个成功 update ID 中计两次；恢复/重写的同 ID 同统计只计一次、统计冲突报错；不同活跃计数批次按 S/N 合并而非平均均值；全窗口 N=0 或非法值停止报告。精确动作计步/21 点快照/10 完整 episode 与 RNG 隔离也须形成可复核验收记录，不以既有 overlay 小 fixture 代替。

## 04 实施前工程条件（源码只读核对）

源码核对基准为 `4b8b00f54e7f1f5bbaa959391d0c18beba2952f2`；以下问题来自确认前的只读核对，本轮未改代码、未连接服务器运行。作为交接条件继续有效，不撤销 `CODE_PHASE1/2` 的小 fixture 验收，也不把它们升级为协议链路已可直接执行。

| 条件 | 当前源码/证据 | 实施前要求 |
| --- | --- | --- |
| c 统计可审计 | `dreamerv3/rep_probe.py` 只输出活跃均值等；活跃 count 和 `sum(active*w)` 仍是内部量，未作为输出暴露。`train_eval` 又将多更新 metrics 聚合 | 提供每次成功更新的 S/N、有效位置总数、更新开始时 s_k 和 update ID；半开区间归属，仅同 ID 恢复/重写记录去重，正常 replay 重采样保留，能从原始统计重算 c |
| 动作计数与停止 | `Driver` 与 `train/train_eval` 对 reset 返回也计 step；原 `run.steps` 不直接等于本协议动作预算 | 独立 train_action_steps/reset/eval/update 计数；窗口和评价网格用同一个真实动作计数器，检查多环境 overshoot、精确预算/终点快照与预填充 |
| 独立评价随机流 | `embodied/jax/agent.py:policy` 不按 train/eval 分开 n_actions；`stream` 对 train/report/eval 共用 n_batches；现有 train_eval 使用同一个 agent | 评价/报告不得消耗训练 policy、更新或 replay 随机流；核查共享 counter、report prefetch、恢复的 checkpoint counter 和 params 同步。使用隔离评估实例/进程或经验证的独立流方案 |
| 评价调度与 episode 数 | `train_eval` 用 `Clock(report_every)` 按墙钟调度；`eval_only` 以 step 预算循环；Driver 多环境可能超额完成 episode | 另验证动作步网格、初始化/终点评估、10 个预定完整 episode、参数版本/快照与 seed 映射，不把 report_every 改成 50000 秒作为替代 |
| 环境 seed 与观测 | make_env 用 `hash((config.seed,index+seed_offset))` 派生，DMC 接收 seed；早期 smoke 只覆盖小规模接口与首次状态 | 记录实际 seed、index/offset、版本及跨进程映射；固定纯视觉、clean、repeat/分辨率/相机/episode horizon，独立 eval 环境与动作随机性 |
| 作用诊断精度 | 现有 `raw_active_D_near_zero_frac` 是活跃且 D<=1e-6 对全部位置的比例，不是精确零值比例或条件概率；空活跃均值可能为 NaN | 提供精确零值计数/总数、活跃计数和实际减弱 S/分母；区分 undefined 均值与非法 D，基于实际权重 v 分模式检查 |
| 模型与配置一致性 | 已有 identity、overlay 与跨 shard shuffle 小 fixture 证据；size50m 四任务资源及完整协议链路未验收 | 固定不带 debug 的完整配置/同提交；核查显存、吞吐、观测/动作接口、KL/梯度开关和全局 shuffle，不能根据作用强弱改 alpha |
| 提交/恢复/产物 | 当前本机 main 相对本地 origin/main 记录 ahead 25；不代表本轮已验证远端或服务器 HEAD | 运行前核实本机/服务器同一提交、依赖/GPU；每条训练和每次评估独立产物索引，checkpoint/RNG/计数及 S/N exactly-once 恢复，禁止结果驱动重跑筛选 |

既有 logging/off/overlay 工程测试结果见 `docs/CODE_PHASE1.md`、`docs/CODE_PHASE2.md`；只说明算法局部语义，不替代上述收集与评估链路条件。03 可按已确认 v1 实现并验收这些条件，04 在条件满足后才可依据后续运行指令启动正式比较。本次只更新并提交三份协议文档，不改算法、不推送、不启动实验；交接提交 SHA 以 Git 完成报告为准。

## 运行记录要求

正式实验前固定：任务及环境版本、baseline、方法配置、训练步数或交互预算、seed 列表、评价频率与指标、GPU 分配、模型规模、停止与续跑规则。对照实验仅改变预先声明的变量。

每次运行保存：运行编号、Git commit、实际 `config.yaml`、环境依赖快照、服务器与 GPU、seed、启动命令、开始与结束时间、退出状态、指标文件和检查点位置。不同运行使用不同 `logdir`；只有明确续跑才复用。

## v1 之前的协议讨论（2026-10-08 历史记录）

当时只冻结任务和观测，曾建议先校准再反算 alpha、活跃位置权重匹配和三个训练 seed。当前 v1 已确定 alpha=20、只用 seed0 且无独立校准训练 seed；这些旧建议不再是当前选择。环境首次状态重复性和 `58cae2a` 小模型训练/独立评估烟测仍是工程事实；当前正式预算、窗口和评价已由本轮确认。

保留当时的协议讨论原文，以下不是当前运行要求：

> 四个真实 DMC 环境已在三台服务器完成创建与步进烟测；提交 `58cae2a` 上四者也已完成小模型训练更新和独立评估的工程短跑。环境 seed 在 `make_env` 入口的首次物理状态可重复；`train_eval` 合并运行、完整训练重跑和正式评估协议仍待验证或冻结。先固定可复现 baseline 与 logging-only 校准规则，再冻结评价协议和预算，之后按预定规则计算并冻结 alpha/c。return AUC、活跃位置平均权重匹配和先用三个训练 seed 筛查均为候选方案，尚未成为正式协议。

### v1 草案讨论（2026-10-08，确认前历史）

确认前曾建议闭区间 `[100000,300000]`，将 baseline 复用、预算、21 点评价和指标列为候选。本轮用户已确认半开区间 `[100000,300000)`、优化更新开始时归属、成功更新的有效 loss 位置和仅重复 update ID 去重，并确认本文件 v1 的预算、评价及描述性解释范围。闭区间建议和“仍待确认”状态不再有效。

## 结果索引

### 2026-10-09 原始 return 统计与表格验收（非正式 M2 数据）

C/C.1统计实现与字段映射已由03落实：`dreamerv3/return_stats.py`、`analysis/return_metrics.py`、兼容的protocol_v1评价schema v2；使用说明在 `analysis/README.md`，验收报告在 `docs/CODE_PROTOCOL_V1.md` 最新节。统计/相关单元测试实际SHA `1f5b5a385346da01e1108c90114fed721c8bbaf2`，CPU完整agent fixture实际 `c4de170928e9314559be793634209320d97fe0b4`。15项手算统计及39项相关回归通过；多seed合成表只作统计fixture，不授权新增训练seed。

既有04主运行的只读表根目录（本机Git忽略）：`analysis/outputs/return-v1-engineering-audit-1f5b5a3/`，含episode/评价点/seed/任务/总体/三项差值六表及JSON/audit。48个工程点、96条episode的score映射为原始return，源提交仍是 `1393548fb6e46d28f92a834204f7b09495a3eab1`；96个输入文件实时哈希与三台服务器原目录相同，来源index/哈希在 `analysis/outputs/return-v1-source-20261009/`。工程点按每点2episode核对为 `source_complete`，不表示满足正式每点10episode；预算4098/网格0、2049、4098不满足正式21点。各层正式AUC/指定末段指标与差值全部null/不完整，记录缺失与来源原因，未生成缩短区间或归一化score结果。旧聚合日志缺少逐reward/terminal轨迹，完整性依据已审查源码逐episodeDriver路径及快照读审计，不能声称重算了每步原始奖励。

手算fixture表 `analysis/outputs/return-v1-hand-fixture-1f5b5a3/`，产物哈希/核查摘要 `analysis/outputs/return-v1-validation-summary-1f5b5a3.json`；CPU/debug完整agent新schema表在sv1 `/data/Policy_Discrepancy/repo/analysis/outputs/return-v1-agent-schema-audit-1f5b5a3/`。这些不是正式运行或M2效果证据。03实现和本次允许的只读核查已完成；下一步现有04做独立产物质量检查与前轮真实size50m工程补验，正式比较还需工程闭环和用户授权，score讨论不阻塞。

### 2026-10-09 真实 size50m 工程集成（非正式 M2 数据）

共同产物根目录 `/data/Policy_Discrepancy/runs/engineering-size50m-20261009-1393548/`；每任务下 `logging/dt/constant/shuffle` 四个独立目录。主矩阵实际提交 `1393548fb6e46d28f92a834204f7b09495a3eab1`，四任务/四模式均seed0，size50m/CUDA/bfloat16，4098训练动作、每2049动作评价2完整episode、匹配窗口 `[2200,3600)`，保留固定科学设置。工程baseline生成本任务隔离工程c，再从头跑constant；不能复用于正式比较。

| 任务 | 服务器/物理GPU | 四组主链路 | 工程c（非正式） |
| --- | --- | --- | --- |
| dmc_hopper_hop | sv1/0 | 完成，主链路审计通过 | 0.9999511233767909 |
| dmc_quadruped_run | sv1/1 | 完成，主链路审计通过 | 0.9999361077376775 |
| dmc_quadruped_walk | sv2/0 | 完成，主链路审计通过 | 0.9999402671200889 |
| dmc_reacher_hard | sv3/0 | 完成，主链路审计通过 | 0.9999290457282546 |

补验：sv1 hopper `isolation` 在CUDA报告路径失败；`evaluation-isolation`和`same-config-repeat`完成但严格参数比较未通过（后两者提交 `63d36a111d247aa8cd05af173f74e43e4e35d2e1`，方法代码未变）；sv3 reacher `interrupted`在2049动作故意中断并验证拒绝续跑。四个logging完整重开不追加评价或收据。所有失败/补验目录保留；整体验收未闭环、没有方法效果结论。完整配置、命令、环境、诊断、资源估计、故障证据及给03的prompt见集成报告，原始产物留服务器。

### 真实环境工程短跑（非正式 M2 数据）

下列训练均用提交 `58cae2a`、`dmc_vision debug` 小模型、CUDA、训练 seed 7、1200 环境步；独立 `eval_only` 用训练后检查点及 seed 17，均得到一条 1001 步 episode 和 `scores.jsonl`。各训练的 `metrics.jsonl` 均含 `train/loss/*`。这组极小模型的 return 不用于比较任务或判断方法效果。

| 任务 | 服务器 | 训练目录 | 评估目录 |
| --- | --- | --- | --- |
| `dmc_hopper_hop` | sv2 | `smoke-dmc-hopper-train-sv2-58cae2a` | `smoke-eval-hopper-sv2-58cae2a` |
| `dmc_quadruped_run` | sv1 | `smoke-dmc-quad-run-sv1-58cae2a` | `smoke-eval-quad-run-sv1-58cae2a` |
| `dmc_quadruped_walk` | sv2 | `smoke-dmc-quad-walk-sv2-58cae2a` | `smoke-eval-quad-walk-sv2-58cae2a` |
| `dmc_reacher_hard` | sv3 | `smoke-dmc-reacher-hard-sv3-58cae2a` | `smoke-eval-reacher-hard-sv3-58cae2a` |

所有目录均位于 `/data/Policy_Discrepancy/runs/`；`quadruped_run` 另有 `smoke-eval-quad-run-repeat-sv1-58cae2a`，同检查点/seed 的 episode return 与首次运行完全一致。正式实验另建目录并先冻结协议。

| 运行编号 | 任务／方法 | seed | 提交 | 服务器／GPU | 结果目录 | 状态与主要观察 |
| --- | --- | --- | --- | --- | --- | --- |

正式结果和检查点保留在服务器 `/data/Policy_Discrepancy/runs`。此处记录可追溯的索引、比较结论与失败运行原因，不把大型产物提交到 Git。
