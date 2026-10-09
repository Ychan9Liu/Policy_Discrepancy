# 项目状态

更新日期：2026-10-09。

## 03 修复及真实 size50m CUDA 验收（2026-10-09，当前）

- 03 已修复 CUDA `report_on_batch` 设备数组切片的 transfer guard 异常：复用训练消费的主机 batch，主机切片后显式 sharded 传输；报告使用独立 seed，不重新抽 replay。已有不完整 checkpoint（包括动作数0）拒绝续跑，先核对 checkpoint 与持久化收据/评价；重复持久化 ID 拒绝，完成运行重开不追加。最后代码/测试实测提交 `20ab3486bfaa067cc7950d3f5c8408fe2ed92ec3`，22项相关 unittest 通过。详细提交、配置、命令、结果与边界见 `docs/CODE_SIZE50M_REPAIR.md`。
- sv1 真实 hopper size50m CUDA/bfloat16 报告开启运行：原4098动作/2049报告节奏在 `98225e...` 子进程完成2次报告；最新代码在独立2048动作工程 fixture 上完成报告与2点评价，并逐项审计报告/评价前后训练参数、训练计数、replay RNG/metrics 等未变。完成运行同配置重开退出0，关键产物哈希不变。原04零动作故障目录只读审计确认 checkpoint 更新0而收据0..4，明确拒绝。新证据目录 `/data/Policy_Discrepancy/runs/engineering-size50m-repair-20261009-98225e1/`。
- 真实同配置 hopper 的首个差异已定位为**视觉图像**：两次2048动作追踪中物理状态、动作、奖励逐转移相同，6条图像不同，第一条在动作0/reset；首批训练图像输入已不同、参数从首次更新后开始分歧。独立 DMC/EGL 固定物理轨迹复现单像素差异；冻结相同输入的真实 size50m CUDA 四次更新完全一致。该环境下不可要求独立渲染轨迹逐位一致；不以放宽容差掩盖差异。审计支持评价/报告未改变训练 RNG 来源，但不声称完整环境/replay 崩溃后无缝恢复。
- 03 当前任务工程修复已完成。首选由用户把本报告交回**现有04 chat**做确定提交、新目录的工程补验与质量核查；不自动发送。02 的 return/score 口径仍待确认，正式百万步比较还需用户授权。工程 c 不能用于正式运行；本轮没有正式效果结论。

## 04 真实 size50m 集成试跑（2026-10-09，修复前记录）

- 本轮四任务×四模式 16 个真实 size50m 工程主运行全部完成；实际 SHA `1393548fb6e46d28f92a834204f7b09495a3eab1`。CUDA/bfloat16、seed0、alpha20、默认 batch16×64/16环境/train_ratio256；工程预算4098、网格0/2049/4098、每点2完整episode、窗口 `[2200,3600)`。每运行4098训练动作/517成功更新，评价与reset分开计数；工程c均有隔离来源/校验，正式freeze拒绝工程来源。真实快照/收据/窗口重算/重复ID拒绝冲突/完成重开检查通过。
- **整体验收未闭环**：sv1 hopper 开启 CUDA report 在 `report_on_batch` 切片触发 transfer guard 异常；独立评价补验及同配置重跑的最终参数不逐位相同，最早收据分歧在update2/动作2040。初始化294数组、训练随机counter一致，但不能因同配置自身差异就认定评价RNG泄漏。补验 SHA `63d36a111d247aa8cd05af173f74e43e4e35d2e1`，相对主矩阵仅外部脚本变化，算法/runner不变。03修复与定位后再由04补验，不在服务器直接改代码。
- 每任务工程c约0.99993–0.99995，活跃位置有非零但很弱的减弱；未改alpha，不把工程通过写成方法有效。正式c尚无、正式16条百万步比较未启动。用户本轮告知return/score正由02重审，旧v1指标记录保留依据，最终口径须由02确认并落入文档。
- 主运行采样峰值6.48–9.31GiB、预热吞吐约24.5–24.9动作/s；百万动作及210episode资源线性估算约11.6–11.7小时/条，不是长程实测。完整配置、资源、边界、故障与可复制03修复prompt见 `docs/SERVER_INTEGRATION_SIZE50M.md`；小证据索引含完整配置见 `docs/size50m_integration_evidence.json`。执行/分析脚本在 `scripts/run_size50m_integration.py`、`scripts/analyze_size50m_integration.py`。
- 产物根目录（三台各有其承担任务）：`/data/Policy_Discrepancy/runs/engineering-size50m-20261009-1393548/`。sv1承担hopper/quad_run，sv2承担quad_walk，sv3承担reacher；完整checkpoint/replay留服务器，失败目录保留。中断非零checkpoint明确拒绝续跑，未声称无缝恢复。补验曾与sv3后来新增GPU0作业短暂共享，污染资源数据已排除，supervisor已新增每次启动前占用拒绝检查。
- 本次sv3根分区已有约321–323GiB可用，旧“根分区满”不再是当前事实；没有清理基础环境/其他作业。正式启动条件是03修复和补验闭环、02最终指标确认、正式完整配置/资源核验以及后续正式运行授权。首选下一步交现有03 chat，由用户手动发送本报告prompt；不自动派发。

## 科研交接状态

- 01 · Idea 讨论已结束，02 · 方法与机制已接手并完成首轮源码只读核对；研究记录见 `docs/RESEARCH.md`。
- 用户已确认以《DT方法迁移研究说明.docx》为参考，不采用读取该文档前助手提出的假设。原文件路径、参考设计、工作假设、反例和未决问题均已记录在 `docs/RESEARCH.md`。
- M2 主问题、首版参考方法与四组已确定；四个 clean 仅视觉 DMC 任务统一 size50m，free-nats=1、dyn/rep=1/0.1，ac_grads=false、reward_grad=true、repval_grad=true。Dt/shuffle alpha=20；只用训练 seed0，不另设校准训练。v1 的 baseline 复用、半开匹配窗口/统计规则、100万动作步预算和21点评价已有确认记录；指标现由02重审，详见 `docs/EXPERIMENTS.md` 当前补充。03 已实现完整配置、随机流映射和协议链路并以缩小预算 fixture 验收；04 的真实 size50m 短预算链路和资源记录见上方当前章节，报告/重跑缺口及正式长程仍待验证。
- 既有 DMC 适配与工程烟测、03 logging/overlay fixture 验收保留为工程证据，详见下文及 CODE_PHASE1/2；本轮协议链路验收见 `docs/CODE_PROTOCOL_V1.md`。正式 M2 比较尚未启动，没有效果结论。

## 已验证

- 本机代码仓库：`D:\program\PD\dmr3`；`origin` 为 `Ychan9Liu/Policy_Discrepancy`，`upstream` 为官方 DreamerV3。
- `sv1`、`sv2`、`sv3` 的代码位于 `/data/Policy_Discrepancy/repo`，Python 环境位于 `/data/Policy_Discrepancy/envs/dreamer`。
- 三台机器在提交 `e3f02248693a79dc8b0ebd62c93683888ddaccfe` 上通过了 `debug` 假环境的 CUDA 训练短跑，产生训练损失、日志和检查点。测试目录分别为 `/data/Policy_Discrepancy/runs/smoke-verified-sv1`、`smoke-verified-sv2`、`smoke-verified-sv3`。
- 三台机器已安装 `dm-control==1.0.48`、`mujoco==3.15.0`，`pip freeze` 哈希一致且 `pip check` 通过；依赖快照见 `env/requirements-dreamer.lock`。
- 三台机器的四个 DMC 任务均通过真实环境创建、reset、零动作步进及 64×64 RGB/EGL 渲染。动作维度依次为 4、12、12、2；仅视觉观测未暴露 proprio。
- 提交 `58cae2a` 在四个视觉 DMC 任务上均完成 1200 环境步的极小 `debug` 模型 CUDA 训练短跑，日志含训练损失；加载各自训练后检查点的独立 `eval_only` 均完成一条 1001 步 episode。运行目录见 `docs/EXPERIMENTS.md`。这是工程验收，不是 baseline 性能或 Dt 证据。
- `quadruped_run` 同一检查点与评估 seed 17 的两次独立评估得到完全相同的 episode return。`make_env` 入口四个任务同 seed/index 的首次物理状态重复、不同 index 不同；仍未证明完整训练轨迹或不同服务器之间逐位一致。
- `sv3` 的 DNS 已写入持久 Netplan 配置，GitHub 和 PyPI 访问已验证；尚未通过重启验证。

## 尚未决定或验证

- v1 的 baseline 复用、c 匹配规则、预算、repeat 和评价已有确认，不重新打开这些参数选择；指标依用户2026-10-09说明由02重审。只作描述性探索，不设最小增益/等效阈值，不作跨 seed 稳定性或统计等效结论。完整工程配置、随机流映射、成功更新统计和精确评价链路已有缩小预算 fixture 及本轮主链路记录，报告/真实重跑隔离尚未闭环；正式 c 实测数值尚未产生，不由03自选，不反算alpha或选择新增训练seed。
- 更强 M2 所需条件打乱的分层/区间/边界与退化处理协议仍未定；基础四组及首版验收要求已确定，不受其阻塞。03 在已确定范围内制定并记录配置字段、测试容差、数值处理和 RNG/全局跨 shard 置换等工程方案。
- 真实任务的小模型训练/独立评估短跑及dummy协议fixture已有记录；本轮size50m四任务短预算资源/吞吐、独立评价和四模式主链路记录见上方，不能升级为百万步长程验收。真实训练重跑与隔离缺口需03/04闭环。v1只作单seed0探索，跨seed确认属于后续工作；短跑不代表正式任务效果。
- 源码审计发现原 DMC 适配器未接收和传递环境 seed；现已补上 `DMC(seed)`、视觉预设的 `use_seed` 及 `train_eval` 评估环境索引偏移。`make_env` 的环境初始物理状态重复性已验证，但 `train_eval` 合并路径及完整训练重跑未验证；直接 DMC 渲染有极少量 1 级像素差异。
- 正式实验启动、结果汇总和跨服务器分配脚本尚未实现。
- 历史记录：`sv3` 曾根分区满，用户当时决定跳过处理；2026-10-09 的04核查见本文件当前章节，已有约321–323GiB可用，产物仍放 `/data`。

## 02 首轮方法核对进展（2026-09-27 历史记录）

- 核对提交：`f0236c7699a87895a8cfde8e13a0d1c198024f75`。接手时仅研究、状态文档有未提交修改；本次保留并增补这两份文档，未提交或推送。
- 已读取原始 DOCX 全文，并只读核对 latent、actor、KL/free-nats、replay/reduction、初始化及梯度路径；具体源码定位见研究文档。这些是源码事实及数学分析，不是运行验证。
- 确认源码中的 bounded_normal 是均值经过 tanh 的普通 Gaussian，执行动作另有裁剪；参考 Dt 对应裁剪前分布。rep KL 当前对 replay context 处理后的 `[B,T]` 直接求均值，没有 padding loss mask。
- 明确 gate 分支梯度隔离不意味着 rep 对 history 的总梯度为零，也不意味着共享参数更新后的 prior 轨迹不变。
- 研究文档新增科学主张层级、常数匹配与条件打乱的证据边界、alpha 校准退化处理、机制反例和 03 候选验收清单；均标明助手建议及未决项，尚未成为用户确认的算法或实验协议。
- 未修改算法代码，未启动 pilot、训练或实验，未运行算法数值/梯度测试。`docs/EXPERIMENTS.md` 继续保持协议待确定。

## 2026-10-08 方法选择与规范（v1 前历史记录）

下述当时未决状态及后续工程阶段记录保留；本轮 size50m、KL/梯度设置、alpha=20 和 seed0 的选择已更新，以文末 v1 状态为准。

- 已确认：M2 为主问题；首版采用参考定义，仅连续动作与标准 RSSM；至少 baseline、Dt、预先校准固定常数、全局打乱四组，更强 M2 再加条件打乱。
- 已确认：mode 漏检、早期不敏感、干扰误敏感和延迟线索列为局限并诊断，暂不叠加修复。
- 已整理 `docs/METHOD_SPEC.md` v0.1：数学/梯度路径、对照语义、诊断义务以及回退、free-nats、随机流等验收要求；研究决策和实验边界已同步到相应文档。
- 方法实现规范已可交接 03，实现可配置 overlay 与验收测试；任务、观测、校准规则、主指标、训练 seed、预算等实验字段继续按用户指定顺序固定，随后确定 alpha/c 与效果判据，不阻塞可配置方法的实现。
- 2026-10-08 先前检查现有任务列表及 Codex 归档时未找到 03 接收任务，交接材料当时已准备但未发送；指定或创建接收任务属于交接安排，不表示方法或验收仍待确认。本次一致性核对不派发实现任务。
- 先前源码核对基准为 `f0236c7699a87895a8cfde8e13a0d1c198024f75`。本次保留原有未提交文档内容并修正歧义，仅提交上述四份文档、推送 main；实际提交 SHA 和 Git 状态以完成报告及 Git 记录为准。没有算法修改或实验运行，验收条件尚未实际通过。

当时下一步是选择模型规模并补齐校准与评价；本轮模型规模已固定为 size50m，剩余协议和资源核验继续推进。正式比较不因已有方法规范或 fixture 验收自动启动。

## 03 第一阶段 logging-only 工程验收（2026-10-08）

以下两阶段保留当时的测试、同步状态及推进建议；“多 seed 比较”等旧建议由文末已确认的单 seed0 v1 替代，历史推进建议不构成本轮实验授权。

已在交接基准 `e935ff7` 之上实现可配置的只读 Dt 探针，并在 sv1 的 CPU/float32 与 A100 CUDA/bfloat16 完成未改动源码、off、logging-only 三模式的完整连续动作 agent 比较。各后端 618 个共享数组逐位相同，覆盖初始化、原始梯度、采样特征、分支 loss、训练状态和一次更新；五项解析 KL/配置/梯度 unittest 通过。诊断包含 Dt、候选 w、raw rep KL、free-nats 活跃比例及活跃区权重；训练目标仍为 baseline。代码验收提交 `a8ca727`，详细配置、命令、容差、环境、结果、性能样本和风险见 `docs/CODE_PHASE1.md`。这是极小工程 fixture，没有四个真实任务的正式效果结论；第二阶段 Dt/constant/shuffle 尚未启动。

本机和 sv1 的 `/data/Policy_Discrepancy/repo` 已通过 Git bundle 同步到同一提交；截至本记录，`origin/main` 仍停在 `e935ff7`。本机 GitHub HTTPS 不可达，sv1 的 HTTPS 推送缺少凭据且随后连接超时；远端推送待网络/认证恢复后完成。验收产物仅在 sv1 的 `/data/Policy_Discrepancy/runs/verify-logging-phase1-30bb5a6/`。

## 03 第二阶段 overlay 工程验收（2026-10-08）

用户已验收第一阶段，第二阶段完成 Dt、训练前校准后固定的 constant 配置入口和全局 shuffle；条件 shuffle 未实现，等待更强 M2 协议。算法提交 `9e0fb8e`，后续测试脚本提交和服务器实际运行提交详见 `docs/CODE_PHASE2.md` 与本机 Git 历史。gate 只乘在 free-nats 后、原 reduction 前的 rep KL；probe 与 shuffle 不消耗原 Ninjax 随机流。所有 alpha/c/seed 均为工程 fixture，未定科研协议没有被填为正式决定。

sv1 的 CPU/float32、A100 CUDA/bfloat16 完整连续动作 agent fixture 已验证：off、logging、Dt alpha=0、constant c=1、shuffle 全 1 均与未改动 `e935ff7` 基准的 789 个共享数组逐位相同；有效三模式各有 490 个未改动数组逐位相同，活跃 rep loss 与原始梯度按预期变化。9 项 unittest 通过，包括两卡跨 shard 全局置换；两卡完整 shuffle agent 也完成初始化、loss、梯度和一次更新。补充 replay context 开关、内部 reset、不同 imag_last、连续 chunk，以及 free-nats 全不活跃诊断。测试目录为 `/data/Policy_Discrepancy/runs/verify-overlay-phase2-64459ae/`。此验收只证明工程语义，不提供四个真实 DMC 任务上的性能或 M2 结论。

下一步可在已确认四个 clean 仅视觉任务上做**工程集成试跑**，逐次记录确定提交、完整配置、seed、服务器/GPU、环境版本、独立产物目录及 Dt/权重/活跃性/活跃位置作用；先核查运行路径和资源。正式多 seed 比较仍须冻结模型规模、baseline、校准规则及 c 匹配口径、主指标、训练/评估 seed 列表、预算和效果判据。`origin/main` 的推送状态以最新 Git/网络核验为准。

本阶段本机 `git push origin main` 因 GitHub HTTPS connection reset 失败，远端仍为 `e935ff7`；本机提交保留，sv1 已用校验过的 Git bundle 快进至相同确定提交。bundle 路径、SHA-256 和运行提交详见 `docs/CODE_PHASE2.md`；网络恢复后再推送，不在服务器直接改项目源码。

## M2 实验协议 v1 草案（2026-10-08，确认前历史）

本节保留上一轮草案与当时未决状态；下节为本轮确认后的 03 交接状态。此处“待确认”不再适用于已冻结的 v1。

- 已确认设置见本文件开头及 `docs/EXPERIMENTS.md`；候选 baseline 复用与 c 匹配、窗口/预算/评价/指标均明确标为待确认，没有新增 c 数值或效果结论。
- 草案：每任务一次 seed0 logging-only baseline 同时作正式参照和 c 数据来源；10万至30万动作步实际训练 batch 上按活跃位置统计和/计数匹配本任务 c。constant 前冻结，不从 checkpoint 分叉，不随 return 调窗口/alpha/c。该方案仍须用户确认。
- 草案：每组100万训练动作步，repeat=1；初始化和每5万步含终点评估10完整 episode，统一采样；主指标梯形 return AUC/预算，次指标末20%评价点均值，报告三项差值及四任务等权汇总。窗口/端点/计步/随机性/指标规则未自动成为决定。
- 单训练 seed0 只支持探索性判断；episode/时间点不是独立训练重复。D=0、w=1 正常；报告实际作用不足，不自动改 alpha，工程非有限值另报。
- 04 实施前需补齐/核验：训练更新级 source_active_weight_sum/active_count 和动作步 stamp；排除 reset 的动作计数与精确终点；独立训练/评估/report/shuffle RNG；按动作步评价而非墙钟；恰好10完整 episode 的不可变参数评估；完整配置/提交、size50m 资源、checkpoint/统计去重恢复。具体源码事实与要求见协议工程表。
- 当前本机核对 HEAD `4b8b00f54e7f1f5bbaa959391d0c18beba2952f2`，工作区原先干净，main 相对本地 origin/main 记录 ahead25；本轮未核验远端或服务器同步状态，也未推送这批既有提交。仅修改三份指定文档，算法及 METHOD_SPEC 未修改，未启动运行或实验。

下一步由用户确认候选窗口、预算、评价和匹配/汇总规则，再由 03/04 给出并验收工程方案；候选未冻结或工程条件不满足时不启动正式比较。

## M2 实验协议 v1 确认与 03 交接（2026-10-08，本轮）

- 用户已确认：仅训练 seed=0；每任务一次 logging-only baseline，同时是正式 baseline 与 c 匹配来源，不另设校准训练，实际 rep 权重1。
- 窗口为 `[100000,300000)`，按优化更新开始时的累计训练动作步归属；统计每次成功优化更新的有效 loss 位置。raw rep KL>1 时累计 S=sum(active*w)、N=sum(active)，w=1/(1+20D)，c=S/N。正常 replay 重采样保留计数，仅恢复/日志重写导致的同一 update ID 重复记录去重。
- 四任务各计算并冻结 c，constant 运行前冻结；窗口无活跃位置或出现非法值时停止并报告，不填 c=1、不自动调整 alpha=20，不随 return 改窗口/参数。
- 四组从头训练；每组100万训练动作步，包含 replay 预填充、排除 reset 和评估，repeat=1。评价固定为0、5万……100万共21点，每点恰好10完整 episode，固定参数快照、采样动作和独立评价随机流。
- 主指标为梯形 return AUC/100万；末段为80万、85万、90万、95万、100万五点的算术平均。报告 Dt−baseline、Dt−constant、Dt−shuffle，四任务原始 return 等权汇总并保留逐任务。只作单 seed 描述性探索，不设最小增益或等效阈值，不作跨 seed 稳定性或统计等效结论。
- 03 按已确认协议落实统计收集、成功更新/ID/恢复去重、c 冻结产物、非 reset 动作计数、精确网格、固定快照评价、独立 RNG 与完整配置，验收条件见 `docs/EXPERIMENTS.md`。剩余事项是工程实现/验收及运行产物，不是研究参数待选。方法假设仍无正式效果证据。
- 本轮沿用源码核对基准 `4b8b00f54e7f1f5bbaa959391d0c18beba2952f2`，保留上一轮三份未提交草案并纳入本次文档提交；未改算法或 METHOD_SPEC，不推送、不启动实验。交接提交 SHA、提交范围和 Git 状态以完成报告为准，不在文档中写自引用 SHA。

下一步由 03 按本版本实现和验收工程条件；04 核实同提交、完整配置、随机流、环境/GPU 和产物记录后，依据后续运行指令实施。已确认 v1 无需再次等待窗口、预算、指标或阈值选择；本轮不启动实验。

## 03 协议 v1 工程链路验收（2026-10-09）

代码与测试验收到 `b20ffeda07fe741deb1b9ee285174a3e34bcaaf8`。本机实现 `protocol_v1`：预填充也计入的非 reset 训练动作计数、多环境精确停止、动作步评价网格和固定参数快照、独立评价与报告随机流、每成功优化更新的窗口 S/N 收据、冻结 c 产物与检查点审计。正式预设固定 size50m、纯视觉 DMC、seed0、alpha20、100万动作、21 点各 10 完整 episode；缩小预算 fixture 不改变正式口径。具体提交、命令、容差和运行目录见 `docs/CODE_PROTOCOL_V1.md`。

sv1 的 17 项算法/协议单元测试通过。缩小预算连续动作完整 agent 中，报告开关、评价 episode 数及并行环境改变后，baseline 最终参数、训练动作/更新计数和匹配收据逐位一致；Dt/constant/shuffle 也完成动作网格与预算。CPU/float32 与 CUDA/bfloat16 八模式对未改动基准的完整 agent 回归均通过；结果与产物索引见工程报告。中途 checkpoint 能保存累计量和计数，但环境/replay 随机内部状态无法完整恢复，runner 明确拒绝不一致的训练续跑。没有执行正式四任务百万步训练，没有正式 c 或方法效果结论。

下一步由 04 在确定提交、干净服务器工作区及独立运行目录上检查四个真实 size50m DMC 任务的显存、吞吐、日志、作用诊断、评价快照和磁盘，再按已冻结 v1 执行正式比较；中途失败不伪装无缝续跑。本轮 GitHub 远端初次查询曾出现 HTTPS connection reset，重试后确认 `origin/main` 为交接基准 `e935ff7`，随后成功快进推送；服务器也通过可校验 Git bundle 快进同步。最终提交及 bundle 哈希见本轮交接报告。
