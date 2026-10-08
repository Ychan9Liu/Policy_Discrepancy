# 项目状态

更新日期：2026-10-08。

## 科研交接状态

- 01 · Idea 讨论已结束，02 · 方法与机制已接手并完成首轮源码只读核对；研究记录见 `docs/RESEARCH.md`。
- 用户已确认以《DT方法迁移研究说明.docx》为参考，不采用读取该文档前助手提出的假设。原文件路径、参考设计、工作假设、反例和未决问题均已记录在 `docs/RESEARCH.md`。
- 用户于 2026-10-08 确认 M2 主问题、首版连续标准 RSSM 参考方法、四组基础对照及四类局限诊断；随后选定 `hopper_hop`、`quadruped_run`、`quadruped_walk`、`reacher_hard` 四个 DMC 任务，均使用仅视觉和 clean 环境。规范见 `docs/METHOD_SPEC.md`。校准/评估/预算等协议字段仍未定，Dt 的有效性、选择性机制和抗干扰效果均无本项目实验证据。
- 前一次一致性核对仅修改研究、方法规范、实验协议和状态四份文档，并推送 main；当时未改算法代码或启动实验。本次增加 DMC 任务可行性审计、环境 seed 适配和真实环境工程短跑；尚未启动 Dt 正式实验，Dt 验收测试尚无通过记录。

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

- M2 主问题、四个真实 DMC 任务及仅视觉 clean 观测已明确；实际 baseline 配置/提交、校准规则、评价指标、seed 和预算仍待讨论，之后确定 alpha/c 与效果判据。
- 更强 M2 所需条件打乱的分层/区间/边界与退化处理协议仍未定；基础四组及首版验收要求已确定，不受其阻塞。03 在已确定范围内制定并记录配置字段、测试容差、数值处理和 RNG/全局跨 shard 置换等工程方案。
- 真实任务的小模型训练/独立评估短跑已通过；正式 baseline 的模型规模、吞吐、配置、评估和多 seed 可复现性尚未验证。此前 `debug` 假环境短跑及本次极小模型短跑均不代表正式任务效果。
- 源码审计发现原 DMC 适配器未接收和传递环境 seed；现已补上 `DMC(seed)`、视觉预设的 `use_seed` 及 `train_eval` 评估环境索引偏移。`make_env` 的环境初始物理状态重复性已验证，但 `train_eval` 合并路径及完整训练重跑未验证；直接 DMC 渲染有极少量 1 级像素差异。
- 正式实验启动、结果汇总和跨服务器分配脚本尚未实现。
- `sv3` 的根分区仍满；用户已决定当前先跳过处理，实验文件继续放在 `/data`。

## 02 首轮方法核对进展（2026-09-27 历史记录）

- 核对提交：`f0236c7699a87895a8cfde8e13a0d1c198024f75`。接手时仅研究、状态文档有未提交修改；本次保留并增补这两份文档，未提交或推送。
- 已读取原始 DOCX 全文，并只读核对 latent、actor、KL/free-nats、replay/reduction、初始化及梯度路径；具体源码定位见研究文档。这些是源码事实及数学分析，不是运行验证。
- 确认源码中的 bounded_normal 是均值经过 tanh 的普通 Gaussian，执行动作另有裁剪；参考 Dt 对应裁剪前分布。rep KL 当前对 replay context 处理后的 `[B,T]` 直接求均值，没有 padding loss mask。
- 明确 gate 分支梯度隔离不意味着 rep 对 history 的总梯度为零，也不意味着共享参数更新后的 prior 轨迹不变。
- 研究文档新增科学主张层级、常数匹配与条件打乱的证据边界、alpha 校准退化处理、机制反例和 03 候选验收清单；均标明助手建议及未决项，尚未成为用户确认的算法或实验协议。
- 未修改算法代码，未启动 pilot、训练或实验，未运行算法数值/梯度测试。`docs/EXPERIMENTS.md` 继续保持协议待确定。

## 2026-10-08 方法选择与规范

- 已确认：M2 为主问题；首版采用参考定义，仅连续动作与标准 RSSM；至少 baseline、Dt、预先校准固定常数、全局打乱四组，更强 M2 再加条件打乱。
- 已确认：mode 漏检、早期不敏感、干扰误敏感和延迟线索列为局限并诊断，暂不叠加修复。
- 已整理 `docs/METHOD_SPEC.md` v0.1：数学/梯度路径、对照语义、诊断义务以及回退、free-nats、随机流等验收要求；研究决策和实验边界已同步到相应文档。
- 方法实现规范已可交接 03，实现可配置 overlay 与验收测试；任务、观测、校准规则、主指标、训练 seed、预算等实验字段继续按用户指定顺序固定，随后确定 alpha/c 与效果判据，不阻塞可配置方法的实现。
- 2026-10-08 先前检查现有任务列表及 Codex 归档时未找到 03 接收任务，交接材料当时已准备但未发送；指定或创建接收任务属于交接安排，不表示方法或验收仍待确认。本次一致性核对不派发实现任务。
- 先前源码核对基准为 `f0236c7699a87895a8cfde8e13a0d1c198024f75`。本次保留原有未提交文档内容并修正歧义，仅提交上述四份文档、推送 main；实际提交 SHA 和 Git 状态以完成报告及 Git 记录为准。没有算法修改或实验运行，验收条件尚未实际通过。

下一步选择正式模型规模与 baseline 配置/提交，核实吞吐和显存；随后补齐 logging-only 校准、评价和预算字段。`train_eval` 合并路径与完整训练重跑的重复性仍待验证。源码核对不能替代 identity/gradient 测试；方法假设和效果必须由后续结果检验，正式比较不因文档规范自动启动。

## 03 第一阶段 logging-only 工程验收（2026-10-08）

已在交接基准 `e935ff7` 之上实现可配置的只读 Dt 探针，并在 sv1 的 CPU/float32 与 A100 CUDA/bfloat16 完成未改动源码、off、logging-only 三模式的完整连续动作 agent 比较。各后端 618 个共享数组逐位相同，覆盖初始化、原始梯度、采样特征、分支 loss、训练状态和一次更新；五项解析 KL/配置/梯度 unittest 通过。诊断包含 Dt、候选 w、raw rep KL、free-nats 活跃比例及活跃区权重；训练目标仍为 baseline。代码验收提交 `a8ca727`，详细配置、命令、容差、环境、结果、性能样本和风险见 `docs/CODE_PHASE1.md`。这是极小工程 fixture，没有四个真实任务的正式效果结论；第二阶段 Dt/constant/shuffle 尚未启动。
