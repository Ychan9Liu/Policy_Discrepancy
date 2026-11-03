# Pendulum seed2/3 原参数复验

2026-11-03，用户已明确回复“按此方案准备并启动四条复验”。这撤销本批此前的“先不启动”，确认下面的四条范围、同卡串行安排和先工程验收再正式启动。无需再请求启动确认；本chat连续承接已授权的工程、执行、审计与分析，不以旧03/05分工中断。先前诊断与未批准候选记录保留为历史。

## 已确认配置与资源

|服务器 / 物理GPU|训练seed|任务|执行顺序|
|---|---:|---|---|
|sv3 / GPU1|2|dmc_pendulum_swingup|logging baseline完整审计通过 → PPD-KL（dt）|
|sv3 / GPU4|3|dmc_pendulum_swingup|logging baseline完整审计通过 → PPD-KL（dt）|

sv3 GPU5/6/7保留，不增加实验、不自动换卡或共享；指定卡如被外来任务占用，停止新启动并报告，不自动等空闲重试。pendulum seed1由既有vision20计划负责，不在本批重复或迁移。hopper seed1/2/3已完成，本批不重跑。

独立范围`M2-v1-pendulum-seeds23-baseline-dt`，preset `m2_pendulum_seeds23`，script `protocol_pendulum_seeds23`。仅允许上述任务、整数seed2/3、logging/dt；原v1、DMC16、hopper和vision20入口不放宽。完整配置逐字段对照原pendulum seed0相应组，仅script、seed、logdir改变，其余包括优化器全部不变。

- clean、仅64×64 RGB视觉、size50m、repeat1；每条从头创建模型/optimizer/replay/环境，不续跑或分叉seed0。
- baseline实际rep权重1；PPD-KL为既有dt，alpha20；free nats1，dyn/rep外层系数1/.1；ac_grads=false、reward_grad=true、repval_grad=true、repval_loss=true。不改reward、熵系数、模型、门控公式或随机流派生。
- batch16×64、replay_context1、16训练环境、train_ratio256、CUDA/bfloat16、prealloc=False；正式report关闭。
- 每条1000000训练动作包含预填充、排除reset/评价。0..100万每5万固定参数快照，恰好10完整episode，动作采样及独立评价随机流沿用原规则。四条合计400万训练动作、840评价episode；pendulum标准1000动作episode预计84万评价动作另计。
- 每seed从自身baseline成功更新开始动作`[100000,300000)`、raw rep KL>1有效位置累计S/N冻结c；完整收据、非法值/无活跃停止规则不变，不使用工程c或seed0的c。本批只跑两组，c用于审计，不新增constant/shuffle；目前新c尚未产生，baseline终点审计后报告数值与来源。
- JSONL + 在线W&B，`ychan9liu-tsinghua-university/PD_1`，独立run ID、resume=never。逐episode原始return均值、21点梯形AUC/100万、80/85/90/95/100万五点均值；单独`pendulum-seeds23` profile保留逐seed配对差值，两seed汇总只是本批任务均值，不与seed0/1自动合并、不作显著性或位置机制结论。

23:29:14北京时间初查：sv3主机lyg0326，物理1/4/5/6/7均A100-SXM4-80GB，显存占用0MiB、逐卡XML及pmon无C/G PID；/data余349321277440字节。按60GB/条预留四条共240GB，不能只按同时运行两条预留120GB。GPU0/2/3有其他作业，不使用、不干扰。该快照非预约，工程探针/正式启动/串行后继均重新核查占用。

## 工程实现与验收

新增严格scope、preset和既有runner分派，扩展正式c身份核对及独立统计profile；原runner计步/评价/RNG、agent/rep_probe/RSSM/optimizer/DMC实现保持原文件。`prepare_pendulum_seeds23.py`只生成确定SHA的四配置/队列，未绑定时无launch_argv；与原seed0配置逐项比较。`run_pendulum_seeds23.py`提供独占设备检查、CPU/在线logger/四条真实CUDA工程证据绑定、两条同卡串行队列、每条完整审计和最终W&B精确摘要收尾。

工程产物必须按seed和组分目录：相同任务名不能覆盖seed2/3。四条真实size50m CUDA/bfloat16短程使用4098动作、2049评价间隔、每点1episode、[2200,3600)工程匹配窗口、两次只读report；这些只用于工程验收，不进入正式指标、c或训练预算。各自需517连续成功更新、3个固定评价点及报告/评价训练状态隔离证据。CPU检查、实际CUDA/EGL同卡probe和线上logger均须同一确定提交。

本机已有33项范围/队列/指标回归通过；新增正式c函数测试在最小本机venv缺portal依赖，未用stub或跳过宣布通过，转至已有完整dreamer环境执行，作为绑定门槛的一部分。实际服务器测试、工程与训练SHA/命令/设备映射以本批证据及后续状态为准；测试fixture不代表真实pendulum验收。

## 输出与当前状态

正式根：`/data/Policy_Discrepancy/runs/formal-m2-pendulum-seeds23-20261103-sv3/`，每seed下`logging-attempt01`/`dt-attempt01`。工程根：`/data/Policy_Discrepancy/runs/engineering-pendulum-seeds23-20261103-sv3/`。本机证据：`analysis/outputs/pendulum-seeds23-start-20261103/`。确定提交通过Git同步至sv3独立checkout，避免活动checkout随后被其他文档提交改变；只同步sv3，不覆盖原训练目录或本机其他chat草稿。

本报告初稿时：实现和本机检查进行中，四条正式运行均未启动，尚未宣称真实CUDA/线上logger验收完成。随后启动、PID、终点与实际c追加至本报告/STATUS及实际RUN_INDEX；进程退出或W&B Finished不能替代产物验收。失败保留证据，新尝试不得覆盖旧目录；不支持缺环境/replay的无缝恢复。

首选继续本chat完成同SHA服务器CPU/四条真实CUDA验收、绑定并按已授权范围启动，持续监测至完整审计和比较，不再问一次启动许可。全部四条accepted需百万动作、连续成功更新、21×10、快照/原始指标/各seed c及W&B精确读回；完成再分析失败是否重复及后续科学问题。新干预仍由02/用户确认，不把工程通过写成方法有效。不自动创建chat或发送消息。
