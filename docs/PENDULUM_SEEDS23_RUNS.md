# Pendulum seed2/3 原参数复验

**2026-11-04资源安排更新**：用户授权使用其余已分配空闲GPU并行测试。原两baseline进程/训练版本保留，尚未启动的两条Dt改为sv3 GPU5/6，GPU7保留；新卡验收、旧槽位防重复及新合并索引见[并行执行记录](PENDULUM_SEEDS23_PARALLEL.md)。下文同卡串行方案是原启动时安排，不再作为新Dt的分配依据；科学配置与原始证据不改。

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

### 2026-11-03实际版本与已通过检查

实现、服务器CPU测试、设备探针及CUDA短程使用同一完整SHA：`c5f86ca1a16648f57db3c3abfa2584ac560495ca`。sv3独立干净worktree为`/data/Policy_Discrepancy/checkouts/pendulum-seeds23-c5f86ca1a166`；Git bundle SHA256为`a244eb8df98f97fd724b2bc8e4b995c7c76a8c85514e68236df0db532febf98e`，校验后Git fetch/worktree部署。原`/data/Policy_Discrepancy/repo`保持`d8997f64a301bfdcfe08758faeae8957a7395766`不动，没有同步sv1/sv2或推送GitHub。

与原pendulum baseline训练SHA `dbd28b6703d51d61e5d3485dbfea6c413075ea85`和原Dt训练SHA `d8997f64a301bfdcfe08758faeae8957a7395766`分别逐Git blob比对：`dreamerv3/`和`embodied/`各59个文件字节一致，仅configs/main/protocol_scope/freeze_c四个入口及范围文件不同。完整文件哈希见本机`source-equivalence.json`（SHA256 `438211d7b6d80624a800e98c531c249ca60c291475bec34ab15682d5e0936ba7`）。配置仍逐字段仅script/seed/logdir改变，未把新训练宣称为与旧运行完全相同的数值轨迹。

23:46服务器18个测试模块、129项测试全部通过，耗时85.624秒，包含新的8项seed2/3测试；此前本机因缺portal未运行的冻结测试在这里真实执行通过。回归日志SHA256 `6a9f83ef3d91ee8f23f76e88015c6fd055d3401099f52c19dffd09608c416957`。随后三个CPU float32完整agent工程变体（不报告/有报告/评价episode从2改为1）均完成；两组对比的12训练动作、9更新、参数哈希、训练随机流计数及全部成功收据精确一致。它们只证明工程隔离性，不当成真实任务学习效果。

23:47:32实际CUDA计算和EGL渲染探针通过，PID同时显示C+G在指定物理卡；映射不是按编号猜测：

|物理GPU|CUDA_VISIBLE_DEVICES|进程内CUDA编号|实测MUJOCO_EGL_DEVICE_ID|
|---:|---|---:|---:|
|1|GPU-5b1ebede-99ec-d080-29b0-3c04011dc663|0|1|
|4|GPU-9c070d5a-ac68-4c26-9522-a790ad874b23|0|4|

正式preset/seed2的线上logger在零训练动作、零agent更新下写入并读回成功；[PD_1 logger验收](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/logger-pendulum-seeds23-c5f86ca1a166)为工程记录，不是第五条正式实验。环境仍为现有dreamer：Python3.11.16、jax/jaxlib0.4.33、numpy1.26.4、dm-control1.0.48、mujoco3.15.0、wandb0.30.0；没有安装、更换依赖。依赖锁SHA256 `cd50c7936690f9adbc5c35f99b3df2f7a977cef3222582f988979b11f65b6f22`，完整版本见deployment.json。

已记录执行辅助脚本`checks.py`/`start-checks.py`/`start-formal.py`、每条命令与helper哈希，位于本机证据目录和服务器engineering根；辅助脚本仅调度已提交模块，不修改服务器源码。首次工程监督PID3679126，23:45:11启动。CPU父进程隐藏GPU，正式子进程按绑定设置单个UUID并清除继承的CPU/XLA/动态库覆盖；绑定必须同时具备同SHA CPU、四条CUDA、在线logger和实际同卡证据。

23:55:34四条真实CUDA短程全部退出0并通过原始产物审计：各4098训练动作、517连续成功更新、3点评价×1完整episode、2次report，合计2068更新/12点评价/8报告隔离证据。seed2/3的baseline工程匹配有效位置数309784/318268；Dt不收集匹配c，计数0符合原规则。工程结束后进程退出、两卡释放。日志中可选TensorFlow profiler hook不可用未中断运行，实际收据、终点及全部审计通过；没有为此改依赖或放宽检查。

## 输出与当前状态

正式根：`/data/Policy_Discrepancy/runs/formal-m2-pendulum-seeds23-20261103-sv3/`，每seed下`logging-attempt01`/`dt-attempt01`。工程根：`/data/Policy_Discrepancy/runs/engineering-pendulum-seeds23-20261103-sv3/`。本机证据：`analysis/outputs/pendulum-seeds23-start-20261103/`。确定提交通过Git同步至sv3独立checkout，避免活动checkout随后被其他文档提交改变；只同步sv3，不覆盖原训练目录或本机其他chat草稿。

23:56:14正式启动前复查：五张指定卡均0MiB、无C/G PID，/data余340700663808字节，超过240GB门槛；绑定前、监督器启动前、每个子进程启动前又分别独占核查。23:56:37正式监督器PID3690542启动，23:56:55确认PID及创建时间一致、两个训练子进程存活。绑定计划SHA256 `409518544423fa9dd520508b38f80da3aaabb9f6c6a07f7b5194ca040397a8a5`；训练和审计代码继续固定为`c5f86ca1a16648f57db3c3abfa2584ac560495ca`，不会随本机后续文档提交移动。

|sv3物理GPU|seed|组|尝试/相对正式根目录|PID|当前状态|
|---:|---:|---|---|---:|---|
|1|2|baseline / logging|seed2/logging-attempt01|3690581|运行中|
|1|2|PPD-KL / dt|seed2/dt-attempt01|—|排队，等待自身baseline完整审计|
|4|3|baseline / logging|seed3/logging-attempt01|3690588|运行中|
|4|3|PPD-KL / dt|seed3/dt-attempt01|—|排队，等待自身baseline完整审计|

两条Dt没有启动PID/manifest或训练产物，不能写成四条均已训练。GPU5/6/7保留。本批正式训练尚未完成，正式c、最终return指标和失败是否跨seed重现均待终点审计；进程退出或W&B Finished不能替代产物验收。失败保留证据，新尝试不得覆盖旧目录；不支持缺环境/replay的无缝恢复。

### 2026-11-04 00:00:40启动验收通过

两条baseline均通过实际启动检查：完整manifest与绑定配置/seed/SHA完全一致，设备为cuda:0；0点各10个完整episode，episode ID、派生环境seed、固定0动作/0更新快照及完成文件正确，逐episode原始return重新求均值一致。这里只核对已有episode累计值，没有声称独立重放评价逐步reward。两条初始平均return均为0，不是训练结果判定。

seed2/seed3当次分别读取458/461条连续成功更新，末条开始动作为3860/3872，invalid_count全0。逐卡所有计算/图形PID均属于对应训练进程后代，环境白名单与UUID/EGL绑定一致，无继承动态库、PYTHONPATH或XLA覆盖；各卡仅本项目作业。线上两条均running，full config精确一致，startup/exact_values_json及其SHA256回读成功。此时/data余338569392128字节；完整证据`formal-startup-audit.json`。

- [seed2 baseline · PD_1](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-pendulum-seeds23-20261103-sv3-pendulum-seed2-logging-01)
- [seed3 baseline · PD_1](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-pendulum-seeds23-20261103-sv3-pendulum-seed3-logging-01)

两条PPD-KL仍为已授权队列，不提前创建线上正式run。其预定ID分别为`formal-m2-pendulum-seeds23-20261103-sv3-pendulum-seed2-dt-01`与`formal-m2-pendulum-seeds23-20261103-sv3-pendulum-seed3-dt-01`。实际启动后以RUN_INDEX/manifest/线上读回为准。当前不需工程修复或重跑；最终收益、跨seed重复性和正式c仍待完整训练。

后台监督器每30秒检查已授权队列，逐条产物审计通过才放行同seed的Dt；占用冲突、空间不足或前驱失败会记录blocked，不自动换卡、共享、删除数据或等待外来作业空闲重试。四条全部通过后自动运行`analysis.return_metrics --profile pendulum-seeds23 --require-complete`，保存逐seed指标/配对差值及W&B精确JSON/hash回读，完成文件位于正式根`final-audit-attempt01/`。批次`accepted`必须等此收尾完成，训练中没有完整四条比较。

本次实际启动命令（由本机SSH执行，完整环境/子命令在check-commands.json及各execution-initial.json）：

```text
/data/Policy_Discrepancy/envs/dreamer/bin/python /data/Policy_Discrepancy/runs/engineering-pendulum-seeds23-20261103-sv3/start-checks.py
/data/Policy_Discrepancy/envs/dreamer/bin/python /data/Policy_Discrepancy/runs/engineering-pendulum-seeds23-20261103-sv3/start-formal.py
```

上述目录已经使用，不要重跑启动命令覆盖或重复提交。监督器执行的正式命令为`python -u -m scripts.run_pendulum_seeds23 run --plan <engineering根>/bound-plan.json`；后续只读检查实际RUN_INDEX、进程身份及原始日志。若需重新审计，使用`audit --plan <engineering根>/bound-plan.json --output <新的审计目录>`，不会恢复或重新启动训练。

首选继续当前chat承接03执行/监测职责，完成两条baseline→两条Dt及终点审计，再按05职责分析失败是否重复。无需创建新chat或再请求已批准范围的启动许可。全部四条accepted需百万动作、连续成功更新、21×10、快照/原始指标/各seed c及W&B精确读回。保持seed0/1来源和本批seed2/3分开，保留低回报seed，评价episode不作为独立训练seed；新alpha、奖励、损失或控制组选择仍由02/用户确认。不自动创建chat或发送消息。

最终交付SHA、相对实测版本的差异、本机小证据hash及可复制交接见`analysis/outputs/pendulum-seeds23-start-20261103/DELIVERY.json`与`HANDOFF.md`。最终提交仅更新文档，服务器活动checkout继续固定实测SHA，不为补写文档重测或迁移训练。

00:03:06小证据快照已下载并逐项验证220个文件的大小/SHA256：`small-evidence-20261103T160306Z.tar.gz`为200795字节，SHA256 `44deddd337a11ad5446bcbf7acb2d9a0c22a4632b68e2b0fdde8813938c3de0f`。本机解包目录`mirror-startup/`含原服务器路径、完整配置/索引、测试日志、工程/启动审计与快照小标记；`SOURCE_HASHES.json`的SHA256为`36130ee9700a8fa710345ffa4fa5dd9366d6b9c730c95a01f46764b9a59c561a`。这是运行中的时间截面，不是终点证据；大checkpoint和replay留服务器。

00:04:31最后复查：监督器及两条baseline PID存活，最新收据开始动作seed2=9892、seed3=9752，对应更新ID1965/1930、末条invalid0；两条Dt仍排队。GPU1/4各17个C/G PID、约8002MiB，GPU5/6/7各1MiB但无C/G进程；/data余338510004224字节。本批仍占用GPU1/4，没有释放正式训练卡；其余三卡保留。快照`formal-final-check.json`随DELIVERY记录hash，不把此时间截面当持续独占保证。
