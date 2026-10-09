# 04 · PD_1 正式 baseline / Dt 执行记录

日期：2026-10-09，Asia/Shanghai。**状态：八条运行中，尚未完成正式验收。** 用户已授权sv2物理GPU0–3四任务logging baseline并生成正式c/return，随后明确prealloc=False、JSONL+W&B、无scope、project PD_1，再追加直接启动GPU4–7的四任务Dt。constant/shuffle尚未授权；不等待baseline c启动Dt，不复用baseline训练状态。

## 实际安排与现场核查

| 物理GPU | 任务 | 组 | 训练PID | CUDA_VISIBLE_DEVICES UUID | EGL物理枚举索引 |
| --- | --- | --- | --- | --- | --- |
| 0 | dmc_hopper_hop | logging baseline | 2783723 | GPU-76344ce3-10a1-952d-68f5-7720a3cda667 | 0 |
| 1 | dmc_quadruped_run | logging baseline | 2783734 | GPU-1e27fd5e-4231-926a-754f-e7d4c37d41ab | 1 |
| 2 | dmc_quadruped_walk | logging baseline | 2783748 | GPU-24608f8d-a512-70b2-8578-45945b57bc35 | 2 |
| 3 | dmc_reacher_hard | logging baseline | 2790186 | GPU-92630efa-be4e-7074-d29a-ccff0ad7f801 | 3 |
| 4 | dmc_hopper_hop | Dt | 2801952 | GPU-7bf636ec-fa03-aa3f-7314-dd803c1658be | 4 |
| 5 | dmc_quadruped_run | Dt | 2801971 | GPU-ac733835-08ee-7a17-a2fa-1e49d62d385b | 5 |
| 6 | dmc_quadruped_walk | Dt | 2802262 | GPU-548be1c8-86f0-b7a2-de92-5c455b12221b | 6 |
| 7 | dmc_reacher_hard | Dt | 2802550 | GPU-4cbf5b69-adb3-5210-2396-ca5d4e400f26 | 7 |

全部sv2/lyg0360、A100-SXM4-80GB、dreamer Python3.11.16。CUDA_DEVICE_ORDER=PCI_BUS_ID、每进程仅logical CUDA0，MUJOCO_GL/PYOPENGL_PLATFORM=egl。训练和渲染的实际PID→UUID均核对；子环境图形PID按祖先训练PID分类，没有将未知进程猜成本项目。22:55:17开始的八条现场审计无未知/错卡进程，八条W&B配置与批准值一致，在线状态均running。该时间是快照，后续必须重查，不声称卡已经释放。

baseline在22:19:50空闲复查后启动；前三条22:20启动，reacher22:28启动。Dt启动前22:43:41，GPU0–3已有本项目baseline，4–7进程表/pmon无作业、显存各1MiB；/data可用2144209166336 bytes。22:48 GPU4–7各自单进程CUDA数组/EGL单帧探针通过，训练动作/agent更新为0，随后才启动Dt。探针及真实训练共同提供设备证据，不能把单帧探针写成size50m训练验收。

## 完整配置、W&B和产物

科学设置沿用EXPERIMENTS C/C.1：M2四个clean仅视觉DMC、size50m、image=True/proprio=False、seed0、repeat1、alpha20、free_nats1、dyn/rep外层1/0.1、ac_grads=False、reward_grad/repval_grad=True。batch16×64、replay_context1、16训练env/1串行评价env、train_ratio256、CUDA/bfloat16、单逻辑卡，其他字段沿用defaults+m2_v1。两组共同prealloc=False、logger=[jsonl,wandb]、无scope；同任务完整配置比较仅mode/logdir不同。

每条1000000执行训练动作，包含预填充、排除reset/评价；21点0..100万、间隔5万，每点10完整episode、固定快照、采样动作、独立评价流。baseline实际rep权重1、候选w同用alpha20；Dt使用冻结算法权重。baseline完整审计后，从成功更新开始步[100000,300000)的活跃raw rep KL>1位置累计S/N冻结各任务c，正常replay重采样保留，仅同update ID重复统计去重；N=0/非法停止，不填c1或改alpha。Dt不需要c。正式指标遵守原始return、固定分母100万梯形AUC、80/85/90/95/100万尾部五点、四任务等权；其他未运行组及其差值保持null。

- baseline根B：`/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-seed0-20261009/<task>/logging-attempt01`。
- Dt根D：`/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-dt-seed0-20261009/<task>/dt-attempt01`。
- 证据/监督根P：`/data/Policy_Discrepancy/runs/wandb-pd1-setup-20261009/`。
- 全配置与计划：sv2 repo `analysis/outputs/formal-baseline-pd1-20261009-48d71f1/`、`formal-dt-pd1-20261009-48d71f1/`；全部由生产Flags解析并按冻结checker核对，分别保存JSON/YAML/argv、UUID/EGL及seed/预算。实际展开配置另在各运行config.yaml/manifest；完整命令/允许列环境在execution-initial.json。
- 实际索引：B、D各自RUN_INDEX.json；P/all-eight-RUN_INDEX.json是22:55快照。本机小证据/完整计划镜像为 `analysis/outputs/formal-baseline-pd1-20261009/`，原始replay/参数留服务器，不提交。
- W&B project实际默认entity为 `ychan9liu-tsinghua-university`，项目 https://wandb.ai/ychan9liu-tsinghua-university/PD_1 。八个run链接在实际index及P/all-eight-startup-audit.json；baseline ID形如`formal-m2-v1-pd1-seed0-20261009-<task>-logging-01`，Dt形如`formal-m2-v1-pd1-dt-seed0-20261009-<task>-dt-01`。工程logger检查run单独分组，不当作训练重复。完整config已传入生产W&B logger，日志本地文件位于对应logdir；没有密钥输出或提交。

## 实际SHA、同步和工程处理

- 八条正式训练SHA全为 `48d71f1541721208322442f5a1d5977124f3e888`；此版本相比之前运行路径仅W&BOutput新增完整config和dir参数，另增加外部监督脚本、补充依赖锁与文档，不改变agent/runner/梯度/RNG/协议。
- 首次baseline外层控制器在GPU3前退出：把memory.used必须0当作空闲条件，现场1MiB但无实际作业。三个已启动训练独立存活，修复通过XML所有类型进程+pmon+UUID判断作业、照实记录显存，不修改数值验收容差。修复源 `3a6f3aa17f5ef7b408c7ecf9aa5b2bfd59f10d2b` 接管监测，严格核对已有PID命令/创建时间；不是runner/checkpoint续跑，没有训练重跑。原失败日志P/formal-supervisor.log保留。
- 接管后的baseline监督PID2790099；三个孤儿训练的原始exit code无法回收，结束后必须留null，不能伪造exit0；终点checkpoint/收据/快照/return与W&B finished单独验收。初始前三条22:20–22:28的外层资源采样存在缺口，最终显存只能报告已采样峰值。
- Dt监督源 `9e4a4ea5415858921373d5ae1456d10c18b168fc`，PID2801820，仅允许已授权logging/0或dt/4，启动后逐运行保存index；四项作业拒绝测试CPU通过。GPU4–7真实单帧探针源 `a4a0979e7bce536d38a2cc26286fbc729c68a25e`；这些源对象经bundle verify/hash传入服务器，导出到P执行，服务器HEAD始终48d71f1且工作区干净，没有修改服务器checkout或影响已有训练。
- baseline依赖收尾源 `0ec636a7036546bf6d83724d84a33d114fecbc47`，CPU等待进程2795089：原freeze_c→原return统计→检查四baseline完整性/收据/c哈希与S/N→W&B finished及summary上传验证。目标B/baseline-completion-audit.json，失败另保留failed文件。四组全比较仍不完整。
- baseline/Dt合并收尾源 `520dc0b9a540a83a4e62f6767affd6e97d75ecb4`，CPU等待进程2808914：等待两组完成后在repo `analysis/outputs/formal-baseline-dt-pd1-20261009-final/` 生成新index/六表/审计；等待baseline c上传验收后，再写各run完整return及Dt−baseline差值，避免并发summary写入。constant/shuffle保持缺失，不补值、不重加权。此收尾程序已部署/语法检查与独立工程summary写读验证通过，**真实百万步终点路径尚未触发，不当作已通过**。
- 仅sv2补充wandb0.30.0及十项新包，metadata确认全部已有包版本未改变，pip check通过；新增锁env/requirements-wandb-pd1.lock，原共用锁保留。首次用原freeze作constraints因Conda构建路径失败，P/install.log保留，改实际包名版本约束后成功。未修改/删除base-py311，sv1/sv3未安装/同步。源码对象与外部脚本同步范围、最终文档SHA见完成回复；不要把外部监督SHA当训练SHA。
- JAX日志中的TensorFlow profiler trace不可导入是非致命提示，已有数千成功更新；未为此安装TensorFlow或改变训练配置。驱动低于PTX编译器版本使并行编译关闭，保留已知耗时边界，不升级驱动。

## 已验证范围与剩余条件

22:55快照：baseline hopper/run/walk更新12120/12033/12018，已保存0/50000两点评价；reacher更新9743、41000动作、仅0点。Dt四任务更新1399/1286/1326/1380，最新更新开始动作7624/7172/7332/7548，均0点评价。所有当前持久化receipt ID连续且invalid_count0，每个已有评价点恰好10条显式完整episode；W&B配置/状态和计算/渲染设备验证通过。这是启动和早期运行质量证据，非正式全预算验收，更不是方法有效。

八卡仍有本项目作业，没有释放任何已分配卡。正式c未产生、正式AUC/tail/Dt差值待完整运行；工程c不可复用。后续不能从不完整checkpoint续跑，失败保留证据并在新尝试目录从头重跑，不择优删选；同资源/同配置的工程修复可在已有授权内继续。收尾失败不意味着可以修改科学参数或放宽检查。

首选下一任务继续现有04：监测八条已授权运行，核对B/D supervisor-final、baseline c及依赖收尾markers，独立审计完整21点/10episode/实际快照、收据/动作预算/随机流来源、c哈希、W&B上传、资源释放，更新正式结果索引。constant/shuffle需用户追加授权后才启动；正式c有效也是constant依赖。无需新chat，不自动发消息。

可复制prompt（交接完整最终SHA以完成回复为准）：

```text
继续现有04，项目根D:\program\PD\dmr3，核对最终Git/修改不回退。
必读AGENTS.md、docs/CHATS.md、docs/STATUS.md、docs/FORMAL_BASELINE_DT_PD1.md、
docs/FORMAL_BASELINES.md、docs/EXPERIMENTS.md C/C.1、docs/CODE_PROTOCOL_V1.md、analysis/README.md。
用户已授权sv2 GPU0–3四logging baseline、GPU4–7四Dt；任务顺序hopper/run/walk/reacher。
八条训练SHA48d71f1541721208322442f5a1d5977124f3e888；服务器HEAD保持该版本，外部监督源另记。
prealloc=False、JSONL+W&B PD_1、无scope；M2 clean视觉size50m seed0 repeat1 alpha20，
free_nats1、dyn/rep1/0.1、ac_grads=false、reward_grad/repval_grad=true不改变。
各100万训练动作含预填充排除reset/评价，21点×10完整episode、固定快照/采样独立评价流。
baseline完整后按成功更新开始动作[100000,300000)的活跃S/N冻结c，无活跃/非法停止；
不填c1、不改alpha、不加seed；Dt从头训练不需c；constant/shuffle未授权。
按本报告B/D/P目录和PID读取当前进度/依赖审计，复查实际GPU占用与CUDA/EGL映射。
完成八条及c/return/Dt差值审计、W&B summary/上传、资源释放检查，更新报告/STATUS/EXPERIMENTS。
不完整checkpoint拒绝训练续跑，失败保留、新attempt重跑；孤儿PID原exit code必须null。
源码修复本机提交后同步确定版本，不改运行中源码/科学语义；新表/审计用outputs新目录，
大产物留服务器，不覆盖旧证据、不自动创建chat或发送消息。完成按AGENTS完整交接。
```
