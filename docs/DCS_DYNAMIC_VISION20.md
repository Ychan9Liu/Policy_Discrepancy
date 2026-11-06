# Vision20 dynamic baseline / Dt：准备记录

状态时间：2026-11-06 11:04（北京时间）。本轮目标为20任务各baseline、原Dt一条，共40条；资源为用户指定的sv2物理GPU0–7。当前**0条正式启动、0条实际排队**，没有创建正式训练控制器。下面的队列矩阵和缩小预算均为已测试的准备能力，不是已启动记录。

## 已确认与待补齐

- 用户明确：全20任务、dynamic背景、baseline/Dt、sv2全部八卡、先挂八条并排八条；本次资源包括GPU6，旧批次“不用GPU6”不适用于新授权。
- Dt指原PPD-KL，`agent.rep_probe.mode=dt`，不是DtL；baseline为logging-only，实际训练权重1。DtLatch关闭。原clean批次和已完成结果不被覆盖或重新分类。
- 共同继承项：size50m、64×64 RGB、仅视觉、每条100万实际训练动作、repeat=1；alpha20、free-nats1、dyn/rep=1/0.1、ac_grads=false、reward_grad/repval_grad/repval_loss=true；16训练环境、batch16×64、context1、train_ratio256、CUDA/bfloat16、prealloc=false；评价0至100万每5万一次、每点10完整episode、固定快照及采样动作；匹配窗口[100000,300000)，成功更新开始动作步归属、有效loss位置S/N。此轮不运行constant/shuffle，不把背景问题改写成新的算法定义。
- 用户本轮答复已确认：训练seed0；DAVIS train 60视频训练、独立val 30视频评价，video_alpha=1，14任务上游地面规则，另6任务地面alpha=0.3。背景独立MT19937沿用现有派生实现，根标识-1、cache_size=2、num_videos=-1；物理环境/动作/agent RNG沿用原seed0映射。
- 仍待用户答复：40条自动补入与仅先执行16条的范围；存储安排。正式控制器尚未启动。
- 若选择自动补入，代码最多允许8条运行、8条就绪，其余24条初始为backlog；若只执行前16条，另外24条仅准备，不能自动启动。

启动前的配置声明包含seed、全部背景字段、每任务地面alpha和视频划分，并以SHA256绑定到plan与实际config。科学配置已由本轮两项答复补齐，可以执行同配置工程验收；正式队列范围和容量仍按AGENTS第2节待答复，不是重新索取已授予的服务器启动权限。

## 资源与准备产物

sv2入口仍为既有SSH配置，hostname `lyg0360`。开始时SSH多次在认证前关闭，10:41恢复一次；后续以持续SSH会话完成准备。不能据成功连接推断今后入口稳定。

10:41八卡均空闲，/data剩147,643,842,560字节。准备素材和测试后，服务器10:57:32 UTC+8（本机10:58）余量145,960,243,200字节，GPU0–7均0MiB、无C/G进程。零动作设备检查进程已结束，没有本轮GPU任务遗留。

最终准备快照为服务器11:03:29，145,947,152,384字节可用，八卡仍均0MiB且无C/G进程；该时间和原始输出保存在`PREPARATION.json`。

既有size50m每条产物约23–25GB，八条完整运行约200GB、40条累计约1TB；这是容量规划估计，不是本轮实测用量。控制器保留60GB/卡的保守首批门槛，八卡合计480GB；用户可以明确接受首批缺口，但该例外不自动删除历史产物，也不扩展至后继启动：后继不足60GB时继续等待，运行中余量低于5GB停止本控制器自己的子进程并保留失败证据。147GB不足以承诺八条完整跑完；旧DtL三条的容量例外没有自动套用本轮。

物理GPU与本次实际设备检查：

| 物理GPU | UUID | EGL index / 进程内CUDA |
| --- | --- | --- |
| 0 | GPU-76344ce3-10a1-952d-68f5-7720a3cda667 | 0 / 0 |
| 1 | GPU-1e27fd5e-4231-926a-754f-e7d4c37d41ab | 1 / 0 |
| 2 | GPU-24608f8d-a512-70b2-8578-45945b57bc35 | 2 / 0 |
| 3 | GPU-92630efa-be4e-7074-d29a-ccff0ad7f801 | 3 / 0 |
| 4 | GPU-7bf636ec-fa03-aa3f-7314-dd803c1658be | 4 / 0 |
| 5 | GPU-ac733835-08ee-7a17-a2fa-1e49d62d385b | 5 / 0 |
| 6 | GPU-548be1c8-86f0-b7a2-de92-5c455b12221b | 6 / 0 |
| 7 | GPU-4cbf5b69-adb3-5210-2396-ca5d4e400f26 | 7 / 0 |

每条进程仅暴露对应UUID，`CUDA_DEVICE_ORDER=PCI_BUS_ID`；CUDA运算和MuJoCo球体渲染的实际PID经compute inventory和pmon同时核对，训练动作/更新均为0。该检查**不是dynamic完整agent验收**。正式启动仍逐次重查占用，不把本表当实时资源状态。

服务器准备根：`/data/Policy_Discrepancy/runs/engineering-dcs-dynamic-preparation-20261106/`。

- 素材：`assets/DAVIS/JPEGImages/480p/`，官方归档832,766,765字节，SHA256 `e3d0b5b77c3d031b000a19e0e25e3e2cac65d183755601bc2cf066df1a2aa492`；6208张RGB均逐图解码和hash，60训练视频4209帧、30验证视频1999帧，集合无交集。`assets/download-extraction.json`、`assets-audit/{train-assets,val-assets,environment-plan}.json`保留完整收据；其中`authorized_to_train=false`仍正确。
- 本机证据：`analysis/outputs/dcs-dynamic-vision20-20261106/`，包含回归日志、同步bundle与小产物镜像；大素材不入Git。
- 旧DtL三条只读核查显示`supervisor-final.json passed=true`、三条accepted、exit0；不修改或重新训练这些作业。其目录、协议和历史证据仍见`DTL_THREE_SEED0.md`。

## 实现语义与边界

新增`dreamerv3/dcs_protocol.py`读取独立JSON声明并校验SHA256、20任务/两组/seed、显式训练及评价背景；`m2_dcs_dynamic`预设seed=-1、声明为空，不能靠默认值意外启动。`main.py`只给新`protocol_dcs_dynamic`放行；旧`protocol_*`继续拒绝含噪，dynamic路径也拒绝clean override及未声明强度。

`embodied/run/dcs_dynamic.py`复用**未修改**的`protocol_v1.run`。新factory将原seed32派生出的16个训练环境seed和21×10个评价seed逐一映射到训练或评价素材；检查环境seed及背景seed碰撞，拒绝未知seed或错误index，不额外调用任何随机数生成器。映射可序列化，供原parallel driver使用。每次运行在agent创建前hash实际素材、检查视频不相交，写`dcs_environment.json`。评价仍走原固定快照、完整episode和独立动作RNG路径，训练目标、free-nats顺序、梯度和shuffle实现未改。

六个核心文件与准备前`34c7a8570569791d791c3fd94b170dde47ab8d79`按LF规范化字节一致：`dreamerv3/{agent,rssm,rep_probe}.py`、`embodied/jax/agent.py`、`embodied/envs/dmc.py`、`embodied/run/protocol_v1.py`。文件未改不替代实际CUDA执行验收。

控制器`scripts/run_dcs_dynamic.py`为每张卡建立一个串行lane。第i任务分配GPU(i mod 8)，顺序baseline→Dt；完成并通过审计后才进入下一条。失败、审计失败或资源冲突保留证据并阻塞该lane，不跳过依赖、不重用目录、不自动重试。没有完整环境/replay恢复承诺。第一批规划如下，**均未启动/未实际入队**：

| GPU | 首条 | 后继 |
| --- | --- | --- |
| 0 | hopper_hop baseline | hopper_hop Dt |
| 1 | quadruped_run baseline | quadruped_run Dt |
| 2 | quadruped_walk baseline | quadruped_walk Dt |
| 3 | reacher_hard baseline | reacher_hard Dt |
| 4 | acrobot_swingup baseline | acrobot_swingup Dt |
| 5 | cup_catch baseline | cup_catch Dt |
| 6 | cartpole_balance baseline | cartpole_balance Dt |
| 7 | cartpole_balance_sparse baseline | cartpole_balance_sparse Dt |

其余12任务继续使用`protocol_scope.VISION20_TASKS`顺序。W&B设置为原`ychan9liu-tsinghua-university/PD_1`、独立campaign group、新run ID、online、resume=never；本轮尚无实际W&B训练run。正式c匹配保留S/N和matching_result供审计，不给未请求的constant组启动。终点控制器审计完整网格、episode数量、快照、更新去重和计数；标准clean结果表入口明确拒绝将本DCS协议合并到clean指标，DCS分层结果表需在本轮参数确定后使用独立profile。

## 测试版本与证据

代码初版`3f18274a00776c884aa70fb49df60978ce961f93`；去除末尾空行`0dfbb5cfc0aed6823d18f331eebfd7f553bcea69`；补充拒绝clean override为`35dbfa0491e247018486ff0c7ad256e4078b1031`；进一步将完整agent验收与声明hash强绑定后的**实际最终代码测试SHA为`5245096ee52de5ba1ff6bcafb96d19298b74dcad`**。

- 本机Python3.12隔离环境：81 tests，28.730秒，全部通过，`regression-acceptance-code.log`。
- sv2既有dreamer环境：同81 tests，94.308秒，全部通过，`regression-5245096.log`。Python3.11.16、NumPy1.26.4、JAX/jaxlib0.4.33、dm-control1.0.48、MuJoCo3.15.0、Pillow12.3.0、wandb0.30.0；`pip check`通过，未修改服务器Python环境，详情`environment.json`。
- 两处均`JAX_PLATFORMS=cpu`、`MUJOCO_GL=disable`、`XLA_FLAGS=--xla_force_host_platform_device_count=2`，跨shard测试实际使用两个CPU逻辑设备。新增10项涵盖40配置、原算法文件身份、精确seed/素材分流、无RNG消耗、序列化、两类seed碰撞、素材hash/重叠拒绝、manifest不覆盖、队列8运行+8就绪、first16不扩张、错误资源/命令拒绝、clean指标隔离和同配置CUDA验收门槛。
- 保留旧算法、梯度、free-nats、shuffle、计步/去重/恢复、准备与原始return统计测试。沿用原测试容差：数值/梯度示例rtol=1e-6，梯度atol=0；离散映射、随机状态、配置和文件身份要求精确一致，没有放宽容差。
- 八卡零动作CUDA/EGL检查均通过，实测同5245096 SHA；证据`device-binding-5245096/gpu0-binding.json`至`gpu7-binding.json`记录CUDA可见UUID、枚举、实际进程、图像hash、前置占用和物理卡。
- 中间版本0dfbb5c、35dbfa0的80项CPU也通过；35dbfa0八卡绑定也通过。它们只作过程证据，当前代码以5245096测试为准。完整汇总及结束资源快照为`PREPARATION.json`。

测试命令：

```text
python -m unittest tests.test_dcs_protocol tests.test_dcs_background tests.test_rep_probe tests.test_rep_overlay tests.test_protocol_v1 tests.test_dmc16_preparation tests.test_vision20_seed1 tests.test_return_metrics -v
python -m scripts.prepare_dcs_background --dataset /data/Policy_Discrepancy/runs/engineering-dcs-dynamic-preparation-20261106/assets/DAVIS/JPEGImages/480p --output <新的素材审计目录>
python -m scripts.run_dcs_dynamic probe --gpu <0至7> --output <新的gpuN-binding.json>
```

**未通过/未执行的门槛**：未执行本轮dynamic真实size50m完整agent CUDA训练、CUDA报告及评价隔离验收，也未验证正式W&B配置远端可读；不能称本轮真实任务验收通过。准备代码的`integration`入口规定4098动作、间隔2049、每点1完整episode、匹配[2200,3600)、报告2049；四代表任务hopper_hop/quadruped_run/finger_spin/walker_walk各baseline/Dt共8条。所有缩小预算与测试seed/背景仅为工程fixture。正式`run`要求同SHA、同声明hash的8条完整agent acceptance；当前没有该文件，不能启动。

## Git与下一步

开始HEAD为34c7a85，已有AGENTS/CHATS/RESEARCH/STATUS草稿及论文文件均保留。工作期间main新增并行分析提交`ba59fc981f2621da176744346e32020b66c925b3`和`9a57256717359c8b30c5fc89cfe7918575ab9e23`。先核查后保留，以merge db8f5eb及`1f5d218fdc89cf7cbd2f7d32324cc25fb0bd7577`合入本轮代码，没有reset。后一merge相对实测5245096只新增`analysis/compare_seed01_dtl.py`及其测试；补跑3项通过，不影响本轮训练核心。之后本报告/STATUS/EXPERIMENTS的变更仅文档；最终完整SHA和同步范围另记交付收据及回复。

GitHub推送因连接重置失败，不宣称远端分支已更新。确定代码已通过三个bundle及SHA256/前置提交校验同步sv2；保留`git-sync-{0dfbb5c,35dbfa0,5245096}.json`。canonical repo的工作区仍干净、HEAD仍`d8997f64a301bfdcfe08758faeae8957a7395766`；实际准备测试在独立checkout `/data/Policy_Discrepancy/checkouts/dcs-dynamic-5245096ee52d`。未在服务器修改项目源码，未同步sv1/sv3。

首选下一步继续当前合并窗口（03），不新建chat：先按已确认seed0和背景配置执行同声明/同提交完整agent CUDA验收；收齐队列及容量答复后复核资源，按确认的rollout启动首八条和八条后继。启动后核查0点评价、实际同卡和W&B在线记录。依赖未决项的工作不被本报告视为完成；后续完整产物再交05分析，工程通过不代表Dt有效。

完成声明后使用已提交入口；以下占位项尚未绑定，不是已执行的启动命令：

```text
python -m scripts.run_dcs_dynamic prepare --declaration <已确认声明JSON> --campaign <新的dcs-dynamic-vision20-*名称> --rollout <all40或first16> --output <新的plan目录>
python -m scripts.run_dcs_dynamic integration --plan <plan.json> --evidence <同SHA八卡binding所在目录>
python -m scripts.run_dcs_dynamic run --plan <plan.json> --evidence <同一evidence目录>
```

只有本轮用户明确接受首批容量缺口时才添加`--accept-capacity-deficit`，不沿用旧批次例外。科学配置确定后若需改代码，使用新提交重新验收，不能直接改服务器源码或手工改acceptance。
