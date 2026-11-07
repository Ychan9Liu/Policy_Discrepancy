# Vision20 dynamic baseline / Dt：准备记录

**2026-11-07执行更新**：用户清理后要求按原计划启动；先启动七条，等待GPU2外部作业释放后已补接。20:06合并索引为8运行/8就绪/24backlog/0阻塞；训练仍为5245096，GPU2接续控制器为e04b1e8。实时启动核查、容量执行判断、完整命令及后续状态见 [11月7日启动报告](DCS_DYNAMIC_LAUNCH_20261107.md)。以下为11月6日准备阶段的历史状态与原始工程证据，保留当时未启动的记录。

状态时间：2026-11-06 11:40（北京时间）。本轮目标为20任务各baseline、原Dt一条，共40条；资源为用户指定的sv2物理GPU0–7。当前**8条缩小预算完整agent工程验收通过，0条正式启动、0条正式排队**，没有创建正式训练控制器。补充的GPU额外渲染逐像素检查未通过，边界见下文；不称20任务全部CUDA逐位验收通过。按用户原40条总目标准备全40自动补入，首批容量安排仍待答复。

## 已确认与待补齐

- 用户明确：全20任务、dynamic背景、baseline/Dt、sv2全部八卡、先挂八条并排八条；本次资源包括GPU6，旧批次“不用GPU6”不适用于新授权。
- Dt指原PPD-KL，`agent.rep_probe.mode=dt`，不是DtL；baseline为logging-only，实际训练权重1。DtLatch关闭。原clean批次和已完成结果不被覆盖或重新分类。
- 共同继承项：size50m、64×64 RGB、仅视觉、每条100万实际训练动作、repeat=1；alpha20、free-nats1、dyn/rep=1/0.1、ac_grads=false、reward_grad/repval_grad/repval_loss=true；16训练环境、batch16×64、context1、train_ratio256、CUDA/bfloat16、prealloc=false；评价0至100万每5万一次、每点10完整episode、固定快照及采样动作；匹配窗口[100000,300000)，成功更新开始动作步归属、有效loss位置S/N。此轮不运行constant/shuffle，不把背景问题改写成新的算法定义。
- 用户本轮答复已确认：训练seed0；DAVIS train 60视频训练、独立val 30视频评价，video_alpha=1，14任务上游地面规则，另6任务地面alpha=0.3。背景独立MT19937沿用现有派生实现，根标识-1、cache_size=2、num_videos=-1；物理环境/动作/agent RNG沿用原seed0映射。
- 队列按用户原40条总目标解释为全40自动补入，最多8条运行、8条就绪，其余24条初始为backlog。关于“仅先执行16条”的偏好问题尚无答复；这是可覆盖的调度偏好，不另当作40条已授权范围的启动许可。若用户改选first16，切换到对应准备计划，另24条不得自动启动。
- 仍待答复的是存储安排。正式控制器尚未启动；不将未回复或等待时间当作接受容量缺口。

启动前的配置声明包含seed、全部背景字段、每任务地面alpha和视频划分，并以SHA256绑定到plan与实际config。科学配置已由本轮两项答复补齐，同配置工程验收已完成；现存储不足以承诺首八条跑完，仍待容量处理，不重新索取已授予的服务器/40条任务启动权限。

## 资源与准备产物

sv2入口仍为既有SSH配置，hostname `lyg0360`。开始时SSH多次在认证前关闭，10:41恢复一次；后续以持续SSH会话完成准备。不能据成功连接推断今后入口稳定。

10:41八卡均空闲，/data剩147,643,842,560字节。准备素材和测试后，服务器10:57:32 UTC+8（本机10:58）余量145,960,243,200字节，GPU0–7均0MiB、无C/G进程。零动作设备检查进程已结束，没有本轮GPU任务遗留。

最终准备快照为服务器11:03:29，145,947,152,384字节可用，八卡仍均0MiB且无C/G进程；该时间和原始输出保存在`PREPARATION.json`。

完整agent验收后11:16:11八卡均释放，无C/G进程，余量128,629,153,792字节，证据为`device-binding-5245096/integration/engineering-summary.json`。11:18:24重查GPU0空闲后补做20任务固定动作EGL环境检查，模型更新/正式训练动作均为0；其独立收据为`render-execution.json`，不占用其他卡。

11:35:20最终资源快照：八卡均0MiB、无C/G进程，/data余量128,594,694,144字节。所有本轮工程进程已退出，证据`resources-final.json`。原始素材、成功及失败证据均保留，未删除已有产物。

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

**完整agent工程验收通过**：11:09启动、11:14:34八条全部accepted。每条4098训练动作、517成功更新，0/2049/4098三点各1个完整1000动作episode（评价动作3000另计），两次CUDA report的训练/回放/输入fingerprint不变、三次评价隔离检查通过。匹配窗口[2200,3600)，四baseline匹配S/N均有效，工程c不用于正式实验。

GPU0/1为hopper_hop baseline/Dt，GPU2/3 quadruped_run baseline/Dt，GPU4/5 finger_spin baseline/Dt，GPU6/7 walker_walk baseline/Dt。科学设置采用本轮已确认seed0/background，只有预算、评价数量和窗口为工程fixture；日志仅JSONL。实际运行代码5245096，控制器收据为准备根`integration-controller.json`，运行/审计位于`device-binding-5245096/integration/`；`acceptance.json`的code SHA和声明hash均匹配。`live-device-audit.json`将136个实际C/G进程沿父进程追溯到八条作业，全部位于各自指定物理卡，无未知归属或跨卡项。四代表任务完成不等于40条百万步任务验收，更不是方法效果结论；尚未验证正式W&B运行的远端记录。

最后一个训练聚合段中，四baseline的实际权重均精确为1、active_effective_frac=0；四Dt的rep_active_frac均为1，实际活跃位置权重均小于1，不能误记为无作用。下表只是工程诊断，作用幅度很小，不解释收益或因果：

| Dt工程任务 | 活跃位置实际平均权重 | active_effective_frac |
| --- | --- | --- |
| hopper_hop | 0.9999258259 | 0.90794945 |
| quadruped_run | 0.9999214894 | 0.99726105 |
| finger_spin | 0.9999287064 | 0.87075615 |
| walker_walk | 0.9999250706 | 0.99769020 |

实际声明`declaration.json`的SHA256为`c51f0760c481057571719c932c6d84e400c9122760b94e0e4f959ead816691a7`，本机与服务器字节一致。工程调用使用`integration-plan/plan.json`，其中临时rollout=first16不会被integration用来启动正式作业，不代表用户选择了正式16条范围。正式方案为`formal-plan-all40/plan.json`及20任务`rng-mapping.json`，尚未据此启动。正式`run`要求同SHA、同声明hash的8条完整agent acceptance；此工程门槛现已满足，容量条件仍未解除。

## 额外渲染的严格失败与适用边界

`scripts.probe_dcs_background`在同5245096、sv2 GPU0、NVIDIA A100-SXM4-80GB的EGL后端补测20任务时，首任务hopper的dynamic-diagnostics检查失败：恢复原纹理后的重新渲染，18/12288个颜色坐标与之前图像不同，最大绝对差1。输出`render-vision20-5245096/acceptance.json`保持`accepted=false`、tasks为空，不能写成全20通过；原始`render-vision20-5245096.log`保留。没有放宽精确相等要求，也没有删除失败或给现有严格probe增加跳过选项。

新增只诊断、无agent的`scripts/diagnose_dcs_render.py`，测试代码`eb724d8889f8b6d473821d494bd25eba78316f36`，实际环境导入仍固定5245096。对同一任务固定动作/重置序列分别执行不额外渲染、额外渲染、不换内容重新上传纹理、临时替换再恢复。结果：dynamic仅多渲染时同样出现低位像素差，18帧未来图像有991/221184颜色坐标不同、最大差2；其物理状态、原始奖励、任务RNG、背景状态/RNG均精确相同。重新上传和替换恢复也有类似差异，texture字节检查均相同。因此不能把差异归因于frame_override恢复失败，也不能据此归因hopper学习失败。

进一步的隔离对照代码`d205bdac6913d43c3e80dcd624f37fe718841755`只在诊断实例关闭GL_DITHER：该实例的GL状态确认为False，但像素差异未消除（obs→重复render累计1122坐标、相邻render累计595坐标，最大仍2）；该次clean-repeat也出现1个坐标差1。因此“dithering是唯一原因”不获支持。GL_DITHER作为颜色量化相关状态的背景依据见[Khronos OpenGL规范](https://registry.khronos.org/OpenGL/specs/gl/glspec30.pdf)，本结论依据实际对照，不把规范当作故障原因证明。

所有对照NPZ、逐位置统计和精确状态检查保存在`render-controls-eb724d8/`、`render-dither-control-d205bda/`。生产渲染代码/GL设置未修改，未改变视频、alpha、地面规则或训练协议。GPU低位像素非确定性的具体底层原因仍未定位；此前CPU逐位证据仅限CPU，不能推广到该GPU。`render_background_frame`的纹理/物理/RNG恢复有证据，但**不承诺此GPU上的图像逐位恢复或额外render之后未来图像逐位相同**。

这条辅助配对渲染接口未在本轮训练/评价/report路径使用。原report是对已保存batch的模型报告，八条完整agent中的参数/回放/输入和评价训练状态隔离检查仍已通过；不将上述辅助检查改标通过，也不宣称额外渲染与没有额外渲染的GPU训练轨迹严格一致。未来需要逐像素配对的机制研究应由02/用户先处理此边界，不能直接用该接口宣称控制了全部输入。

## Git与下一步

开始HEAD为34c7a85，已有AGENTS/CHATS/RESEARCH/STATUS草稿及论文文件均保留。工作期间main新增并行分析提交ba59fc9、9a57256、2e54499及7d5ee9b，均先核查后保留，没有reset。代码及后续诊断脚本通过merge合入main（至`9f48911303c603f1adcc74a09404e87b6ec36a51`）。相对完整agent实测5245096，增加了只诊断脚本、并行分析及其4项已补跑通过的测试、相关文档；训练/环境生产代码没有继续变化。最终完整SHA和同步范围另记交付收据及回复。

GitHub首次推送因连接重置失败。确定代码已通过bundle及SHA256/前置提交校验同步sv2；保留`git-sync-{0dfbb5c,35dbfa0,5245096,ee272ab,eb724d8,d205bda}.json`。canonical repo的工作区仍干净、HEAD仍`d8997f64a301bfdcfe08758faeae8957a7395766`；实际准备/完整agent测试在独立checkout `/data/Policy_Discrepancy/checkouts/dcs-dynamic-5245096ee52d`；诊断测试使用自己的已提交checkout，显式导入5245096的生产环境。未在服务器修改项目源码，未同步sv1/sv3。最终推送/交付bundle结果以交付收据为准。

首选下一步继续当前合并窗口（03），不新建chat：完整agent工程门槛已通过；落实容量后复核资源，按全40自动补入方案启动首八条和八条后继（用户另改first16时从其最新指令），并保留上述GPU辅助诊断边界。启动后核查0点评价、实际同卡和W&B在线记录。当前不能以交接代替尚未完成的启动；后续完整产物再交05分析，工程通过不代表Dt有效。

完成声明后使用已提交入口；以下占位项尚未绑定，不是已执行的启动命令：

```text
python -m scripts.run_dcs_dynamic prepare --declaration <已确认声明JSON> --campaign <新的dcs-dynamic-vision20-*名称> --rollout <all40或first16> --output <新的plan目录>
python -m scripts.run_dcs_dynamic integration --plan <plan.json> --evidence <同SHA八卡binding所在目录>
python -m scripts.run_dcs_dynamic run --plan <plan.json> --evidence <同一evidence目录>
```

只有本轮用户明确接受首批容量缺口时才添加`--accept-capacity-deficit`，不沿用旧批次例外。科学配置确定后若需改代码，使用新提交重新验收，不能直接改服务器源码或手工改acceptance。
