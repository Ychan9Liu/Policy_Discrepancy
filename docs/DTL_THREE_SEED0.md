# DtL 三任务 clean / seed0 运行

2026-11-05，用户明确启动 quadruped_run、quadruped_walk、finger_spin 各一条 DtL，
指定 sv2 物理 GPU5–7，并确认三条均为 clean。不是 DCS 背景训练，也不新增 baseline。
继承当前已经实际在线运行的 DL-pilot-r3 配置；本轮扩展任务，不把旧预验证门补写为通过。

## 方法与配置

当前实现来源是分支 codex/dl-prevalidation 的完整版本
`0b5d5a6c88d710dfaafd521dda177310a518d05b`，过去 quadruped_walk 1M 在线实测为
`13ec72a214b0c03b886b2c7fac52c8f089966dec`。旧分支 docs/DL_PILOT_R3.md 给出原在线口径。
本次从 main `479c25b71d2383208e9f7c13f1de7942b6d6755c` 集成现有 DtLatch 模块、
独立 dtl_agent/训练包装器、TwoHot.target 和只读 logging 支持，保留 main 已修复的 mode 诊断、report、恢复审计和 DCS 准备。
原 agent.py、RSSM、rep_probe.py、JAX agent.py、protocol_v1.py 和 dmc.py 保持字节不变。
DtL 入口使用独立模块，已有基准源码一致性检查继续执行，不删除或放宽原检查。
不整支覆盖 main，不修改两个既有工作区的草稿或历史。

`agent.dt_latch.mode=dtlatch`，`rep_probe.mode=off`，alpha=20、rho=0.5、H1 reward-support，
normal_only=False、kl_tolerance=1e-5、kappa=-1、bins=[-1]。没有 c 校准、强度匹配或 shuffle 臂。
`e=max(exp(-KL(y||Pq))-exp(-KL(y||Pp)),0)`，实际权重
`v=stop_gradient[1-rho*e*(1-1/(1+alpha*D))]`，其中不可用邻接位置为 v=1。
next 存在、current 非 is_last、next 非 is_first；next terminal reward 可以使用。
只作用于 free-nats 之后的原 rep KL，原 reduction、dyn、奖励/value 路径与梯度开关保持。
该额外信号没有 probe RNG；训练/回放/评价仍沿用原命名空间；原 logging 模式采用单独只读 JIT。

| 字段 | 本轮声明 |
| --- | --- |
| 分配 | sv2 GPU5 quadruped_run；GPU6 quadruped_walk；GPU7 finger_spin |
| 组/seed/初态 | DtL，seed0，三条从头训练；quad_walk 是新尝试，不续接旧 r3 |
| 观测/模型 | clean，64×64 RGB，proprio=False，size50m，CUDA/bfloat16 |
| 训练 | batch16×64/context1，16环境，train_ratio256，repeat1 |
| 损失/梯度 | free_nats1，dyn/rep=1/0.1；ac_grads=False；reward_grad/repval_grad/repval_loss=True |
| 预算 | 1000000实际动作，含预填充，排除 reset 和评价；249493更新 |
| 评价 | 0到1M每50k，共21点，每点10完整episode，固定快照、sampled actor、独立评价 RNG |
| 指标 | 原始完整episode return，实际动作步梯形积分/1M；80/85/90/95/100万五点均值 |
| 日志 | JSONL＋既有 W&B entity ychan9liu-tsinghua-university / PD_1；单独campaign/group/run ID |
| 监督 | 三条子进程启动后的监测周期上限57600秒；失败保留，不自动覆盖/恢复；三卡启动容量建议180GB |

17:42只读容量为182509658112字节，GPU0–4五条原项目作业继续写入。
用户随后明确“先跑吧，我会清理”，允许本轮低于180GB初始容量建议继续启动；
使用显式 `--accept-initial-capacity-deficit` 并记录实际余量，不扩展到其它批次。
监督器持续记录容量；低于5GB时仅终止本轮所属子进程、保留失败证据，不删除任何既有产物。

## 入口与验收

新增 `dtl_three` preset、封闭的 `protocol_dtl_three` scope 和 `scripts/run_dtl_three.py`。
原 M2 四组及扩展入口明确拒绝启用 DtLatch，避免以 baseline/Dt 标签运行新算法。
共用原动作计数、评价快照和恢复拒绝逻辑；DtL 每点评价均校验前后训练状态与随机流指纹。
逐成功更新另存 `dtl_updates.jsonl`：实际权重、free-nats 活跃比例、有效活跃比例、
保护位置、超额释放、前后rep均值、loss/grad_norm，按 update ID 与 checkpoint 严格对应。
正常 replay 重采样不去重；记录缺失、额外更新和重复ID拒绝重开。

准备、设备探针、工程、训练四个入口分别为 `prepare / probe / integration / run`。
训练必须绑定同SHA三任务真实 CUDA 工程 acceptance 和各卡 CUDA/EGL 实际同卡证据。
工程使用 size50m 原参数，2048动作、两点评价各1完整episode、5更新和1次真实report；
它验证接口、报告只读与评价隔离，不作为百万步结果或方法有效证据。
原 DtL CPU 梯度/非平凡权重/旁路/全agent/normal_only 测试与原 M2 算法、协议测试保留。
本轮证据根为 `analysis/outputs/dtl-three-20261105/`，大产物不提交。

实际提交、同步和启动/运行状态由后续带时记录补充；准备文件不等于任务已启动。
当前研究仍是单训练seed探索；三个任务不构成冻结20任务整体结果，episode不是独立训练重复。
不据已有r3或本轮工程通过声称保护机制或方法有效；新增科学复验仍由02与用户确认。

## 2026-11-05 18:07 启动验收

三条均已从头启动并进入真实训练，尚未完成百万步、末点评价或终点审计。
实际训练/三任务CUDA验收代码为 `f1f33bc9a74eb112f091a29f09307f81ba72513a`。
补充CPU核验工具与DCS测试替身修复版本为 `cbd24b2f6c29fb3a772c8472754cf4b844ba16cb`；
后者只改两个测试文件，没有改变训练源码、配置或启动命令。

实际训练根：`/data/Policy_Discrepancy/runs/dtl-three-clean-seed0-20261105-f1f33bc`。
工程/版本/校验根：`/data/Policy_Discrepancy/runs/engineering-dtl-three-20261105-f1f33bc`。
checkout：`/data/Policy_Discrepancy/checkouts/dtl-three-f1f33bc9a74e`；监督器PID3397966。
服务器canonical repo仍为原d8997f6且干净；只导入Git对象并增加独立checkout，GPU0–4旧作业未改动。
本机main已快进包含本次代码；AGENTS、CHATS、RESEARCH、STATUS中原未提交草稿和论文文件均保留。

| sv2物理卡 | 任务 | 训练PID | 18:07已核成功更新 | 对应最后更新开始动作步 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 5 | quadruped_run | 3398014 | 566 | 4292 | 运行中 |
| 6 | quadruped_walk | 3398033 | 599 | 4424 | 运行中 |
| 7 | finger_spin | 3398320 | 576 | 4332 | 运行中 |

这是带时读数，动作数为成功更新的开始动作步，不冒充稍后实时值。
三条0点评价均恰好10个完整1000动作episode、快照动作/更新均为0，评价前后训练指纹相同；
收据连续、已读loss/grad有限、权重位于[.5,1]。早期保护很弱：上述前缀最小权重分别
.9999988675、.9999825358、.9999997020；按原1e-6诊断阈值有保护的位置出现在1/29/0次更新。
这些只描述早期日志，不判断效果，也不据此调参或重跑。

18:06:21实际训练父子进程的compute UUID与pmon graphics再次逐卡核实，均只落在分配卡。
CUDA_VISIBLE_DEVICES按下列UUID，进程内CUDA0，EGL索引依次5/6/7：
GPU5 `GPU-ac733835-08ee-7a17-a2fa-1e49d62d385b`；
GPU6 `GPU-548be1c8-86f0-b7a2-de92-5c455b12221b`；
GPU7 `GPU-4cbf5b69-adb3-5210-2396-ca5d4e400f26`。
证据为训练根 `runtime-device-map-start.json` 和 `startup-health.json`。
18:07容量173978406912字节；用户的本轮容量例外已写入 `resource-before.json`，
`resources.jsonl`持续记录余量。用户仍需继续清理；不自动删除产物。

### 已执行测试与保留限制

- 本机：隔离后的62项回归全过，167.087秒；容量例外补充后6项runner测试全过。
  基准agent/RSSM/rep_probe/JAX agent/protocol_v1/DMC源码检查保留且通过。
  第一版直接合入触发旧源码不变检查失败，已改为独立模块；原失败日志保留。
- sv2 / f1f33bc：`CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu MUJOCO_GL=disable
  XLA_FLAGS=--xla_force_host_platform_device_count=2 DL_FULL_AGENT_TEST=1
  python -m unittest -v tests.test_dt_latch tests.test_dt_latch_logging
  tests.test_dt_latch_normal_only tests.test_dtl_three tests.test_rep_probe
  tests.test_rep_overlay tests.test_protocol_v1 tests.test_dcs_background
  tests.test_dmc16_preparation tests.test_vision20_seed1`：82项，325.038秒，
  80通过、1失败、1错误。**不标为完整CPU套件全绿。**
- DCS错误来自测试替身将 `numpy.str_` 直接交给Linux `os.listdir`，返回bytes文件名；
  只将替身输入显式转为Unicode字符串，与gfile接口一致。cbd24b2的
  `python -m unittest -v tests.test_dcs_background tests.test_dtl_three`：15项全过，2.974秒。
  clean环境与训练实现未变，原错误保留。
- 保留的严格失败：`test_full_agent_bypasses_logging_rng_params_and_optimizer` 中，
  active DtL初始全1权重路径对baseline的梯度/optimizer逐位等价断言在Linux不成立；
  最大绝对差 `2.980232238769531e-08`。off/alpha0/rho0/logging/strength0的严格旁路比较已通过。
  未放宽容差，未删除或标skip该断言，未修改算法来凑过测试。
- 为定位是否为此次集成回归，在独立原版 `0b5d5a6...` 与实际版f1f33bc上各生成baseline/DtL
  固定输入、梯度、采样和两次完整更新快照。cbd24b2工具
  `python -m tests.check_dtl_import --reference /data/Policy_Discrepancy/checkouts/dtl-reference-0b5d5a6
  --candidate /data/Policy_Discrepancy/checkouts/dtl-three-f1f33bc9a74e
  --output <engineering-root>/import-comparison` 退出0：每模式617个数组全部逐位相同。
  原版与新版对baseline均有同样19个差异数组（5梯度、两次optimizer各7），初始化/采样/损失无差异。
  因此该CPU严格等价失败属于原实现已有的固定输入数值边界，不是本次移植新增差异；
  精确编译融合原因未进一步定位，不外推CUDA跨版本或真实轨迹逐位一致。
  启动依照用户明确要求的现有DtL，不将该失败改写为通过，也不声称旧科学准入门已通过。
- sv2 / f1f33bc真实CUDA：三任务各2048动作、5次成功更新、2点评价各1完整episode、
  1次真实report；三worker均exit0，controller exit0，17:54:31审计全部通过。
  报告/评价训练状态隔离成立。前5次权重实际均1，不能拿此短跑声称真实非平凡gate已验收；
  非平凡梯度由保留CPU fixture覆盖，真实早期作用另由上述训练诊断记录。
  `integration/acceptance.json`及各任务完整配置、收据、审计文件保存。

### 启动方式、W&B与证据

先从固定checkout执行 `python -m scripts.run_dtl_three prepare --output <engineering-root>/preparation
--campaign dtl-three-clean-seed0-20261105-f1f33bc`，逐卡 `probe --gpu 5|6|7
--output <engineering-root>/gpuN-binding.json`，再 `integration --plan <engineering-root>/preparation/plan.json
--evidence <engineering-root>`。实际启动：

```text
/data/Policy_Discrepancy/envs/dreamer/bin/python -u -m scripts.run_dtl_three run
  --plan /data/Policy_Discrepancy/runs/engineering-dtl-three-20261105-f1f33bc/preparation/plan.json
  --evidence /data/Policy_Discrepancy/runs/engineering-dtl-three-20261105-f1f33bc
  --accept-initial-capacity-deficit
```

以上为已执行命令展示，当前运行目录不可重复启动。监督器以nohup独立运行；完整子命令、
环境映射和实际flat配置在各任务 `execution-initial.json` 与根 `actual-RUN_INDEX.json`。
W&B读取已确认三条state=running，task/seed0/DtL/alpha20/rho.5/1M与本地一致：
[quadruped_run](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/dtl-three-clean-seed0-20261105-f1f33bc-quadruped_run-dtl-01)、
[quadruped_walk](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/dtl-three-clean-seed0-20261105-f1f33bc-quadruped_walk-dtl-01)、
[finger_spin](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/dtl-three-clean-seed0-20261105-f1f33bc-finger_spin-dtl-01)。
18:11 API复核仍为running且配置正确，但远端summary/history尚空；本地0点记录完整。
安装环境的elements适配器调用 `wandb.log(metrics, step=step)`，未显式commit，
SDK将其发送为partial history；首个0点需后续日志步或关闭时提交，尚不能声称远端曲线已验收。
不为刷新面板改变训练或评价频率；下次50k点核实历史落盘与W&B同步。
原配置的可选JAX profiler提示缺少TensorFlow trace，之后成功更新持续增加；未改依赖或关闭原配置。

Git同步范围为本机与实际执行sv2；GitHub task分支首次push成功，随后独立ls-remote因连接reset失败，
不能把该失败写成重新核实远端成功。服务器bundle均verify并与本机SHA256一致：
生产f1f33bc bundle `3309ac9e4fecd1f092e91f9e4feb1116c9c773ea69376f03af5a6559c5ce90b4`；
补测cbd24b2 bundle `557da94d57a071acb019852192878d27bb72c63cdea45771eb5c34ef16d2ea98`。
后续报告提交推送同一GitHub任务分支再次因connection reset失败；最终交付仍经bundle同步至sv2，
不移动实际训练checkout。最终完整HEAD、bundle校验和及同步回执另存本机证据根 `DELIVERY.json`。
原实现0b5d5a6只部署CPU对照checkout，不启动GPU或重开其旧实验。
小证据镜像 `analysis/outputs/dtl-three-20261105/evidence/`；完整小包
`small-evidence-f1-cbd.tar.gz` 的SHA256为
`1135b7483b3d19bfd40df4d1a9693d3b697fb117d3b7b871453c635e06b6faa8`，
含原失败日志、修复测试、GPU验收和跨版本快照，不含size50m checkpoint/replay。

首选下一步继续当前合并03窗口监测这三条，检查磁盘与实际RUN_INDEX；
训练终点须100万动作/249493更新、21×10完整episode、逐点评价隔离、连续收据和监督审计通过，
并确认W&B同步与资源释放。完成后将完整产物交05分析，不自动建chat或发消息。
不得将这三条与原20任务直接混合汇总；新科学对照/重复仍需02和用户决定。
