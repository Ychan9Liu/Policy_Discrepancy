# Vision20 dynamic：sv2八条Dt删除旧目录与attempt02重跑

2026-11-12，当前合并03窗口执行用户“删除实验记录，重新跑”，随后明确“只删除旧Dt目录，保留W&B失败记录”。范围仅为11月8日磁盘保护中断的八条Dt；已完成baseline、sv3十二条完成运行及尚未启动的十二槽不在本次删除或重跑范围。没有恢复旧监督器，也没有覆盖旧W&B ID。

## 最新状态：11月13日维护导致I/O阻塞

**11月13日10:52–10:55重新核查后，当前状态为八条维护阻塞，尚未完成。用户已明确“正在维护，恢复后通知”。** 下文11月12日的八条正常启动与工程通过均为当时证据，不代表现在仍在正常推进。保留现有训练和监督器进程，等待用户通知存储恢复；不重启、不删新产物、不新开尝试、不干扰其他作业或维护操作。

- 八个原训练PID与启动命令匹配，均处于Linux D/disk-sleep；七个等待`balance_dirty_pages`，quadruped_walk等待`wait_on_page_bit_common`。监督器PID1993585仍存活，处于正常轮询sleep，没有supervisor-error/final。
- 成功更新收据可读到：GPU0 hopper、GPU1 quadruped_run、GPU3 reacher、GPU5 cup、GPU6/7 cartpole均止于起始动作800000/update199492；GPU2 quadruped_walk止于150000/update36992。GPU4 acrobot的收据尾部读取超时，当前精确动作进度未确认，不推测为80万。其余七条均无final_state；八条均未被监督器验收完成。
- `/data`仍余799,411,085,312字节，并非此前5GB容量保护再次触发。系统Dirty约132,429,284KiB，I/O pressure full avg10约98.56%；`rbd0`仍有9个未完成写请求，间隔5秒的块设备统计完全不变，八训练进程I/O计数在连续检查间未前进。这些证据定位到存储I/O等待，不证明具体Ceph后端/网络组件故障；用户随后确认服务器正在维护。
- 本账户读取dmesg被拒绝；GPU查询和个别文件读取亦阻塞，不能据此报告卡已空闲或断言GPU驱动故障。中断的仅是本轮自有只读查询；一个限时读取helper（PID2250495）在内核I/O等待中已发送退出信号，可能需存储恢复才真正退出。未停止训练或其他用户进程，未调整内核参数、卸载设备或重启服务器。
- `actual-RUN_INDEX.json`的checked_utc仍是启动时刻。此控制器只在启动/结束/错误事件发布index，正常轮询写resources；无新的退出事件时不刷新index时间。不能用旧running标记替代实际进程/I/O核查。11月12日的`observed-40-slots.json`亦为历史快照；当前口径为历史20 accepted、8维护阻塞、12未启动。

最新只读证据在本机`analysis/outputs/dcs-dt-retry-20261112/io-block-20261113.json`及`blocked-runs-20261113.json`。为避免继续向阻塞的/data写入，本次维护说明只更新本机/GitHub文档，不强行同步服务器；sv2实际代码和训练checkout仍分别是8d74cf9/5245096，未改变。

首选下一步仍在当前03：用户通知维护恢复后，先只读核对rbd I/O、空间、GPU UUID/进程归属及原八进程/日志是否重新推进；不要重复运行本报告的启动命令。原进程恢复执行时继续同一attempt02并验收收据/快照完整性；如果维护使原进程丢失，遵守runner环境/replay恢复限制，不能把checkpoint接续称为无缝恢复。需要新尝试时按有效原授权和资源重新准备，保留attempt02证据，不覆盖或迁卡。终点仍须每条百万动作、249493更新、21×10完整评价和原始审计通过，完整结果才交05；无方法效果结论，无自动chat或消息。

## 范围、版本与协议

| sv2物理GPU | 任务 | 本次组/尝试 |
| --- | --- | --- |
| 0 | dmc_hopper_hop | Dt / attempt02 |
| 1 | dmc_quadruped_run | Dt / attempt02 |
| 2 | dmc_quadruped_walk | Dt / attempt02 |
| 3 | dmc_reacher_hard | Dt / attempt02 |
| 4 | dmc_acrobot_swingup | Dt / attempt02 |
| 5 | dmc_cup_catch | Dt / attempt02 |
| 6 | dmc_cartpole_balance | Dt / attempt02 |
| 7 | dmc_cartpole_balance_sparse | Dt / attempt02 |

- 训练固定原实测SHA `5245096ee52de5ba1ff6bcafb96d19298b74dcad`，cwd为`/data/Policy_Discrepancy/checkouts/dcs-dynamic-5245096ee52d`。本轮控制器/最终CPU测试SHA为`8d74cf9576ed736161a09bfe6cd9a3ea33927bfc`，控制器cwd为`/data/Policy_Discrepancy/checkouts/dcs-dt-retry-8d74cf9576ed`。
- 本轮开始HEAD为`616bc27756cc9adb21a3e00039fe89d0c68e784d`。代码首提交`46934dc018e04fb24213d689cc80f5d11e126675`及后续进程可见性修订均保留；累计只增加`scripts/run_dcs_dt_retry.py`和`tests/test_dcs_dt_retry.py`。主目录已有AGENTS、CHATS、RESEARCH、STATUS修改及proposal/论文文件全部保留，不将它们夹带进本次提交。
- M2、dynamic、seed0、size50m、64×64仅视觉；原Dt/PPD-KL、alpha20、DtLatch关闭；free-nats1、dyn/rep=1/0.1、ac_grads=false、reward_grad/repval_grad/repval_loss=true。16训练环境、batch16×64、context1、train_ratio256、CUDA/bfloat16、prealloc=false。
- 每条从零训练100万实际动作，含预填充、排除reset/评价，repeat1。0至100万每5万评价一次，共21点，每点10个完整episode；固定快照、采样动作、独立评价随机流。原始return语义不变。
- DAVIS训练train60、评价独立val30，video_alpha1、num_videos=-1、background seed=-1、cache2；地面规则按原声明。声明SHA256仍为`c51f0760c481057571719c932c6d84e400c9122760b94e0e4f959ead816691a7`。每行完整resolved config与原Dt相比**只有logdir变化**；环境只改TMPDIR/WANDB_DIR及新的W&B ID，随后写入重新验证的EGL绑定。
- 本轮不运行baseline、constant或shuffle，不重新拟合c；既有baseline按原[100000,300000)成功更新起始动作及有效活跃loss位置S/N累计规则保留。工程短跑的预算/窗口不是新的科研协议。

## 精确删除与保留证据

旧根R：`/data/Policy_Discrepancy/runs/dcs-dynamic-vision20-seed0-20261106-5245096`。只删除上表八任务的`R/<task>/dt-attempt01`整个目录，最终均不存在。检查精确任务白名单、目标解析路径及非symlink、没有final_state、原manifest/命令/config/训练SHA匹配、可读进程引用、归档前后目录清单未变；未删除父目录或baseline。

16:46:19–16:46:39（北京时间）执行，八目录原分配块合计179,228,352,512字节。删除前可用805,350,096,896，删除后984,568,475,648字节，净增179,218,378,752字节。共享盘余量还受其他写入影响，不能将跨时刻全部变化归因于此操作。

保留206份原始日志/配置/更新收据/评价及快照元数据，八个独立gzip归档合计11,049,423字节；归档逐文件SHA256解包回读一致。检查点模型、replay与本地W&B二进制没有做payload归档。新证据根P：`/data/Policy_Discrepancy/runs/engineering-dcs-dt-retry-20261112`，包含`failure-evidence/`、`deletion.json`、旧监督器原始错误/资源/index/manager.log。旧监督器索引保留原样，陈旧running仍是历史错误，不能作为当前状态。

八条baseline的完整文件元数据清单及final_state SHA256前后相同；这里没有声称重新哈希所有保留模型内容。16:47:46只读W&B回查八个`...-dt-01`均仍crashed，证据`wandb-failed-retained.json`；没有W&B删除/修改操作。失败尝试的训练payload已不可用于恢复，故障分析仍可用原日志及云端记录。

进程检查记录普通账户的可见性边界：可读命令/cwd/fd中的目标引用会拒绝；同账户普通worker fd不可读也拒绝。其他UID及已识别的nondumpable PAM/SSH会话helper的隐藏cwd/fd不可检查，逐PID留档，不声称全机root级可见性；没有提权、停止或干扰其他作业。服务器预检曾因这些系统会话进程拒绝，随后在本机补实现和测试、重新提交同步；所有失败日志保留，拒绝发生在删除和GPU训练之前。

## 监督器修复与恢复边界

新入口仅允许这八个源Dt槽位，生成独立attempt02路径和`source_run_id/source_directory`来源映射。W&B仍在`ychan9liu-tsinghua-university/PD_1`及原campaign group，ID由`-dt-01`变为`-dt-02`，`WANDB_RESUME=never`。独占创建运行根；已有根/目录、错误manifest或旧目录未删除均拒绝，不静默重开。

正式并发门槛为每条60GB×8＋50GB共享余量=530GB；每次启动前按当前运行/待启动总数再次检查，不再只检查单条60GB。工程短跑为每条5GB×8＋50GB=90GB。它们是保守执行门槛，不是磁盘配额或科学参数，不保证其他用户大量写盘时仍能完成。

保留原5GB紧急停止线，每15秒检查。异常/SIGTERM收尾只停止自己创建的进程组；逐条写入`failed_storage`/`failed_supervisor`、真实exit_code、ended_utc和原因，最后重新发布index。单个停止失败或写盘失败不跳过其他自有子进程的停止；已accepted结果保持。SIGKILL、断电或完全不可写的文件系统仍可能无法持久化最终状态，须核对PID/日志，不把任意旧running当现状。没有扩展为通用恢复或修改旧已停止监督器。

原runner不能完整保存环境、driver carry、replay采样RNG和背景状态，故本次从零重跑，不称为80万步无缝恢复；原始77.26万–80.22万段不拼接到新attempt02。

## 工程验收与同步

本机最终新增测试13项中10项通过、3项Linux限定跳过；原监督器回归亦保留。sv2实际最终SHA的命令：

```bash
cd /data/Policy_Discrepancy/checkouts/dcs-dt-retry-8d74cf9576ed
JAX_PLATFORMS=cpu MUJOCO_GL=disable CUDA_VISIBLE_DEVICES= \
XLA_FLAGS=--xla_force_host_platform_device_count=2 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/data/Policy_Discrepancy/envs/dreamer/bin/python -m unittest \
  tests.test_dcs_dt_retry tests.test_dcs_sv3_six \
  tests.test_dcs_dynamic_gpu2 tests.test_dcs_protocol -v
```

40项全部通过，无跳过，190.006秒，退出码0。覆盖530GB含等号边界、重现旧189/176GB不足、配置/RNG相关环境映射不变、新ID/范围、防路径逃逸/完成运行误删、归档哈希、真实进程组停止及无关进程保留、异常之后退出码落盘、重复入口拒绝、写盘失败仍清理其他自有worker、旧调度与DCS协议回归。fixture只证明相应工程行为。

16:46:41–44重新核sv2 GPU0–7均0MiB且无作业；八次零动作CUDA计算＋实际EGL渲染probe均exit0，物理UUID与原分配相同，逻辑CUDA0及对应EGL index记录在P/bindings。16:47:39–42再次确认全空闲后，监督器PID1982900启动八条真实size50m、CUDA/bfloat16 Dt短跑：每条4098动作、517成功更新、0/2049/4098三点评价每点1完整episode、匹配窗口[2200,3600)、报告间隔2049；jsonl而非W&B，不能作为正式指标。完整命令/配置和各卡启动前即时资源证据在integration-launch及各execution-initial。

依赖未安装/升级：Python3.11.16，`pip check`通过，freeze及驱动清单在P，未动base-py311。GitHub最初远端查询连接重置，先通过Git bundle校验/fetch/独立worktree同步至sv2；随后普通push已成功，origin/main从65d5db8前进到8d74cf9。没有在服务器编辑项目源码。同步传输根Q：`/data/Policy_Discrepancy/runs/engineering-dcs-dt-retry-transport-20261112`，保存五版bundle、哈希收据、CPU测试与准备拒绝日志；最终实测为8d74cf9。sv1/sv3未同步、未核卡、未启动。

16:52:47真实CUDA短跑八条全部exit0/accepted，`integration/acceptance.json` passed=true。每条4098动作/517成功更新、3×1完整评价及原始收据/快照/seed映射通过，汇总24个完整episode；两次CUDA report和三次评价均由原runner在写审计前验证训练参数、计数及replay RNG指纹完全相等，不新增容差。八卡live-devices记录主计算进程及16训练环境渲染进程均属于对应worker且在指定卡。`engineering-diagnostics.json`保留最后日志窗口的D、实际权重、rep活跃性和作用前后值：八条rep_active_frac=1，actual_weight_mean约0.999727–0.999929、作用后rep均值小于作用前，无非法D/匹配值；这是短跑实际作用记录，不是收益或稳定性结论。

## 正式启动与后续验收

16:53:34–38再次核sv2物理GPU0–7均空闲、UUID匹配；/data可用965,104,742,400字节，高于530GB门槛。正式监督器PID1993585于16:53:38创建，八条worker于16:53:53–57依次启动，每条另有启动前即时占用记录。新的正式根N：`/data/Policy_Discrepancy/runs/dcs-dynamic-dt-retry-20261112-5245096`，路径为`N/<task>/dt-attempt02`；实际命令如下，不能再执行一遍以重复启动：

```bash
cd /data/Policy_Discrepancy/checkouts/dcs-dt-retry-8d74cf9576ed
JAX_PLATFORMS=cpu MUJOCO_GL=disable CUDA_VISIBLE_DEVICES= \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/data/Policy_Discrepancy/envs/dreamer/bin/python -u -m scripts.run_dcs_dt_retry run
```

监督器使用CPU；训练子进程经过原child_environment清除CPU/XLA等父环境覆盖，使用对应物理GPU UUID、JAX_PLATFORMS=cuda及重新验证的EGL index。W&B八个新ID在启动前只读确认不存在，正式训练在线创建；旧失败ID均保留。完整可重放命令、环境映射与原始配置在P/plan、formal-launch及N各execution-initial。

**16:59:50正式启动审计通过**：八条实际训练进程均存活、命令/cwd/manifest/背景声明/seed/设备与绑定配置一致；每条零动作快照完成恰好10个1000步episode，共80个，原始return均值核对采用已有1e-12绝对/相对容差，快照ID和实际步0/更新0匹配。逐条读取127–268次成功更新前缀，update ID连续唯一、invalid_count=0；W&B八条均running且完整config与运行manifest逐项相等、group仍是原campaign。每卡GPU计算/渲染进程全部为该worker及其子进程，已验证CUDA/EGL同卡及父CPU环境未泄入训练。`formal-startup-audit.json`、`formal-audit-index.json`和`formal-frozen-evidence/`保存对应原始字节及SHA256；审计只读训练及W&B，不将零点评价分数解释为方法效果。此时正式终点指标仍不完整，八条状态为运行中，sv2八卡仍由本项目占用，无已释放的正式运行卡。

首选继续当前03监测这八条，终点要求每条100万动作、249493成功更新、准确21×10完整评价、原始快照/收据/返回值审计、exit0且accepted，再把可审计结果交05分析。正常故障沿原授权修复验证，不能改科学配置、静默复用失败尝试或共享未知占用卡；需要重新训练时保留新失败证据并使用新尝试ID。本次是执行修复与重跑启动，不能将40项fixture或8条工程短跑通过写成方法有效，也不将正式启动写成训练完成。

原40个逻辑槽位的历史完成数仍为20；本轮替换的是八条失败Dt的尝试，不能把旧attempt01与新attempt02相加成48条正式比较。剩余12条（hopper_stand、pendulum_swingup、reacher_easy、walker_run、walker_stand、walker_walk各baseline/Dt）尚未启动，本次控制器不包含它们，不会自动接续；后续安排与当前八条的终点审计分开记录。sv3的既有12条完成状态沿11月11日只读核查，本轮没有重新连接核验。

P/`observed-40-slots.json`提供40个唯一逻辑槽位的派生索引，按`source_run_id`用新attempt02替换八个失败尝试，原attempt01作为已删除payload/保留W&B的来源元数据；20 accepted、8 running、12 not_started。sv3行明确保留2026-11-11观察时间，不伪装为本轮实时检查。不要恢复旧监督器或读取其陈旧running作为当前状态。

完整产物在N，工程、删除与启动证据在P，版本/CPU证据在Q；本机镜像为`analysis/outputs/dcs-dt-retry-20261112/`，大训练产物不入Git。报告与STATUS的交付文档提交可能晚于实际测试8d74cf9；最终完整SHA、差异范围和同步收据在本机`DELIVERY.json`及最终回复中记录。
