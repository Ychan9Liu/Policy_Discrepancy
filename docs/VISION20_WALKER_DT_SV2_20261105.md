# Vision20 seed1：四排队归sv2，补迁walker_run Dt（2026-11-05）

用户要求“四排队的挂到sv2”。核查四条为pendulum_swingup Dt、walker_walk Dt、walker_stand Dt（已经在sv2 GPU0/1/2各自baseline后排队）及sv1 GPU5的walker_run Dt。因此只迁最后一条，不复制已有三条、不增加训练次数、不将另一个blocked的hopper_stand Dt擅自纳入。原三条依赖屏障/配置/位置保持。

10:22:24 sv2 GPU4/5/7空闲、0—3本项目运行，/data余256350208000字节。在既有授权GPU0—5/7范围内选GPU4单条执行walker_run Dt，不用GPU6。独立初始60GB门槛满足，不沿用sv1容量例外、不改变budget/replay。sv1 /data仅19.418GB，GPU4已无外来进程，但hopper Dt旧阻塞与容量门槛仍未解除。

## 从queued源安全撤销启动资格

本轮与前七条blocked迁移不同：walker Dt仍queued，原监督器会在baseline结束后尝试启动，单写claim不足以防重复。10:23:38重新读取源行，确认queued、无PID、旧目录不存在；用mkdir(exist_ok=False)原子预占旧dt-attempt01目录，写唯一MOVED_TO_SV2.json及原正式根reallocation-claims/dmc_walker_run-dt.json。若原启动器先创建目录，迁移将失败且不能新启动；本次占位先成功，原启动器同样的mkdir在打开日志/Popen之前必定失败，不能运行第二份。

占位是运行产物/撤销标记，不是服务器项目源码修改、检查点或伪造训练。原监督器1609040、baseline PID2433371均未停止/重开；旧actual-RUN_INDEX不手改，其后对旧Dt的容量或FileExistsError阻断是预期撤销，不算替代运行失败。旧目录不允许自动清除标记，源slot需按claim从全40中排除。

源快照SHA256 `98694b861090e84cd30be84584d81c783d37035fb8bd1cc8713a3727f626027e`，迁移收据在sv1 `/data/Policy_Discrepancy/runs/migration-vision20-walker-dt-20261105-sv1/`。新run ID附`:moved-sv2-walker-20261105`，W&B ID同样独立后缀，resume=never；目标目录/data/Policy_Discrepancy/runs/formal-m2-vision20-walker-dt-20261105-sv2/dmc_walker_run/dt-moved01。新控制器严格核对source ID/hash/target/原子占位收据，不接受live/错seed/错task/非queued来源。

## 版本、配置与验收

实现提交`4f3431a236aac8903b111ea1b7daf268eaeda3b0`；补测试/实际控制器`2f7e6358fa45f125ac630aabb8b5d9f3ef3bfa2a`，新增scripts/run_vision20_walker_dt.py与tests/test_vision20_walker_dt.py。实际训练仍已验收`bef29480df12d07659c640b22218804dd1d69e63`，没有训练计算/梯度/RNG更改。新controller checkout /data/Policy_Discrepancy/checkouts/vision20-walker-dt-2f7e6358fa45；训练仍/data/Policy_Discrepancy/checkouts/vision20-seed1-bef29480df12，旧sv1/sv2活动checkout和canonical不移动。

完整实际config与原walker Dt逐字段比较，仅logdir改变；执行元数据变为sv2/物理GPU4/实际UUID-EGL及新W&B身份。保持clean、仅64RGB视觉、size50m、seed1、alpha20、free nats1、dyn/rep1/.1、ac_grads=false、reward_grad/repval_grad/repval_loss=true；batch16×64/replay_context1/16env/train_ratio256/repeat1、CUDA/bfloat16/preallocFalse、正式report关闭、JSONL+线上ychan9liu-tsinghua-university/PD_1。百万动作含prefill排除reset/评价，0..百万每5万21点各10完整episode、固定快照/采样动作/独立评价RNG。原始return/实际x积分固定百万分母/指定五尾点和缺项处理按EXPERIMENTS C/C.1。

Dt不读取c，不从baseline checkpoint派生，可独立从头启动；sv1自身seed1 logging baseline仍由原监督器完整执行/审计并按[100000,300000)有效位置S/N冻结c，非法或无活跃停止，不填c1/改alpha。迁移不构成新增科学协议或新seed/预算授权。

本机六项相关测试通过（0.671秒）：完整配置仅logdir差异/独立身份、queued/live/错seed/错task拒绝、源占位/hash/target必需；在真实原run_vision20_seed1.run队列逻辑中模拟已运行监督器的旧目录占位，真实mkdir触发FileExistsError、Popen未调用且标记不变，验证撤销不能被旧控制器忽略。

Git bundle SHA256 `df19b74ff02a920d147abd43c2151d119ecf1288f757f41eab193a38ce86ff6b`已verify/fetch，仅sv2增加干净独立控制器，canonical仍d8997f64a301bfdcfe08758faeae8957a7395766；sv1只写授权迁移运行标记/收据、不部署新代码，无sv3/GitHub推送、无依赖安装/删除，base-py311保留。

10:26:38验收pipeline3182328启动，先保留129项并补本轮3项CPU回归，再实际GPU4 CUDA/EGL同卡探针、真实连续动作size50m CUDA/bfloat16短程4098动作/517更新/3点各1完整episode/2报告，沿用严格收据/快照/报告及评价只读隔离审计；均通过后新鲜核查GPU/空间/PD_1并正式启动。本机fixture不是实际CUDA验收，工程通过也不当百万步或方法有效。

10:28，132项服务器回归通过（94.615秒）、GPU4真实CUDA/EGL同卡探针通过。source-identity.json核对训练bef2948与控制器2f7e635的dreamerv3/、embodied/及复用启动/审计源码Git diff为空，算法、优化器及随机流未改；CPU启动器flags仍由原child_environment隔离，不进入CUDA子进程。

工程根/data/Policy_Discrepancy/runs/engineering-vision20-walker-dt-20261105-sv2/；正式根/data/Policy_Discrepancy/runs/formal-m2-vision20-walker-dt-20261105-sv2/。本机证据/计划/Git包/DELIVERY在analysis/outputs/vision20-walker-dt-sv2-20261105/，不提交模型/replay/凭据。完整argv、环境、UUID/EGL、首次更新/评价/W&B启动读回及实际SHA在后续记录。

首选继续当前03完成本迁移启动及队列/终点审计，全40按所有claim去重替代源槽位，四排队不生成第五份训练。hopper_stand Dt仍单列阻塞；sv1三条原训练仍需空间留存，不擅自清理/停止/换卡或宣称恢复无缝。新科学决定交02/用户，不自动创建chat或发送消息。

## 正式启动与最终核查

10:31:58.940正式walker_run Dt在sv2 GPU4启动，监督器3186189、agent3186204。实际训练bef2948、控制器2f7e635，真实size50m CUDA短程已完成517合法更新/3完整单episode点评价/2次报告只读检查，所有原严格审计通过；正式命令和完整父环境在formal-supervisor-command.json。

10:34:49正式启动审计通过：0动作/0更新固定快照恰好10完整episode、86次连续成功更新、invalid0；manifest SHA/完整config精确一致，实际全部C/G属于该agent树、CUDA/EGL物理GPU4同卡独占、子进程无CPU/XLA/交互CUDA路径污染，W&B running/完整config及精确startup JSON/hash写入读回一致。正式百万步尚未完成，不把短程或启动验收写成方法收益。

10:35:29—34复核：sv2 GPU0—4五条运行、0/1/2各一条Dt仍按依赖queued，余249297391616字节；sv1 GPU0/1/5三条原训练仍运行，最近更新invalid0，walker baseline97.7328万动作，另两条约80万动作。旧walker Dt行仍queued但无PID、源目录仅MOVED_TO_SV2.json，无训练产物；原监督器及baseline存活，占位仍有效。全40正确去重后28训练结束（含2复用、26新增单审待整批收尾）/8运行/3排队/1阻塞，三个排队全部在sv2。sv1余17094717440字节、低于60GB，新迁移未解决其留存容量；hopper_stand Dt不在本次四排队范围、仍旧阻塞。sv2本项目仍占0—4，5/7空闲、6不用；sv1本项目仍占0/1/5，本次无停止/释放这些作业。

46小文件逐项镜像SHA256通过，包45350字节、SHA256 `2fd595e232a3fe1be26921403a9c3236db939405febcd2fa32a63f93721d49f6`。仅JSON/完整行前缀JSONL/config/stdout，排除checkpoint/replay/eval_snapshots/wandb二进制，不是可恢复训练备份；晚于打包的source-withdrawal-final.json另行复制到本机，首次SSH传输关闭后重试成功，未影响训练。最终DELIVERY.json记录完整最终/CPU测试/实际CUDA训练SHA、仅文档差异及同步范围；活动控制器/训练checkout不随最终文档提交移动，其他chat工作区草稿保留。
