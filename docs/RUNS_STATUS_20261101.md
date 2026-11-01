# 03合并窗口 · 全批状态与七条实际补启动（2026-11-01）

核查时间均为北京时间：23:32全体索引/资源，23:34补启动，23:37—38六条启动审计。**尚未全部完成：新增16任务×baseline/Dt的32条中25条训练结束、6条运行、1条明确依赖排队。** 训练结束、逐条审计、分片验收、全32条汇总分别记录，不将25条统称全批验收完成。

## 已完成与此前未完成的工作

- 原前/后八环境已有21条结束：本轮只读原sv1/sv2四个索引，10+2+2+7条仍为accepted或run_audited_pending_group_and_WB，退出0。旧17条仍待所属批次完整汇总/W&B收尾，本轮未重新全量重算其原始收据。
- 后补sv3 GPU4—7的pendulum_swingup、reacher_easy、walker_run、walker_walk四条Dt，10月31日01:30全部accepted，退出0；各100万训练动作、249493成功更新、21点严格0..百万每5万、点内10完整episode。11月1日交叉检查final/所有评价/尾收据/逐条audit，完成表SHA256 `efa624f58d08f56a00e2a36417ce05e82c878bc7bc9f9d2d89895538d6ae9746`与completion一致。线上四条finished、full config一致、正式原始return精确JSON/hash与源表读回一致；shard_complete=true、cohort_complete=false。
- 10月30日其余七条只完成工程验收，**正式训练和后台队列没有启动**。此前工作中断在实际启动之前；不能把工程passed、准备好的plan或中断当成持续后台执行。11月1日发现后继续已有授权补齐，不取消/要求重复确认，不重跑已完成四条。
- hopper_hop seed1/2/3 baseline/Dt六条原索引仍全部accepted，单列不自动并入新16任务。故新增38槽位为31条训练结束、6运行、1排队；历史原四任务四组16条保持原结果，本轮未重新审计其全部大产物，不合并科学指标。

## 七条当前实际状态

全部为Dt、训练seed0，task均带`dmc_`前缀：

| 服务器/物理GPU | task | 23:37启动审计状态 | agent/监督PID |
| --- | --- | --- | --- |
| sv2/1 | cup_catch | 运行，324更新，最近开始动作3324 | 1647319 / 1647284 |
| sv2/3 | cartpole_balance_sparse | 运行，289更新，最近开始动作3184 | 1647326 / 1647284 |
| sv2/5 | cartpole_swingup_sparse | 运行，304更新，最近开始动作3244 | 1647341 / 1647284 |
| sv2/7 | hopper_stand | 运行，337更新，最近开始动作3376 | 1647362 / 1647284 |
| sv3/1 | cartpole_balance | 运行，359更新，最近开始动作3464 | 875379 / 875315 |
| sv3/3 | cartpole_swingup | 运行，378更新，最近开始动作3540 | 875455 / 875315 |
| sv2/1 | acrobot_swingup | 排队，0动作/0更新，正式根未创建 | CPU队列1650585，尚无agent |

sv2六卡以外不借用：GPU6从未使用，GPU0/2/4不擅自改为当前运行卡。acrobot在同卡cup完整训练、终点审计成功之后才释放；仍要刷新GPU1无外来作业/空间门禁。失败、监督器退出或外来占用时保留证据并停止，不自动挤占/恢复/覆盖。

六条分别于23:34:33（sv2）/23:34:32（sv3）实际启动，**23:37:57启动审计全部通过**：零点评价各恰好10完整episode、动作0/update0固定快照；收据从0连续、invalid_count=0；manifest完整配置/运行SHA精确一致；102个C/G PID均属于对应六个agent及子进程，CUDA UUID/进程内cuda:0/EGL同物理卡。W&B六条running、full config一致、独立startup JSON/hash线上读回一致。最近开始动作不是精确实时总动作，启动审计不是正式终点或收益证据。

23:35:12独立CPU队列启动，queue-progress状态waiting_for_own_predecessor_or_its_audit、training_started=false且队列PID存活。原agent退出和监督器审计之间的窗口由CPU辅助器处理，不能仅因agent PID消失而放弃队列。生产使用queue_missing_dt，c768内置等待函数不直接用作长时队列；释放后以queue-released和实际正式索引/收据判断状态，不将等待时的快照继续当实时状态。

## 配置、版本与验收

共同冻结配置继续为M2-v1-dmc16-seed0-baseline-dt/m2_dmc16，clean仅64×64 RGB视觉、size50m、seed0、alpha20、free_nats1、dyn/rep=1/.1、ac_grads=false、reward_grad/repval_grad/repval_loss=true；batch16×64、replay_context1、16训练env、train_ratio256、repeat1、CUDA/bfloat16、prealloc=False、正式report关闭。每条100万动作含prefill、排除reset/评价；21点×10完整episode、固定快照/采样动作/独立评价随机流。JSONL+W&B `ychan9liu-tsinghua-university/PD_1`、独立ID/resume=never。自身旧logging baseline按成功更新开始动作[100000,300000)、raw KL>1有效loss位置累计候选1/(1+20D)的S/N，logging实际权重1；Dt不用c，不加constant/shuffle。原始return及缺失规则按EXPERIMENTS C/C.1，不因本次延期或换卡改变科学协议。

| 用途 | 实际完整SHA | 验收 |
| --- | --- | --- |
| 已结束sv3四条 | `d8997f64a301bfdcfe08758faeae8957a7395766` | 原CPU107/真实CUDA工程、百万动作终点与线上精确收尾 |
| 新七条训练/控制器 | `c7689d50f4caed0fef89912c65280b8d2558b126` | 两机各114 CPU、各机三变体float32 agent精确一致；7条真实size50m CUDA短程各4098动作/517更新/3×1/2报告；本次六条实际启动验收 |
| 独立CPU排队辅助器 | `2cccd5e006a639f10699907ad3f5b2f49c2c0b56` | 本机17项、11月1日sv2实际17项通过，覆盖退出/审计过渡及失败拒绝；实际等候状态通过，未来释放尚未发生 |

本次未新增训练源码修改，完成已有确定版本的同步/测试和实际运行，最后提交仅补工程报告/STATUS/实验索引。相对c768，queue辅助器和3项队列测试是CPU控制层差异；相对2ccc则仅文档差异。没有修改agent/目标/梯度/奖励/计数/RNG，没有放宽容差。两机canonical repo仍d899，独立训练工作树c768，sv2队列独立工作树2ccc；均保持确定提交和干净源码。Git bundle verify/hash及同步范围见工程记录和DELIVERY；只同步sv2/sv3，不同步sv1源码或推送GitHub。

## 资源与产物

23:33—34刷新六张指定卡无C/G PID，重新实际CUDA/EGL探针通过；同SHA原CPU/工程/logger证据逐文件记hash后复用。本次使用相同Python/env/依赖，无包安装，approved_context清除交互CUDA/LD/PYTHONPATH注入。启动前/data可用sv2 764,401,315,840字节、sv3 237,068,619,776字节，分别满足当前4条并行+后继一条及两条并行的容量门槛；不删除checkpoint/replay/base-py311。

23:37资源：sv2 GPU1/3/5/7、sv3 GPU1/3仅本批作业，未释放；原sv3 GPU4—7四条已结束并释放，23:32无C/G PID。sv3 GPU0/2为非本项目作业，不干扰。sv1仅核运行索引、未刷新GPU资源；任何快照不是未来空闲预约。

23:43:32交付复核：六条仍运行，连续更新2492—2558次，最近更新开始动作11996—12260，invalid累计0，六卡仅本批C/G。CPU队列1650585存活、CUDA_VISIBLE_DEVICES为空/JAX_PLATFORMS=cpu，仍等cup，acrobot正式根未创建。/data余sv2 760,121,102,336字节、sv3 234,693,234,688字节；未释放当前六卡，没有新增占卡/换卡。

正式根（均在`/data/Policy_Discrepancy/runs/`）：

- `formal-m2-dmc16-missing-dt-20261030-sv3/`：已accepted四条，保留原产物。
- `formal-m2-dmc16-missing-dt-20261030-partial1357-sv2/`：当前四条。
- `formal-m2-dmc16-missing-dt-20261030-remaining3-sv3/`：当前两条。
- `formal-m2-dmc16-missing-dt-20261030-remaining3-sv2/`：acrobot尚未创建，成功释放后创建。

工程对应根在`engineering-missing-dt-20261030-{partial1357-sv2,remaining3-sv3,remaining3-sv2}/startup-20261101/`，有bound-plan、完整argv、环境/设备probe、PID、启动审计、queue-progress和线上证据；旧23:34前工程验收在父工程根，不覆盖。实际启动日期为11月1日，目录名保留10月30日准备命名，不伪造旧启动时间。已结束四条新增completion-readback-20261101.json位于其原工程根。

本机小镜像/来源hash/状态索引/DELIVERY与继续prompt在`analysis/outputs/missing-dt-start-20261101/`；此前准备材料在`analysis/outputs/missing-dt-reallocation-20261030/`。不提交大产物、checkpoint/replay或凭据。完整配置信息及部署设计见`MISSING_DT_REALLOCATION_20261030.md`。

本机小镜像逐文件hash校验通过：sv2 191文件，压缩包SHA256 `422e5aa4dc33b34c011be5a2414f925b05bc637d7ef6da33fd00e0bedd6fc870`；sv3 268文件，`dfc35a9e83d67767ecf3d1d2ef26ef85f404738805429008900b782877a70f03`。仅配置/日志/工程与启动/终点小证据，运行中JSONL为当时完整行前缀，不当完整终点数据；最终完整SHA及Git对象同步记录在DELIVERY。

## 下一步与限制

首选继续当前03监测六条和队列，完成每条百万动作、249493更新/合法连续收据、21×10固定快照评价、完整raw-return/W&B收尾和实际卡释放，然后统一旧25条与待完成7条的全32条来源和指标审计。当前可以监测/只读核查，终点依赖训练结束，acrobot依赖cup成功与GPU1资源。不要重复启动现有目录或监督器。

Dt-only分片终点只代表shard_complete，不代表全32条cohort_complete；原17条所属批次汇总仍未完成。工程/fixture通过不代表真实终点或方法有效。CPU跨机digest不同、真实环境轨迹/输入微扰、hopper gate因果及旧GPU7短时并发等既有限制保留；单seed两组不支持强M2或跨seed稳定性。新科学复验仍由02与用户确认。全部审计后再由用户交现有05；继续本chat即可，不自动创建chat或发送消息。
