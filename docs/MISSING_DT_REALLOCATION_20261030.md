# 03合并窗口 · 十一条Dt后续资源分片与队列（2026-10-30）

本报告接续`MISSING_DT_RUNS_20261030.md`，保留此前启动/冲突证据。用户先答复sv2先跑GPU1/3/5/7四条、其余三条等待，随后为剩余三条分配sv2 GPU1、sv3 GPU1/3。合并执行，不把第二次资源追加理解为取消前四条。sv2 GPU1原已安排cup，因此acrobot在其后串行；不共享GPU、不调用sv2 GPU6，不使用虽已空闲但未纳入当前计划的sv2 GPU0。

**2026-11-01执行更正**：10月30日后续七条只完成工程验收，正式启动/队列没有执行；中断不等于后台会继续。11月1日23:32重新核查后，按已有授权于23:34实际启动六条，23:35挂acrobot明确依赖队列，23:37—38启动审计通过。原sv3四条已于10月31日01:30全部accepted。当前完整状态和证据见`RUNS_STATUS_20261101.md`，下文准备版本和工程证据保留。

## 实际安排与冻结配置

所有task均为`dmc_`前缀、Dt、训练seed0：

| 服务器/物理GPU | task | 安排 |
| --- | --- | --- |
| sv2/1 | cup_catch → acrobot_swingup | 前驱完整训练且审计通过后串行，先检查卡无外来作业 |
| sv2/3 | cartpole_balance_sparse | 独立运行 |
| sv2/5 | cartpole_swingup_sparse | 独立运行 |
| sv2/7 | hopper_stand | 独立运行 |
| sv3/1 | cartpole_balance | 独立运行 |
| sv3/3 | cartpole_swingup | 独立运行 |
| sv3/4、5、6、7 | pendulum_swingup、reacher_easy、walker_run、walker_walk | 原14:21已启动四条，保留d8997f6运行与目录 |

只补原DMC16中缺失的11条Dt，既有21条baseline/Dt不重跑、原baseline c来源及日志不改。M2-v1-dmc16-seed0-baseline-dt / m2_dmc16：clean、64×64 RGB视觉、size50m、seed0、alpha20、free_nats1、dyn/rep=1/.1、ac_grads=false、reward_grad=true、repval_grad=true、repval_loss=true；batch16×64、replay_context1、16训练env、train_ratio256、repeat1、CUDA/bfloat16、prealloc=False、正式report关闭。每条百万训练动作含预填充，不含reset/评价；0..百万每5万共21点，固定快照、采样动作、独立评价流，每点恰好10完整episode。JSONL+W&B `ychan9liu-tsinghua-university/PD_1`，独立ID/resume=never。

自身已完成logging baseline按成功更新开始动作`[100000,300000)`、raw rep KL>1有效loss位置、候选`1/(1+20D)`累计S/N生成c；logging训练权重1，正常replay重采样保留、仅同update ID重复去重。Dt不用c；不从工程c或默认值替代。原始return AUC按实际x积分/固定百万分母，末段只取80/85/90/95/100万五点，缺失不重加权；沿用EXPERIMENTS C/C.1，不新增科学选择或效果结论。

## 实现、版本和隔离

- `f6916c89dfdde603c4014b35fac9167cec0ca9d5`：受限sv2四卡分片；`a0438a09a3067f5ef4d1fe21c452cb4ac18ce932`：剩余三条重分配/指定项目依赖及测试修正；实际新批次代码/验收目标 **`c7689d50f4caed0fef89912c65280b8d2558b126`** 补齐sv3 GPU1/3探针许可。
- 只改`scripts/run_missing_dt.py`、`tests/test_missing_dt.py`，不改agent/RSSM/probe算法、训练目标、梯度、优化器、奖励、计步、评价或RNG实现。新subset严格四任务/四卡/独立目录，新remainder严格sv2一条/新GPU1及sv3两条/GPU1/3，解析的科学配置与旧Dt仍仅差logdir，拒绝任意增删/变更卡/参数。
- **生产队列使用独立CPU辅助版本`2cccd5e006a639f10699907ad3f5b2f49c2c0b56`的`scripts/queue_missing_dt.py`**，实际训练仍固定c7689d5。c768的内置`await_predecessor`在前驱agent已退出、监督器尚审计时会因agent PID消失而提前失败；生产队列不直接以此轮询。辅助器检查确定cup目录/任务/代码版本及其监督器，覆盖“agent退出但审计未结束”窗口，仅见到成功审计标记后才调用c768 runner；runner再全量核原始输出、刷新GPU1和磁盘门禁。前驱失败/监督器退出/不匹配即失败，不等待外来作业空闲、不自动恢复或覆盖。此辅助器只控制CPU等待/启动，不修改训练目标、梯度、数据或RNG。
- 两服务器分别在共享Git对象的独立工作树`/data/Policy_Discrepancy/checkouts/missing-dt-c7689d50f4ca`运行新任务，canonical `/data/Policy_Discrepancy/repo`仍干净固定d8997f6，不切换既有sv3四条训练checkout。中间a043工作树仅在sv2准备、未执行训练，保留可追溯记录；未宣称以中间版本完成正式验收。
- Git bundle SHA256 `3dff8da6577fe9042a56f73562286f1e44dfa30a412d0ea487a483241284577b`，两服务器均verify/fetch并核新worktree SHA/干净状态，canonical HEAD未动；不在服务器手改项目源码，无sv1/GitHub同步。SSH/SCP认证前间歇断开后同入口重试，未改变登录信息或干扰作业。

## 测试与启动前资源

本机14项受限分片/队列测试通过（5.995秒）；三任务映射、三条来源、同机GPU1依赖、只检查/启动四张批准卡、未启动槽位不覆盖、失败前驱不释放、科学配置不变和原有范围门禁均保留。初版执行fixture把Path.open整体mock，使配置YAML读取被截获直至MemoryError，保留首轮失败日志；修正为先真实验证完整plan，再只隔离执行阶段IO，未放宽算法测试或容差，未影响服务器训练。

14:44:47资源快照：sv2 GPU1/3/5/7无C/G PID，GPU2/4/6为非本项目作业，GPU0虽空闲本轮不使用；sv3新GPU1/3无C/G，4—7原本批继续运行，0/2外来作业未干扰。/data余sv2约806GB、sv3约704GB。UUID/EGL必须实测，启动前再刷新资源；不能将此快照当未来预约。环境沿用`/data/Policy_Discrepancy/envs/dreamer`及base-py311，无包变更。

c7689d5在两机各114项CPU回归通过；各机三变体float32完整agent参数/计数/收据精确一致，沿用旧同机digest（sv2 41a913...、sv3 3c8d1e...）。新7条真实size50m CUDA/bfloat16短程全部通过，各4098动作/517更新/3×1评价/2报告，合计3619更新/21点评价/14次报告，报告与评价隔离指纹通过。acrobot复用同机同SHA的CPU/logger/GPU1 probe并逐文件记hash，但真实本任务短程单独运行。CUDA短程不是百万步结果。

queue辅助器在本机17项和11月1日sv2实际17项测试通过；辅助器新增3项专测退出/审计过渡、成功审计字段、失败/消失监督器拒绝。最终17项实测SHA为2cccd5e0，CUDA与主回归为c7689d5，原sv3四条为d8997f6，不能混称同一实际测试版本。命令和每次argv/env白名单保存在各工程根；动作/ID/配置/hash精确检查，算法既有容差不变。

## 目录与复现

工程根（均`/data/Policy_Discrepancy/runs/`下）：

- `engineering-missing-dt-20261030-partial1357-sv2/`：cup/balance_sparse/swingup_sparse/hopper_stand四条。
- `engineering-missing-dt-20261030-remaining3-sv3/`：balance/swingup两条。
- `engineering-missing-dt-20261030-remaining3-sv2/`：acrobot，复用同SHA/同机CPU和logger/probe证据并逐文件记来源hash，单独重跑本任务CUDA短程。

正式根：`formal-m2-dmc16-missing-dt-20261030-partial1357-sv2/`、`formal-m2-dmc16-missing-dt-20261030-remaining3-sv3/`、`formal-m2-dmc16-missing-dt-20261030-remaining3-sv2/`（acrobot仅前驱通过后创建）。每task为`dt-attempt02/`，W&B ID使用对应campaign/task/seed0/dt/02，不复用旧blocked或其他分片身份。原sv3四条仍在`formal-m2-dmc16-missing-dt-20261030-sv3/`。

```sh
# 以下为复现说明，现有目录/监督器不要重复运行。
cd /data/Policy_Discrepancy/checkouts/missing-dt-c7689d50f4ca
/data/Policy_Discrepancy/envs/dreamer/bin/python -m scripts.run_missing_dt bind --plan ENGINEERING_ROOT/SERVER-plan-c7689d5.json --evidence ENGINEERING_ROOT --output ENGINEERING_ROOT/bound-plan.json
/data/Policy_Discrepancy/envs/dreamer/bin/python -u -m scripts.run_missing_dt run --plan ENGINEERING_ROOT/bound-plan.json
# acrobot必须通过独立队列辅助器启动；不要直接把其run命令作为等待器。
cd /data/Policy_Discrepancy/checkouts/missing-dt-queue-2cccd5e006a6
/data/Policy_Discrepancy/envs/dreamer/bin/python -u -m scripts.queue_missing_dt --plan /data/Policy_Discrepancy/runs/engineering-missing-dt-20261030-remaining3-sv2/startup-20261101/bound-plan.json
```

全部CPU父进程使用approved_context，清除交互CUDA12.4注入/LD_PRELOAD/PYTHONPATH等；训练子进程独立session、CUDA UUID/EGL显式绑定、清除CPU/MPS flags，保留正式随机配置。所有根拒绝覆盖/不一致resume，环境/replay不能完整恢复，不称无缝续跑。

## 完成标准与限制

新六条已于11月1日启动，acrobot仍明确排队，实际PID/映射/线上验收见`RUNS_STATUS_20261101.md`及各工程根`startup-20261101/`。待正式终点验收每条百万动作/249493成功更新、21×10、合法连续收据/快照、原始return完整指标、线上finished/配置和精确JSON/hash。各Dt-only分片完成仅shard_complete=true/cohort_complete=false，后续须统一旧21条与新11条的全32条来源和结果，缺项不重加权。不能把排队acrobot写成正在训练，也不能把工程短程/启动通过写成方法有效。

跨服务器CPU参数digest不逐位相同、真实环境/随机输入微扰及hopper gate因果、原sv1 GPU7短时并发等既有限制保留。两组单seed不能单独证明强M2选择价值或跨seed稳定性；新科学协议交02与用户确认。

本机准备小证据/plan/测试/同步在`analysis/outputs/missing-dt-reallocation-20261030/`；11月1日实际启动/状态核查、最终交付及prompt在`analysis/outputs/missing-dt-start-20261101/`（均Git忽略）。完整产物留各服务器独立runs目录，不提交checkpoint/replay/凭据。首选继续当前03监测新六条及acrobot队列，完成其终点与全32条审计；原四条已完成，不重跑。完成后由用户交现有05分析；无需新建chat、不自动发送消息。
