# sv1 · 新增前八任务 baseline→Dt 串行运行

**2026-10-30复核**：仍为10条已结束且有逐条审计、6条Dt从未启动，supervisor-final complete=false；本批无运行进程。全批统计/线上摘要尚未齐全，当前sv1空间仅约52GB，补跑不能直接照旧重开或删原产物。全批与资源记录见 `RUNS_STATUS_20261030.md`。

**2026-10-24状态更正索引**：只读核查旧supervisor-final：8条baseline+cheetah_run/finger_spin的Dt共10条已结束并有逐条audit，另6条Dt因当时GPU占用而未创建目录，整体complete=false。本报告下方启动时状态保留，不表示16条最终齐全。当前用户分配资源用于后八环境，见 `DMC8_REMAINING_RUNS.md`；旧6条缺项仍需用户明确资源后继续，不能擅自挪到本批卡上。

2026-10-22，03/04合并窗口。用户将暂停的32条计划缩小并恢复为**8任务×2组，共16次训练**，明确“取前八个”。其余八任务仍未启动。旧准备报告 `DMC16_PREPARATION.md` 记录暂停时状态，本文件是本批当前执行记录。

## 已声明配置与资源

| sv1物理GPU | 运行任务（省略dmc_） | 卡内顺序 |
| ---: | --- | --- |
| 0 | acrobot_swingup | logging baseline → 审计 → Dt |
| 1 | cup_catch | logging baseline → 审计 → Dt |
| 2 | cartpole_balance | logging baseline → 审计 → Dt |
| 3 | cartpole_balance_sparse | logging baseline → 审计 → Dt |
| 4 | cartpole_swingup | logging baseline → 审计 → Dt |
| 5 | cartpole_swingup_sparse | logging baseline → 审计 → Dt |
| 6 | cheetah_run | logging baseline → 审计 → Dt |
| 7 | finger_spin | logging baseline → 审计 → Dt |

共同配置沿用冻结v1：M2、clean纯视觉64×64、size50m、训练seed0、alpha20、free_nats1、dyn/rep=1/.1、ac_grads=false、reward_grad/repval_grad/repval_loss=true；batch16×64、replay_context1、16训练环境、train_ratio256、repeat1、CUDA/bfloat16、prealloc=False。每条百万训练动作含预填充、不含reset/评价，0..百万每5万评价10完整episode，共21点，固定参数快照、采样动作、隔离评价RNG。训练/report/评价/shuffle的原随机流实现不变。

每任务自身完整logging baseline按成功更新开始动作 `[100000,300000)`、有效活跃位置候选 `1/(1+20D)` 累计S/N并冻结c；实际logging训练权重为1。c供审计，不传给Dt，本批不跑constant/shuffle。无活跃位置/非法值停止报告，不填默认c或调alpha。顺序控制要求baseline训练及c/评价/收据审计通过，才启动同卡Dt；任一失败保留证据、阻止该任务后继，不重开、不覆写、不挪卡。

运行入口仍为 `m2_dmc16 / protocol_dmc16`；新增控制器 `scripts/run_dmc8_serial.py` 限制本次精确16条及物理卡映射。新的统计 `--profile dmc8-first8` 只汇总这八任务、seed0、两组；原四任务、hopper补seed、完整dmc16 profile保留。分层原始return AUC/固定百万分母与指定末段五点规则不变，缺项不重加权，不自动汇总为16/20任务结论。接球runtime ID为 `dmc_cup_catch`，对应成绩ID `dmc_ball_in_cup_catch`。

正式产物根计划 `/data/Policy_Discrepancy/runs/formal-m2-dmc16-first8-serial-20261022/`。工程验收与依赖/设备证据独立保存在 `/data/Policy_Discrepancy/runs/engineering-dmc8-serial-20261022/`。每条产物独立目录；W&B沿用 `ychan9liu-tsinghua-university/PD_1`、在线、唯一run ID、resume=never，同时保留JSONL。监督器拒绝已有campaign根，不将环境/replay不完整恢复称为无缝续跑。

## 启动前证据与实现状态

2026-10-22 20:20:47北京时间，sv1实际host `lyg2153` 八卡A100-SXM4-80GB全部0MiB/0%、无计算及图形PID。设备UUID完整记录在工程目录 `resources-before.json`；后续启动逐卡重新检查，不把旧快照当作预约。/data当时可用1,104,673,423,360字节。监督器启动容量下限960GB，仅是工程容量检查，不改变replay或检查点配置。

sv1原Python环境缺wandb；按 `env/requirements-wandb-pd1.lock` 添加10个包（wandb0.30.0等），既有包版本全部保持，pip check通过。依赖前后完整版本与安装日志均在工程目录。不读取或记录认证秘密；线上读写另行验收。

代码修改不涉及agent/RSSM/rep_probe、训练目标、梯度、环境或runner计算路径。共享探针/audit仅增加可选host/允许卡/协议参数，旧Hopper默认行为保留。实际代码、服务器测试、真实CUDA验收及正式启动版本均为 **`18b85c4d85ff456e6822626f17cb14cc0f44d2b8`**。后续交付只更新报告/索引，最终完整SHA见本机outputs的DELIVERY及最终回复；不将运行中的服务器checkout切到文档提交。

工程验收使用独立4098动作、间隔2049、每点1完整episode、窗口 `[2200,3600)`，报告间隔2049；这些缩小值仅用于工程测试，不能进入正式配置。真实报告/评价执行前后检查完整训练状态指纹。

### 已完成验收

- 本机41项CPU检查，17.079秒，全部通过；提交后sv1 **85项回归，44.709秒，全部通过、无跳过**。覆盖原phase1/2诊断、回退/梯度与shuffle、协议计步/评价/去重/恢复、原return指标、补seed/新增范围及新串行控制器。新检查覆盖精确8×2、同卡顺序、完成审计才启动Dt、失败只阻止该任务后继、未知C/G占用拒绝、未授权/未绑定/已有根目录拒绝，缺任务不重加权。原四任务及完整dmc16统计保留；新增16条合成CLI产物含3360 episode/336点，总体3.5/5.5仅为手算fixture。
- CPU/bfloat16完整连续agent完成no-report、with-report、少评价episode、并行环境、故意中断/恢复拒绝及Dt/constant/shuffle。三项报告/评价/并行路径参数、训练随机流、更新数精确一致；共同参数hash `4317c6095b04e61698f26beb6e7c077c6c5d49e1baf135a716e3b3849c7d74b2`。入口debug继承bfloat16，不能因platform=cpu称为float32。这些是dummy/debug fixture，不是DMC任务效果；非零作用的损失/梯度由保留的overlay测试覆盖。
- 显式追加 `--jax.compute_dtype float32` 的CPU完整agent补验，仍为18b85c4实际测试版本：no-report/with-report/one-eval三条各12动作、9更新、5点评价，退出码全部0；参数、收据、训练随机计数及共同评价episode精确一致，参数hash `41a913d9d50f02df34e387ba514354e51ca924aeefed36a643aebcd2fabb6be4`。完整命令与结果在 `cpu-float32-attempt02/acceptance.json`（20:49:54北京时间）。首轮错误地给协议runner传off模式，被四组限制在创建agent前拒绝，证据 `cpu-float32/off.stdout` 保留；没有放宽入口、修改代码或正式配置。第二次用独立目录及协议允许的logging模式完成补验。此条修正初稿的CPU精度标注，不改任何正式运行。
- 八张实际分配GPU的CUDA计算与EGL图形PID同卡验证通过；一张UUID mask对应进程内CUDA0，EGL实际索引依物理卡为0..7。记录在 `gpu{0..7}-binding.json`，包括枚举、PID、compute-apps及pmon，不仅凭环境变量判定。
- 八任务×logging/Dt **16条真实size50m/CUDA/bfloat16工程运行全部完成**，每条4098动作、517成功更新、3点评价（每点1完整episode）、2次报告。合计8272更新、48点评价、32次报告的训练状态指纹隔离检查通过，未关闭transfer guard；所有baseline匹配窗口活跃计数非零。logging实际权重精确1；Dt两段实际平均权重范围0.9993799448—0.9999476372，活跃有效比例范围0.4723072—1，证实工程中有实际门控作用，不能据此推断收益或解释hopper失败。
- 生产W&B logger线上写入/读回通过，工程run `logger-dmc8-20261022-18b85c4` 为finished、seed0、sentinel=1，0动作/0更新。没有使用离线替代线上验收。

### 同步与环境

只向sv1同步：以服务器原干净 `313532571f0a0e39759b9229264361c0fd9bd48f` 为prerequisite的Git bundle，双方校验SHA256 **`baf8ef8791f2b44a9bf8cbdce1d7158eee9b256d9f1e5f05a887bae594991f08`**；bundle verify、fetch及fast-forward后HEAD为18b85c4完整SHA、工作区干净。没有服务器源码手工修改、GitHub推送或sv2/sv3同步；已有本机RESEARCH/STATUS/proposal/TeX草稿保留。

sv1 host `lyg2153`，Python3.11.16，jax/jaxlib0.4.33，numpy1.26.4，dm-control1.0.48，mujoco3.15.0，elements3.22.2，ninjax3.6.3，wandb0.30.0，driver570.158.01。完整包版本、安装差异、环境/RAM/磁盘、设备映射见工程目录JSON。20:39:20北京时间启动前八卡再次无C/G进程，/data可用1,070,504,132,608字节，全部工程GPU进程已释放。

### 正式启动与尚未完成

2026-10-22 **20:39:41北京时间** 启动独立CPU监督器PID3990345；八条logging baseline已创建并运行，八条Dt状态queued，尚未启动。`actual-RUN_INDEX.json`、`prelaunch.json`、`bound-plan.json`、每条execution/config/manifest/stdout均保存在正式根；监督器日志与完整启动命令在工程根 `formal-supervisor.stdout` / `formal-supervisor-command.json`。JSONL与在线W&B PD_1启用，唯一run ID为 `{campaign}-{task去dmc_}-seed0-{logging或dt}-01`。

**20:44:22北京时间启动审计通过**：八条baseline均完成0点固定快照的恰好10完整episode，共80条；已观察到各470—701次连续成功更新，非法值计数0，实际manifest完整配置与绑定版本一致。136个C/G PID全部可追溯到本批8个agent及其环境子进程，物理GPU/UUID/CUDA mask/实际EGL一致，无未知作业。八个线上W&B run状态running，完整配置逐字段一致；用独立 `startup/*` 命名空间记录0点评价及当前更新快照，exact JSON/hash读回相同，不改训练logger、随机流或正式指标。原logger按显式step提交，0点history可能在下一step/结束前暂存；startup摘要仅说明启动快照，不冒充实时进度或正式AUC。证据 `formal-startup-audit.json`。该时刻八卡均为本批baseline占用，没有正式卡释放；全部短程工程进程已退出，Dt尚未创建运行进程。

正式百万动作、21×10评价、正式c、全部Dt及完整return/W&B终点审计仍待完成。现阶段只能称“已启动/运行中”，不能称16条正式验收完成。不能完整恢复环境/replay；监督器拒绝已有campaign根，失败证据保留，新尝试须独立标识，不把后续续跑称为无缝恢复。真实环境跨独立运行逐bit一致性及gate/输入微扰因果边界未由本次解决。

本机小证据镜像 `analysis/outputs/dmc8-first8-serial-start-20261022/`（Git忽略），包含bundle、CPU日志、server前后依赖/资源/探针、工程和启动审计。完整命令在服务器 `cpu-commands.json`、`probe-logger-commands.json`、`integration-command.json`、`formal-supervisor-command.json`，生产16配置和8任务随机映射在 `plan/`；其中 `unlaunched-dmc16-catalog/` 只是生成时的32条未授权静态目录，不能当作32条实际运行。实际RUN_INDEX仅16条。

```sh
# 服务器CPU回归完整模块列表见工程根cpu-commands.json。
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 OPENBLAS_NUM_THREADS=1 /data/Policy_Discrepancy/envs/dreamer/bin/python -m unittest -v tests.test_dmc8_supervisor tests.test_dmc16_preparation tests.test_hopper_supervisor tests.test_return_metrics
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 /data/Policy_Discrepancy/envs/dreamer/bin/python -m tests.run_protocol_fixtures --output_dir NEW_CPU_AGENT_DIR
# 下列审计只在全部训练结束后，对新的输出目录执行；不启动训练。
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 /data/Policy_Discrepancy/envs/dreamer/bin/python -m scripts.run_dmc8_serial audit --plan /data/Policy_Discrepancy/runs/engineering-dmc8-serial-20261022/bound-plan.json --output NEW_FINAL_AUDIT_DIR
```

## 操作入口与完成标准

控制器操作 `prepare → probe(8卡) / logger / integration → bind --authorized → run`；准备产物未绑定时不可启动。bind核对同SHA的16条真实CUDA验收、8卡实际CUDA/EGL同卡及在线W&B证据。run只启动8条baseline，完成严格审计后每卡才启动Dt。最终审计复核21×10评价、update连续性/无重复/非法值、精确预算、c来源和快照，再生成八任务独立结果表并逐条发布可读回的W&B指标。

优先在当前03合并窗口完成部署与启动，持续审计同批16条。只有全部训练和产物/W&B审计完成才能标“验收完成”；单seed两组描述不能证明强M2选择价值或跨seed稳定性。不自动创建chat或向其他chat发消息。
