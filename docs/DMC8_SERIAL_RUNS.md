# sv1 · 新增前八任务 baseline→Dt 串行运行

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

代码修改不涉及agent/RSSM/rep_probe、训练目标、梯度、环境或runner计算路径。共享探针/audit仅增加可选host/允许卡/协议参数，旧Hopper默认行为保留。控制器、配置范围、串行屏障及结果profile的CPU测试通过后提交，再同步同一确定提交到sv1。完整CPU agent和八任务×两模式真实size50m CUDA/bfloat16验收尚待执行；**当前文档不是已启动或已通过CUDA的证明**。

工程验收使用独立4098动作、间隔2049、每点1完整episode、窗口 `[2200,3600)`，报告间隔2049；这些缩小值仅用于工程测试，不能进入正式配置。真实报告/评价执行前后检查完整训练状态指纹。已有40项本机CPU检查通过；最终测试SHA、实际结果、bundle同步、正式启动及运行状态将在本文件后续补记。

## 操作入口与完成标准

控制器操作 `prepare → probe(8卡) / logger / integration → bind --authorized → run`；准备产物未绑定时不可启动。bind核对同SHA的16条真实CUDA验收、8卡实际CUDA/EGL同卡及在线W&B证据。run只启动8条baseline，完成严格审计后每卡才启动Dt。最终审计复核21×10评价、update连续性/无重复/非法值、精确预算、c来源和快照，再生成八任务独立结果表并逐条发布可读回的W&B指标。

优先在当前03合并窗口完成部署与启动，持续审计同批16条。只有全部训练和产物/W&B审计完成才能标“验收完成”；单seed两组描述不能证明强M2选择价值或跨seed稳定性。不自动创建chat或向其他chat发消息。
