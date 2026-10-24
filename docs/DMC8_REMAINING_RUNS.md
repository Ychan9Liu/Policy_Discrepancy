# 剩余八任务 · sv1/sv2 baseline→Dt

2026-10-24，03/04合并窗口。用户明确“sv1 GPU2、7，sv2 GPU0–5，跑剩下的任务”。本批承接原DMC16清单的后八个环境，**8任务×logging baseline/Dt，共16条**，每卡同一任务先baseline，逐条审计通过后从头启动Dt。前八任务的旧缺项另列，不用它们替代本批任务或改变本批资源分配。

## 当前分配与声明

| 服务器/实际host | 物理GPU | 运行任务 | 卡内顺序 |
| --- | ---: | --- | --- |
| sv1 / lyg2153 | 2 | dmc_finger_turn_easy | baseline→审计→Dt |
| sv1 / lyg2153 | 7 | dmc_finger_turn_hard | baseline→审计→Dt |
| sv2 / lyg0360 | 0 | dmc_hopper_stand | baseline→审计→Dt |
| sv2 / lyg0360 | 1 | dmc_pendulum_swingup | baseline→审计→Dt |
| sv2 / lyg0360 | 2 | dmc_reacher_easy | baseline→审计→Dt |
| sv2 / lyg0360 | 3 | dmc_walker_run | baseline→审计→Dt |
| sv2 / lyg0360 | 4 | dmc_walker_stand | baseline→审计→Dt |
| sv2 / lyg0360 | 5 | dmc_walker_walk | baseline→审计→Dt |

冻结配置不变：M2，clean纯视觉64×64 RGB、size50m、训练seed0、alpha20、free_nats1、dyn/rep=1/.1、ac_grads=false、reward_grad/repval_grad/repval_loss=true；batch16×64、replay_context1、16训练env、train_ratio256、repeat1、CUDA/bfloat16、prealloc=False。每条100万动作含预填充、不含reset/评价；0..100万每5万评价，共21点，每点恰好10完整episode，固定快照、采样动作、独立评价流。报告在正式配置关闭；训练/report/评价/shuffle的原RNG实现和派生不变。

每任务自身完整logging baseline按成功update开始动作 `[100000,300000)` 对有效loss位置raw rep KL>1累计S/N，候选w=1/(1+20D)、实际训练权重1；正式c供审计，不传给Dt。本批无constant/shuffle，不使用工程c、默认c或调alpha；N=0/非法值停止报告。保持原始return、实际x梯形积分/固定百万分母、80/85/90/95/100万指定末段五点和缺项不重加权。仅seed0两组描述，不支持强M2选择性归因或跨seed稳定性。

正式根分别为 `/data/Policy_Discrepancy/runs/formal-m2-dmc16-remaining8-serial-20261024-{sv1,sv2}/`；工程根为 `/data/Policy_Discrepancy/runs/engineering-dmc8-remaining-20261024-{sv1,sv2}/`。运行入口仍为m2_dmc16/protocol_dmc16，受限控制器 `scripts/run_dmc8_remaining.py`，结果profile为 `dmc8-remaining`。W&B继续在线 `ychan9liu-tsinghua-university/PD_1`，唯一run ID/resume=never，JSONL同时保留。

22:34:14北京时间只读核查：指定8卡全部0MiB、无C/G PID；sv1其他0/1/3/4/5/6卡有Ray未知作业，sv2 GPU7有未知Python作业，均不使用/不干扰。22:38再次核查资源与依赖，sv1可用590.5GB、sv2可用1,077.4GB；后续启动仍逐卡刷新，旧快照不是预约。两台env `/data/Policy_Discrepancy/envs/dreamer` 的Python3.11.16、JAX/jaxlib0.4.33、numpy1.26.4、dm-control1.0.48、mujoco3.15.0、elements3.22.2、ninjax3.6.3、wandb0.30.0一致，pip check均通过，本次未安装/修改依赖。UUID及全部版本见工程根resources/runtime-environment JSON。

## 工程实现与边界

共享first8控制器增加带原默认值的工程参数（host/GPU集合、validator、容量检查、finisher、进程session），旧入口/范围保持；新控制器固定两台server、GPU和任务，不做自由调度。agent/RSSM/rep_probe/JAX agent/协议runner/DMC源码、目标/梯度/reduction/RNG均未修改。每台只运行自己的2或6任务，JSON索引明确host shard；本机生成的16条配置、8任务随机流及两台计划固定同一Git提交。

新训练进程各自拥有独立POSIX session/process group。baseline退出审计后若仍有GPU进程，只在可核实其属于该baseline进程组时有界等待释放（90秒）；遇到未知/外来作业立即阻止后继，不等待抢卡、不共享、不替换。启动前再次require_free；记录实际占用/进程组证据。这处理可能的退出清理竞态，不认定旧六条阻塞已被证明由清理竞态造成。旧记录没有阻塞时完整PID所有权证据。

每台完成只标本机shard完成，并生成逐任务指标/W&B读回。`dmc8-remaining` 仍要求完整八任务，单台缺另一台的总体指标为null，不因sv1只有2任务或sv2只有6任务而重新归一化。两台完整审计产物汇齐后在独立CPU输出中执行require-complete，才能标整个八任务完成；不自动与前八/原四任务合并。

监督器拒绝已有campaign根，失败/重跑使用独立尝试，不覆盖、不将环境/replay不完整恢复称为无缝。容量门槛延续每条60GB工程估算，sv1四条至少240GB、sv2十二条至少720GB；不改replay/检查点配置。此门槛不是运行期空间保证，需持续检查。

## 前批状态核查（与本批分开）

22:40:34北京时间只读核对旧 `formal-m2-dmc16-first8-serial-20261022/supervisor-final.json`：**complete=false**；八条baseline和cheetah_run、finger_spin的Dt已结束并有逐条audit，另六条Dt因当时GPU占用被阻止，目录未创建。不能写“原16条已完成”，也不由本次新8任务补齐旧统计。此处读取已有审计字段，不冒充本轮重新完整验收；源hash及逐条final/c/audit索引保存在本次sv1工程根 `previous-first8-state.json`。本轮不重开或挪卡运行旧六条，后续缺项需用户明确资源安排。

## 验收进度与下一步

本地受影响CPU测试、两台CPU回归/显式float32 agent、设备CUDA/EGL同卡探针、生产线上W&B和八任务×两模式真实size50m CUDA/bfloat16短程按同提交验收后再启动。工程用4098动作、每2049动作评价1完整episode、窗口 `[2200,3600)`、报告间隔2049，均不进入正式配置。当前报告是启动前声明，实际测试/启动提交与证据后续补记；未把工程运行写成正式结论。

首选继续当前03窗口完成本次授权部署和启动，之后监测同批16条至每条百万动作/21×10/正式c/收据/快照/原始return与W&B审计完成，再供05分析。现存科研边界仍交02与用户决定，不扩大任务/seed/组、不自动建chat或发消息。
