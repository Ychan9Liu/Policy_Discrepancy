# DL-pilot-r3 1M 运行记录

## 终点验收与结果复核（2026-10-15）

**验收完成。** 2026-10-14 00:58:33.255724北京时间正常结束，worker exit0、父audited_complete；
1M动作/249493更新/21点×10完整episode/210000eval动作齐全，00:58:34.233197确认GPU4释放。
`final_state.status`保留worker结束时写入的`training_ended_pending_audit`；后续验收完成以
`audit.json`、`supervisor_complete.json`及本次独审`review.json`为准，不覆写原始阶段记录。
实际训练SHA仍 `13ec72a214b0c03b886b2c7fac52c8f089966dec`，无重跑、拼接或配置调整。
独立CPU复核actual/tested `a4d9330a86cb89f9251b00bdad755f3305244a1a` 的4项Linux契约全过，
51.383558453秒/exit0核完21个实际快照、最终checkpoint、全部收据、状态/hash/来源及原始统计。
复核是只读/hash/计数/算术，未重演episode或再次训练。

N AUC652.2809110338405、tail941.5184525181727、endpoint950.7499831742771；
相对B的AUC+2.21%，相对D−18.08%，尾段差小。平均v=.9996853471、
超额KL损失数值减免.346618%，本轮没有建立双信号优于单Dt的证据。
完整曲线、保护统计、可证伪解释和下一步建议见 [DL_PILOT_R3_ANALYSIS.md](DL_PILOT_R3_ANALYSIS.md)。
本次GPU4作业已释放；10/15 14:01 GPU4/5各有647MiB未知C作业（AccessDenied），6/7空闲；未干预。
14:05:14再次核查时，6/7也各有1127MiB未知C作业；最新资源记录在复核根
`resource_check-20261015-final.json`，不将14:01快照视为当前四卡状态。
以下13:37启动信息为历史快照；当前状态以本节和分析页为准。

## 历史启动快照（2026-10-13 13:37:29 北京时间）

**1M已从头启动并有效训练，运行中，尚未完成终点评价和验收。**
实际执行及本轮clean Linux测试完整SHA：`13ec72a214b0c03b886b2c7fac52c8f089966dec`。
sv2隔离checkout `/data/Policy_Discrepancy/dl-worktrees/13ec72a`；canonical原PD_1
`/data/Policy_Discrepancy/repo` 保持48d71f1541721208322442f5a1d5977124f3e888且干净。
13ec72a已push并ls-remote核验，GitHub/main为595e982f1a3939775879728009e2105ee516e86d，未合main。
后续报告提交仅文档，执行checkout不移动；最终报告SHA及与测试版差异以终点Git核验为准。

实际任务/组：sv2物理GPU4 / `dmc_quadruped_walk` / N / seed0 / 1000000实际训练动作。
GPU5–7本次不增加任务。α20/ρ.5/H1 reward-support、size50m/vision64/repeat1、
batch16×64/context1/16env/ratio256、free nats1/dyn1/rep.1及原梯度开关保持。
评价0及每50k至1M，共21点×10完整sampled episode；不新增c/κ匹配或shuffle。
完整规范及已确认/采用初值/假设区分见 [DL_PILOT_R3.md](DL_PILOT_R3.md)。

13:37:29最后完整收据ID2128，2129成功更新、动作10544，loss25.25370979309082、
grad norm63.014076232910156均有限。初始294叶完整params/optimizer/norm与PD_1共同状态
逐字节一致；0点10×1000完整episode，mean raw return56.29089345829127，训练状态不变。
无FAILED或supervisor_failed，supervisor正常等待、worker运行中。
1M/249493更新、余下20点评价、完整审计和最终释放仍未完成；没有方法性能结论。

## 产物、版本与完整命令

- 计划：`/data/Policy_Discrepancy/runs/dl-pilot-r3-plan-20261013-13ec72a`
- 运行：`/data/Policy_Discrepancy/runs/dl-pilot-r3-20261013-13ec72a/dmc_quadruped_walk/N-seed0-attempt01`
- 监督：计划下 `supervisor-attempt01/`（startup/device_probe/worker.log/resources.jsonl及终点记录）。
- CPU预检：`/data/Policy_Discrepancy/runs/dl-pilot-r3-preflight-20261013-13ec72a`。
- 本机小型镜像：`analysis/outputs/dl-pilot-r3/13ec72a/`，ignored，不提交checkpoint/replay。

在上述隔离checkout执行：

```text
/data/Policy_Discrepancy/envs/dreamer/bin/python scripts/dl_pilot_train.py launch \
  --index /data/Policy_Discrepancy/runs/dl-pilot-r3-plan-20261013-13ec72a/RUN_INDEX.json \
  --output /data/Policy_Discrepancy/runs/dl-pilot-r3-plan-20261013-13ec72a/supervisor-attempt01 \
  --git-sha 13ec72a214b0c03b886b2c7fac52c8f089966dec
```

13:28:12.149590父PID673688启动，实际worker674490，独立owned process group。
父在初始化/编译/评价期间仍监督，57600秒whole-entry硬上限；正常终点自动执行连续收据、
评价/训练状态/设备/统计审计并写audit与supervisor_complete，随后记录实际GPU释放。
监督运行独立于本机SSH连接；这不表示训练或科学验收已经完成。
仅该owned group可在故障/上限时清理，不操作未知任务，不自动延长或增加任务。

| 证据文件（相对计划目录） | SHA256 |
| --- | --- |
| RUN_INDEX.json | c11502c195e41bb243590f6877b2a0ed0f59d5c1fb44ebd0024460f59a9e0333 |
| config.yaml | 877f8470cb4f7f45d143e3099fbd3f55a828991afe33779b7b04c4264f6c8377 |
| historical_controls.json | 5d356c9c6c3b21a34f8b4c43ac72a21dc8b0c9d54640b30e43b7dce3e27aa161 |
| supervisor-attempt01/device_probe.json | fdf2d32b0c9b28742a2aa0e122819f06a8c6db153b7c0a11a93cf6da35f559b6 |
| supervisor-attempt01/startup.json | 3bfde8f51e65f353ad310a7483200f6452f4ed00f3a02408260b7b1c891163d3 |

13:37:29完整现场、PID与上述hash保存于计划 `post_start_check.json`。
模型/环境/依赖和初始digest由manifest记录；logger为JSONL，不新建PD_1 W&B run。

## 实际资源与工程验收

13:28:12.147840 fresh完整XML核GPU4–7各0MiB/无进程；本次零动作probe673693于
13:28:13.471526–13:28:15.414652实际CUDA/EGL均仅GPU4，probe退出后再核空闲启动worker。
13:37:29.405915完整XML：GPU4 8010MiB、1个C+G worker加16个G环境进程全部本项目；
5–7各1MiB但无compute/graphics进程，本次未用。GPU4仍有新r3作业，未释放。
UUID `GPU-7bf636ec-fa03-aa3f-7314-dd803c1658be`；UUID可见/CUDA0、
`CUDA_DEVICE_ORDER=PCI_BUS_ID`、`MUJOCO_EGL_DEVICE_ID=4`、`MUJOCO_GL=egl`、
`PYOPENGL_PLATFORM=egl`。父每约30秒记录实际owned PID/UUID/process group映射。
8010MiB为当前采样占用，不是峰值或最低显存保证。

13ec72a的clean Linux新5项契约全部通过（2.917993903秒），pip check0（.915287798秒）；
覆盖完整1M评价seed网格/AUC/tail、配置锁、实际worker double/评价隔离/NaN停止、
真实Linux owned及嵌套进程超时清理。fixture不是实际任务或方法效果证据。
相对已测95ea434，dreamerv3/与embodied/源码diff为空；原95的13kernel/14protocol
检查（25通过/2skip）按相同源码认证复用，不能写成13ec72a新跑32项。
JAX profiler的可选TensorFlow trace导入报错保留在worker.log，后续实际更新继续，未改依赖/配置。
XLA关闭并行编译的driver/PTX版本提示同样保留；没有因此修改训练行为。

## 原100k中止与历史对照

r2实际95ea434在37096动作/8767更新时按用户“改为1m”中止；旧全部产物和
`user_budget_correction.json` 保留，父TimeoutError不当数值失败。旧父/worker/环境已清理；
新r3不续跑、不拼接。详见 [DL_PILOT_R2_RESULTS.md](DL_PILOT_R2_RESULTS.md)。

原PD_1 B/D实际训练SHA48d71f，全部21点×10episode和实际配置/hash经r3计划复核。
同root0初始化与种子规则、1M预算、raw return口径一致；代码版本及渲染/轨迹不保证逐字节相同。
尤其共同初始化不代表0点评价必须完全相等，旧B/D及r2/r3已有轻微差异。

| 原PD_1历史组 | 1M AUC/预算 | 80–100万tail | 1M终点 |
| --- | ---: | ---: | ---: |
| B | 638.158807614521 | 937.7926031842344 | 940.783630606081 |
| D | 796.2536301200965 | 944.8860246267989 | 917.4703475542723 |
| 新N（现已完成） | 652.2809110338405 | 941.5184525181727 | 950.7499831742771 |

当前同预算比较已完成，详见分析页；单训练seed/历史控制、10eval episode不支持跨训练seed推断。
旧native/独立信号/选择性保护机制不足仍保持；不能把新在线启动写成前验证通过。

## 验收标准与后续

本次授权的1M执行与审计已完成。首选由现有02审查第二信号与强度匹配设计，
具体可立即做的CPU工作、未决条件及未批准新实验见分析页；不自动创建或发送chat。
不从旧checkpoint续跑、拼接、调α/ρ/H/loss/seed/评价或增加任务，不抢占/更换资源。
正常终点检查1000000动作/249493更新、210000eval动作、21×10完整episode/快照、
更新收据/opt有限、21点评价训练状态不变、实际CUDA/EGL同卡、exit0/父audit及资源释放。
上述项目现经父审计和独立CPU审阅通过，小型证据已镜像，结果及分析已归档。
最终完整报告SHA、实际tested SHA差异、main/GitHub与同步范围按最终Git核验报告；
工程通过或单seed回报均不写成机制成立。
