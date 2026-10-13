# DL-pilot-r2 100k 运行记录

## 用户更正后的最终状态（2026-10-13 13:19）

**按用户预算更正中止，未完成100k；不是数值故障，不进入性能比较。** 用户改为1M，
新协议见 [DL_PILOT_R3.md](DL_PILOT_R3.md)。13:19:26.895126核验父命令/PID后仅向本次父668492
发SIGTERM；父按已有owned group清理worker669298及16环境/辅助进程，13:19:28.365921退出。
最后完整收据ID8766（8767成功更新）/训练动作37096；0点10完整评价已留证，未到50k。
该attempt无final_state/完成审计。父原始TimeoutError失败记录保存，不解释为训练错误；
计划目录 `user_budget_correction.json` 明确用户更正、命令绑定、已清理PID与最后收据。
13:21:17.533450复核668492/669298均不存在，GPU4–7各0MiB；新r3从头开始，不恢复或拼接。
旧有限保护统计（约29.8k时平均v=.999976681）仅运行诊断，不形成性能或机制结论。
下方13:03为历史快照。

## 当前状态（2026-10-13 13:03 北京时间）

**运行中：13080/100000动作、2763成功更新，0点评价10完整episode已完成；尚无终点验收。** 实际/clean Linux测试完整SHA：
`95ea4348b8923d947cbb6e57a4d68dc5029c1fad`。代码仅同步sv2隔离checkout
`/data/Policy_Discrepancy/dl-worktrees/95ea434`，canonical保持原PD_1的48d71f且干净。
GitHub HTTPS在本机两次连接失败；当前新增提交未确认推送、未进main。

用户指定sv2物理4–7，实际只使用GPU4/A100-SXM4-80GB运行
`dmc_quadruped_walk` / N / seed0 / 100k；5–7不增加作业。完整配置和依据见
[DL_PILOT_R2.md](DL_PILOT_R2.md)。α20/ρ.5/H1、size50m/vision64/repeat1、
16env/ratio256、free nats1/dyn1/rep.1/原梯度开关，0/50k/100k各10episode。
基于PD_1原B/D相同前缀的探索对照，不建立机制或跨训练seed结论。

## 路径与监督

- 计划：`/data/Policy_Discrepancy/runs/dl-pilot-r2-plan-20261013-95ea434`
- 运行：`/data/Policy_Discrepancy/runs/dl-pilot-r2-20261013-95ea434/dmc_quadruped_walk/N-seed0-attempt01`
- 监督：计划目录下 `supervisor-attempt01/`，包含startup/probe/worker.log/resources.jsonl/终点记录。
- CPU预检：`/data/Policy_Discrepancy/runs/dl-pilot-r2-preflight-20261013-95ea434`。
- 本机小型计划镜像：`analysis/outputs/dl-pilot-r2/95ea434/`（ignored，不提交检查点/replay）。

实际命令：

```text
/data/Policy_Discrepancy/envs/dreamer/bin/python scripts/dl_pilot_train.py launch \
  --index /data/Policy_Discrepancy/runs/dl-pilot-r2-plan-20261013-95ea434/RUN_INDEX.json \
  --output /data/Policy_Discrepancy/runs/dl-pilot-r2-plan-20261013-95ea434/supervisor-attempt01 \
  --git-sha 95ea4348b8923d947cbb6e57a4d68dc5029c1fad
```

12:52:32.560207启动父PID668492，worker PID669298；独立owned group，完整入口14400秒。
12:52:11.015458启动前fresh XML确认4–7各0MiB/无compute或graphics进程。
零动作probe PID668500于12:52:33.854492–12:52:35.822339验证实际CUDA/EGL都仅GPU4后退出；
12:52:37.387072启动前再次fresh核查。训练worker12:54:10.446786已实际CUDA在GPU4，
当时采样显存2517MiB。12:56:45.560914实际训练和环境初始化后，1个compute/17个graphics
PID全部仅物理GPU4，显存8010MiB；父监督持续留证，零动作probe不代替实际训练落卡证据。
GPU4 UUID为 `GPU-7bf636ec-fa03-aa3f-7314-dd803c1658be`，逻辑CUDA0，
`CUDA_VISIBLE_DEVICES`设UUID、`CUDA_DEVICE_ORDER=PCI_BUS_ID`、`MUJOCO_EGL_DEVICE_ID=4`、
`MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`。

## 验收依据与限制

clean Linux新pilot5项、原kernel13项、原protocol14项共32项：30通过、2环境限定skip；
包括真实owned子进程/嵌套进程超时清理、实际worker double的动作/reset/评价隔离与异常停止。
各日志在CPU预检目录，pip check exit0。kernel预期非法signal callback报错不是失败。
CPU double不是真实任务验收；实际size50m旁路/梯度历史验收另见DL_VALIDATION_LOG。

原PD_1历史文件hash与0/50k/100k每点10完整episode通过。两条初始checkpoint同字节、
294叶params/optimizer/norm digest一致，counters全0。新N从头初始化已实际匹配共同digest，
manifest留证。0点10完整episode均1000动作，mean raw return55.848517901797926，
评价不改训练状态。13:03:23.121759收据至ID2762/动作13080，无失败，GPU4显存8010MiB。
XLA实际记录driver CUDA12.8比PTX12.9.86旧，关闭并行编译，
保持环境不变；包含这一耗时，不解释为方法效果。

完整100k、24493更新、30完整评价episode、完整evaluation state audit、
final_state/父audit/exit0及释放尚待完成。初始化及实际训练compute/EGL已有上方有限证据。
旧native初帧失败与双信号/保护机制证据不足仍保留。

## 终点与后续

由当前chat继续监测至本次100k完成并审计，普通工程故障本机修复后新attempt，不续跑或拼接。
终点联报100k梯形AUC与endpoint和PD_1 B/D前缀、有效释放统计及限制，
更新STATUS/EXPERIMENTS/DL_VALIDATION_LOG和本页；不增加任务/seed/预算或修改gate。
