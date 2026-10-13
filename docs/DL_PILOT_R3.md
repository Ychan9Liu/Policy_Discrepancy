# DtLatch 1M 探索实验：DL-pilot-r3

## 用户更正与冻结范围（2026-10-13）

用户将预算明确更正为“跑1m”，覆盖此前100k预算；sv2物理GPU4–7分配继续有效。
沿用单任务 `dmc_quadruped_walk` / N / seed0，实际只用GPU4，5–7不新增作业。
100k的r2已按用户更正中止，完整保留为历史；r3从头开始，不恢复或拼接其状态。
这是完整1M预算的探索性在线学习实验，不宣称旧机制预验证通过，不改正式M2-v1。

**已确认决定**：用户明确启动、将预算改为1000000实际训练动作，并指定sv2/GPU4–7。
**本次沿用初值与设计**：α20、ρ.5、H1 reward-support、单任务seed0、PD_1历史B/D对照；
来自此前助手推荐的本次采用，不写成已获效果证据或用户逐项确认。
**工作假设**：选择性保护可能改善学习。
**仍无证据的机制解释**：识别关键观测、后验信息免受压缩、跨训练seed性能改善。
同预算一条seed0阳性不能区分通用正则强度、轨迹/渲染随机性或代码版本影响。

## 固定配置与统计口径

| 项目 | r3设置 |
| --- | --- |
| 任务/组/训练seed | quadruped_walk / N（DtLatch）/ 0 |
| 方法 | DL-code-r1已有实现，`dt_latch.mode=dtlatch`、`rep_probe.mode=off` |
| 信号与参数 | α20、ρ0.5、H1；TwoHot真实下一步reward支持差；raw KL不作为新gate |
| 权重 | `v=sg[1-ρ max(exp(-d_q)-exp(-d_p),0) (1-1/(1+α D))]`，原有效邻接mask |
| 模型/观测 | size50m、64×64 RGB、proprio=False、camera=-1→实际quadruped2 |
| 更新 | batch16×64/context1、16env、train_ratio256、consec_train1 |
| 损失/梯度 | free nats1、dyn1、rep.1、reward_grad/repval_grad=True、ac_grads=False |
| 预算 | 1000000实际训练动作，预填充计入，reset/eval排除，repeat1 |
| 评价 | 0及每50k至1M，共21点×10完整1000步episode、原生sampled actor/独立eval RNG |
| 主汇总 | 实际动作网格梯形积分/1000000；80/85/90/95/100万五点等权tail均值；终点联报 |
| 旁路/匹配 | 不新增S/P/W，不使用c/κ匹配或shuffle；κ/bins未绑定 |
| 运行 | BF16、prealloc=False、一张A100-SXM4-80GB、JSONL和完整运行证据 |
| 工程硬上限 | 单attempt整个入口57600秒（16小时），含初始化/编译/评价/审计，不自动延长 |
| 恢复 | 不续跑、不拼接、不覆盖旧产物；必要修复本机测试/提交后新attempt |

16小时是监督上限，不是ETA或GPU必须占满的时间。r2实测正常训练约24–25动作/秒；
外推1M纯训练约11–12小时，编译、21点评价、checkpoint与审计另计，最终按实测记录。
不会因中途回报调α/ρ、增任务/seed/预算或改变停止门。

## 历史对照、版本与种子

正式PD_1训练SHA `48d71f1541721208322442f5a1d5977124f3e888`，复用下列完整B/D：

- B：`/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-seed0-20261009/dmc_quadruped_walk/logging-attempt01`
- D：`/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-dt-seed0-20261009/dmc_quadruped_walk/dt-attempt01`

绑定实际manifest/evaluations/final_state SHA256，保存全部21点×10完整episode小型证据。
新配置逐字段对照实际B；预算与评价现在相同，仅允许新入口/日志/路径/方法开关差异。
原 `train_env` / `train_replay` / `eval_env` 命名空间及PCG64 policy/batch/eval规则不变。
replay seed3974293035；评价root0/task/`eval_env`/grid_index/episode0–9。
更新数应为50k11993、100k24493、1M249493；中间非0网格更新数为point/4−507。
每点评价前后params/optimizer/norm、训练计数、replay sampler RNG/metrics必须相同。

两条旧0步checkpoint同文件SHA2566ad6dfae…，完整294叶初始化params digest为
`18199ec5fb45b13f18ec29ae6dcc438d3018807888ce9f2c5e6ee64b621a9267`。
r3重新从root0初始化并核对；不加载r2或旧已训练checkpoint。
Python3.11.16/JAX及JAXLIB0.4.33/NumPy1.26.4/dm-control1.0.48/MuJoCo3.15.0；
lock SHA256 `cd50c7936690f9adbc5c35f99b3df2f7a977cef3222582f988979b11f65b6f22`。

相同seed/初始参数不保证渲染或轨迹逐字节相同，旧B/D及r2的0点评价已有轻微差异。
本次采用通常在线训练/评价对象，不放宽旧native pixel identity门，不称机制诊断通过。
单训练seed/历史控制仅为描述性对比；10eval episode不等于10训练seed。
PD_1原constant c不等于DL匹配强度κ；尽管新预算覆盖[100k,300k)，本次没有批准新匹配臂。

## 执行与验收

入口仍为 `scripts/dl_pilot_train.py`，新Git对象保存r3；r2真实95ea434 checkout不改。
只同步承担新任务的sv2隔离clean checkout；不在服务器修改源码，不移动canonical48。
GPU4 UUID `GPU-7bf636ec-fa03-aa3f-7314-dd803c1658be`，UUID可见/CUDA0、
`MUJOCO_EGL_DEVICE_ID=4`、`CUDA_DEVICE_ORDER=PCI_BUS_ID`、`MUJOCO_GL=egl`、
`PYOPENGL_PLATFORM=egl`；每次重新fresh XML核4–7、零动作CUDA/EGL探针及实际训练子PID映射。
原DL/M2 runner准入、算法、loss/梯度/随机规则不改；只预算、网格完整性、汇总与上限改为r3。

完成条件：clean Linux确定SHA/CPU契约、实际共同初始化、1000000动作/249493更新、
21点×10完整episode/210000eval动作、逐更新连续且loss/grad有限、21点训练状态不变、
实际compute/EGL同卡、worker exit0、父audit/终点与释放记录完整。
CPU fixture是工程证据，不冒充真实任务或科学效果。运行/命令/实际SHA/产物状态更新至
`docs/STATUS.md`、`docs/EXPERIMENTS.md`、`docs/DL_VALIDATION_LOG.md`及r3结果页。

## 仍待决定

先完成本次授权的1M及审计，再由05对比完整同预算B/D与gate有效释放统计。
增加训练seed/任务、匹配强度/条件shuffle、独立留出、合法nuisance/错误依赖和
保护因果机制实验均尚未批准；不以单条性能阳性替代它们。
