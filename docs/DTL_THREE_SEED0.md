# DtL 三任务 clean / seed0 运行

2026-11-05，用户明确启动 quadruped_run、quadruped_walk、finger_spin 各一条 DtL，
指定 sv2 物理 GPU5–7，并确认三条均为 clean。不是 DCS 背景训练，也不新增 baseline。
继承当前已经实际在线运行的 DL-pilot-r3 配置；本轮扩展任务，不把旧预验证门补写为通过。

## 方法与配置

当前实现来源是分支 codex/dl-prevalidation 的完整版本
`0b5d5a6c88d710dfaafd521dda177310a518d05b`，过去 quadruped_walk 1M 在线实测为
`13ec72a214b0c03b886b2c7fac52c8f089966dec`。旧分支 docs/DL_PILOT_R3.md 给出原在线口径。
本次从 main `479c25b71d2383208e9f7c13f1de7942b6d6755c` 集成现有 DtLatch 模块、
独立 dtl_agent/训练包装器、TwoHot.target 和只读 logging 支持，保留 main 已修复的 mode 诊断、report、恢复审计和 DCS 准备。
原 agent.py、RSSM、rep_probe.py、JAX agent.py、protocol_v1.py 和 dmc.py 保持字节不变。
DtL 入口使用独立模块，已有基准源码一致性检查继续执行，不删除或放宽原检查。
不整支覆盖 main，不修改两个既有工作区的草稿或历史。

`agent.dt_latch.mode=dtlatch`，`rep_probe.mode=off`，alpha=20、rho=0.5、H1 reward-support，
normal_only=False、kl_tolerance=1e-5、kappa=-1、bins=[-1]。没有 c 校准、强度匹配或 shuffle 臂。
`e=max(exp(-KL(y||Pq))-exp(-KL(y||Pp)),0)`，实际权重
`v=stop_gradient[1-rho*e*(1-1/(1+alpha*D))]`，其中不可用邻接位置为 v=1。
next 存在、current 非 is_last、next 非 is_first；next terminal reward 可以使用。
只作用于 free-nats 之后的原 rep KL，原 reduction、dyn、奖励/value 路径与梯度开关保持。
该额外信号没有 probe RNG；训练/回放/评价仍沿用原命名空间；原 logging 模式采用单独只读 JIT。

| 字段 | 本轮声明 |
| --- | --- |
| 分配 | sv2 GPU5 quadruped_run；GPU6 quadruped_walk；GPU7 finger_spin |
| 组/seed/初态 | DtL，seed0，三条从头训练；quad_walk 是新尝试，不续接旧 r3 |
| 观测/模型 | clean，64×64 RGB，proprio=False，size50m，CUDA/bfloat16 |
| 训练 | batch16×64/context1，16环境，train_ratio256，repeat1 |
| 损失/梯度 | free_nats1，dyn/rep=1/0.1；ac_grads=False；reward_grad/repval_grad/repval_loss=True |
| 预算 | 1000000实际动作，含预填充，排除 reset 和评价；249493更新 |
| 评价 | 0到1M每50k，共21点，每点10完整episode，固定快照、sampled actor、独立评价 RNG |
| 指标 | 原始完整episode return，实际动作步梯形积分/1M；80/85/90/95/100万五点均值 |
| 日志 | JSONL＋既有 W&B entity ychan9liu-tsinghua-university / PD_1；单独campaign/group/run ID |
| 监督 | 每attempt最多57600秒，失败保留，不自动覆盖/恢复；三卡启动容量门槛180GB |

17:42只读容量为182509658112字节，GPU0–4五条原项目作业继续写入。
用户随后明确“先跑吧，我会清理”，允许本轮低于180GB初始容量建议继续启动；
使用显式 `--accept-initial-capacity-deficit` 并记录实际余量，不扩展到其它批次。
监督器持续记录容量；低于5GB时仅终止本轮所属子进程、保留失败证据，不删除任何既有产物。

## 入口与验收

新增 `dtl_three` preset、封闭的 `protocol_dtl_three` scope 和 `scripts/run_dtl_three.py`。
原 M2 四组及扩展入口明确拒绝启用 DtLatch，避免以 baseline/Dt 标签运行新算法。
共用原动作计数、评价快照和恢复拒绝逻辑；DtL 每点评价均校验前后训练状态与随机流指纹。
逐成功更新另存 `dtl_updates.jsonl`：实际权重、free-nats 活跃比例、有效活跃比例、
保护位置、超额释放、前后rep均值、loss/grad_norm，按 update ID 与 checkpoint 严格对应。
正常 replay 重采样不去重；记录缺失、额外更新和重复ID拒绝重开。

准备、设备探针、工程、训练四个入口分别为 `prepare / probe / integration / run`。
训练必须绑定同SHA三任务真实 CUDA 工程 acceptance 和各卡 CUDA/EGL 实际同卡证据。
工程使用 size50m 原参数，2048动作、两点评价各1完整episode、5更新和1次真实report；
它验证接口、报告只读与评价隔离，不作为百万步结果或方法有效证据。
原 DtL CPU 梯度/非平凡权重/旁路/全agent/normal_only 测试与原 M2 算法、协议测试保留。
本轮证据根为 `analysis/outputs/dtl-three-20261105/`，大产物不提交。

实际提交、同步和启动/运行状态由后续带时记录补充；准备文件不等于任务已启动。
当前研究仍是单训练seed探索；三个任务不构成冻结20任务整体结果，episode不是独立训练重复。
不据已有r3或本轮工程通过声称保护机制或方法有效；新增科学复验仍由02与用户确认。
