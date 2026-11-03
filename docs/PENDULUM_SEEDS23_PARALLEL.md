# Pendulum seed2/3：PPD-KL改为GPU5/6并行

2026-11-04，用户明确补充“刚分配的其他的空闲gpu也可以用，可以并行的测试可以尝试”。本chat据此在原sv3 GPU1/4/5/6/7资源范围内，将尚未启动的两条PPD-KL安排到物理GPU5/6，与既有两baseline并行；GPU7保留。不增加任务、seed、组或科学参数，不重复seed1/hopper。旧同卡串行安排仅对尚未启动的Dt被本记录替代；[原启动报告](PENDULUM_SEEDS23_RUNS.md)和证据保留。

## 资源、配置与版本边界

|sv3物理GPU|任务/seed/组|本次动作|
|---:|---|---|
|1|pendulum_swingup seed2 baseline|保留PID3690581及原训练目录|
|4|pendulum_swingup seed3 baseline|保留PID3690588及原训练目录|
|5|pendulum_swingup seed2 PPD-KL / dt|通过新卡验收后启动|
|6|pendulum_swingup seed3 PPD-KL / dt|通过新卡验收后启动|
|7|—|保留，不安排额外实验|

00:59:31资源快照：GPU1/4各17个本项目C/G进程，5/6/7无进程；/data余336555589632字节。GPU5 UUID `GPU-79704918-e096-92b4-252c-e3244497a8b3`；GPU6 UUID `GPU-70db6c66-9917-e0e2-44db-0fc80fcf41b9`。实际CUDA/EGL映射尚待新卡探针；启动前重新核查，不把快照当预约。

全部四条agent继续执行同一个已验收的训练SHA `c5f86ca1a16648f57db3c3abfa2584ac560495ca`，路径`/data/Policy_Discrepancy/checkouts/pendulum-seeds23-c5f86ca1a166`，保持干净且不移动。新提交只增加控制器/测试及文档，部署到另一个独立checkout；控制器SHA和训练SHA分别记录。新控制器启动子进程时cwd明确指向原训练checkout，不把控制器版本冒充训练版本。

原配置不变：clean、64×64 RGB视觉、size50m、repeat1、batch16×64/context1/train_ratio256；seed2/3；baseline=logging实际rep权重1，PPD-KL=dt/alpha20，free nats1，dyn/rep=1/.1，ac_grads=false、reward_grad=true、repval_grad=true、repval_loss=true；原optimizer/RNG/奖励不变。每条百万训练动作含prefill排除reset/评价，0..100万每5万固定快照/10完整episode/原采样动作与独立评价随机流，CUDA/bfloat16/prealloc=False/JSONL+online PD_1。

PPD-KL的权重来自自身批次Dt，不读取baseline c，提前并行不改变训练方法。各baseline仍在百万动作终点按自身`[100000,300000)`成功更新、raw rep KL>1位置S/N完整冻结c；终点审计和四条完整指标仍依赖baseline全部产物。没有把工程c、部分窗口或旧seed c当正式c。统计继续严格`pendulum-seeds23`，未平滑原始return AUC/百万及80/85/90/95/100万五点均值，不自动混算seed0/1。

## 保留baseline并阻止重复启动

原监督器PID3690542保留；不发停止信号、不重启baseline。它继续持有两个baseline子进程，可真实记录退出码并完成原始收据/评价/快照/c审计。

新控制器只接管两个尚未启动的Dt槽位：先核原actual-RUN_INDEX身份及queued状态，再在原未使用的`seed2/dt-attempt01`和`seed3/dt-attempt01`目录建立唯一`SUPERSEDED_BY_PARALLEL.json`。marker记新run ID、目录、控制器/训练SHA及新合并索引，目录不含训练产物；若已经存在任何正式数据或不匹配marker，拒绝覆盖/启动。原监督器后续因目录已存在而拒绝该旧槽位，不能再额外启动Dt；此拒绝是已明确替代的排队项，不记成失败训练，也不纳入正式比较。原baseline目录与内容不修改。

新两条Dt使用新独立根`/data/Policy_Discrepancy/runs/formal-m2-pendulum-seeds23-parallel-dt-20261104-sv3/seed{2,3}/dt-attempt01`及新W&B ID；W&B仍PD_1，run group沿用原四条campaign。新根`actual-RUN_INDEX.json`作为四条合并索引，只包含原两baseline和新两Dt，保留各自真实路径/ID/SHA，不包含两个旧替代槽位。

新监督器读取原监督器的baseline验收结果，并自行记录新Dt真实退出码和逐条审计。四条均审计通过后才运行原始return完整统计及W&B精确JSON/hash收尾。旧根supervisor-final可能因两个替代槽位blocked而complete=false，不能据此否定已通过的baseline，也不能用新Dt单独完成冒充四条完整；四条最终验收以新合并索引和final-audit-attempt01/completion.json为准。

## 本次验收与输出

控制器`run_pendulum_parallel.py`及9项CPU测试验证：精确seed2/3、仅Dt、只用5/6；原训练SHA和科学配置保持；两个新卡绑定；已存在任务不可覆盖；marker阻止原launcher执行Popen；合并索引保留原baseline与唯一新Dt；容量不足和验收缺项不得启动。首次本机测试因导入TestCase导致重复发现旧冻结测试、遇最小venv缺portal；修正测试发现方式，9项新测试全部通过。旧冻结测试不跳过，服务器完整回归仍会执行。

原训练SHA已通过的129 CPU/三变体完整agent/在线logger证据复用；新控制器在服务器完成相关回归，并在GPU5/6各执行实际训练SHA的size50m CUDA工程短程（4098动作、517成功更新、3×1评价、2报告/评价隔离审计），绑定严格要求两个seed/卡及完整收据。新卡工程通过后才给原旧槽位写marker、启动两条正式Dt；工程数据不用于正式指标。重新使用保守240GB总剩余容量门槛，不删除原checkpoint/replay。

本机记录：`analysis/outputs/pendulum-seeds23-parallel-20261104/`；服务器工程根`/data/Policy_Discrepancy/runs/engineering-pendulum-seeds23-parallel-20261104-sv3/`。完整配置、命令、SHA、probe、CPU/CUDA验收、guard、新旧索引和W&B读回随后记录。当前文档初稿时两条新Dt尚未启动；不把准备计划写成运行完成。

首选继续本chat完成新卡验收/并行启动，然后监测并完成四条终点审计和05分析。新科学干预仍交02/用户，不自动建chat或发送消息。交付完整SHA/实测控制器SHA/实际训练SHA、差异与同步范围、实际资源状态、原始指标/限制及可复制prompt；本机其他chat草稿保留。
