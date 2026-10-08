# 03 第二阶段：Dt、constant、global shuffle 工程验收

日期：2026-10-08。方法及验收依据：`docs/METHOD_SPEC.md` v0.1。未改动源码基准 `e935ff7bb01085fec8a1672e31ddfe0536c0f9c0`；第一阶段验收代码 `a8ca727dbd3f3698bb217c00d15997681ee644ef`。本阶段算法提交 `9e0fb8e03f9b3d155bae663c5ec962560db9dafc`；初始测试提交 `64459ae`，修正/扩展后的最终测试提交 `fac4b704d79a2ca4aea7837ec8fe003e5ae9c04b`。中间测试提交见 Git 历史。所有数值均为极小 debug fixture，不是四个 DMC 任务的科研比较。

## 实现语义

- 配置字段 `agent.rep_probe.mode` 接受 `off`（默认）、`logging`、`dt`、`constant`、`shuffle`。默认 `alpha=-1.0`、`c=-1.0` 是“未提供”占位，启用时须显式配置。Dt、logging、shuffle 的 `alpha` 须有限且非负；constant 的 `c` 须有限且 `0<c<=1`。constant 的 c 从运行配置读取并固定，不在 batch 或 step 内校准；正式训练前的校准规则仍由实验协议确定。`dt alpha=0`、`constant c=1` 静态旁路，原损失及随机流逐位回退。`shuffle alpha=0` 的全 1 权重也完成回退验收，但保留只读 probe 与独立置换。
- 仅对已经由原 RSSM 计算且完成 free-nats 的 `losses['rep']` 逐位置乘实际权重，然后沿用原 `mean([B,T])` 和原 `rep` scale。dyn KL、其他 world-model/actor/value/reconstruction/reward/continuation/repval 目标、原采样 latent 及 optimizer 不改。probe 在正常 actor/value 和 repval 前向之后执行，使用已得 prior logits、同一 actor、unimix 后的逐因子 mode 与显式 stop-gradient；Dt 与源权重沿用第一阶段 float32 解析对称 Gaussian KL。
- 实际权重 `v` 为：logging=1，Dt=`w=1/(1+alpha*D)`，constant=`c`，shuffle=`reshape(take(flatten(w), permutation))`。全局置换索引覆盖本次更新**完整有效 B×T**，含跨设备 shard 的位置和普通置换的不动点；无额外 episode/terminal mask。key 为独立 `jax.random.PRNGKey(train_seed)`，依次 `fold_in(0x4454)` 和 `fold_in(optimizer_step)`，不消耗 Ninjax 的 seed。相同 seed/step 可重放，下一 step 换 key。它保留本组本次更新的权重多重集合，不借用另一组训练轨迹。
- 开启模式仅支持标准 categorical RSSM、连续 action heads 和 `bounded_normal`；不支持组合或非法参数在初始化前报错。非有限/非正标准差及非有限 Dt 产生可见 NaN、`invalid_D_frac`，不静默转成合法 gate。解析 KL 的平方差形式没有负 KL 裁剪；`raw_active_D_near_zero_frac` 的 `1e-6` 仅为诊断阈值。
- `train/dt/` 诊断记录源 Dt/w 与实际权重的均值/分位数、raw rep KL、free-nats 前后 rep、活跃比例、活跃区源/实际权重、加权超额 KL 及减少量、源/实际权重与 raw KL 的协方差、shuffle 换位比例、mode/actor 参数差异和非法 Dt。空活跃区以 `active_weight_available=0`、`active_weight_mean=NaN` 标示。对 logging，候选 w 与实际权重 1 分开记录。

## 环境、fixture 与容差

服务器 sv1；本地仓库修改后通过 Git bundle 同步到 `/data/Policy_Discrepancy/repo`，运行产物只在 `/data/Policy_Discrepancy/runs/verify-overlay-phase2-64459ae/`。Python 3.11.16、JAX 0.4.33、A100-SXM4-80GB、驱动 570.158.01，依赖锁文件 SHA-256 `cd50c7936690f9adbc5c35f99b3df2f7a977cef3222582f988979b11f65b6f22`。比较使用 CPU/float32 与 CUDA/bfloat16。连续二维动作、B=2、world-model T=3、replay context=1、`free_nats=0.1`、`imag_last=2`、seed=7、`alpha=50000`、`c=0.6` 是**工程 fixture**；大 alpha 仅让极小 debug actor 的微小 Dt 在测试中产生可测作用，不是正式选值。

实际运行提交：主 CPU/CUDA 八模式快照及比较为 `b8918a7`；shuffle 全 1、replay context、内部 reset 与不活跃 fixture 为 `cadfc59`；连续两块与两卡完整 agent 为 `830fd79`；100 更新开销样本及最后的 9 项 unittest 为 `fac4b70`。各提交均以本机 Git bundle 快进同步到 sv1，运行过程中没有在服务器直接编辑项目代码。最终提交在文档完成后另行快进同步，便于下一次工程试跑从确定 SHA 启动。

GitHub 同步记录：本机 `git push origin main` 在提交 `aa6eb64` 后因 HTTPS connection reset 失败，`origin/main` 仍为 `e935ff7`。本机 `phase2-final.bundle` 包含 `e935ff7..aa6eb64`，SHA-256 为 `fdfda041ca8be384c38bbb5207def8013e8676fb3c372cf6b9f391570b0407e0`；sv1 上同一 bundle 哈希一致、`git bundle verify` 通过，并快进到 `aa6eb64`、工作区干净。推送失败不影响已确定提交的服务器验收；文档后续提交也须以 bundle 快进同步并在交接报告中给出最终 SHA。

未改动基准从 `e935ff7` 单独导出到第一阶段的 `baseline-src`；每模式在全新进程、同输入/carry/seed 下运行。off、logging、Dt alpha=0、constant c=1、shuffle 全 1 对所有共享数组逐位相同（容差 0），包括参数初值、原始梯度、训练状态及一次 optimizer 更新。有效模式要求原采样 rep 特征、初始化、训练状态、dyn/其他独立梯度及非 rep loss 逐位相同；总 loss 和原始梯度按分支重组比较。CPU/float32 数值比较 `atol=2e-6, rtol=2e-5`；CUDA/bfloat16 的总梯度重组 `atol=0.002, rtol=0.04`，只容纳低精度加法/反传舍入，未改动分支仍要求逐位相同。解析 KL float32 fixture 使用 `rtol=1e-6`；相同分布 Dt 与隔离梯度要求精确零。不同后端之间不要求相同。

## 验收结果与命令

在 `/data/Policy_Discrepancy/repo`、使用 `/data/Policy_Discrepancy/envs/dreamer/bin/python` 执行：

```sh
CUDA_VISIBLE_DEVICES=0,1 python -m unittest tests/test_rep_probe.py tests/test_rep_overlay.py -v
RUNS=/data/Policy_Discrepancy/runs/verify-overlay-phase2-64459ae
python tests/run_phase2_snapshots.py --output_dir /data/Policy_Discrepancy/runs/verify-overlay-phase2-64459ae --baseline_src /data/Policy_Discrepancy/runs/verify-logging-phase1-30bb5a6/baseline-src --platform cpu --dtype float32
CUDA_VISIBLE_DEVICES=0 python tests/run_phase2_snapshots.py --output_dir /data/Policy_Discrepancy/runs/verify-overlay-phase2-64459ae --baseline_src /data/Policy_Discrepancy/runs/verify-logging-phase1-30bb5a6/baseline-src --platform cuda --dtype bfloat16
python tests/compare_overlay.py --baseline $RUNS/baseline-cpu.npz --identity off=$RUNS/off-cpu.npz --identity logging=$RUNS/logging-cpu.npz --identity dt0=$RUNS/dt0-cpu.npz --identity c1=$RUNS/c1-cpu.npz --active dt=$RUNS/dt-cpu.npz --active constant=$RUNS/constant-cpu.npz --active shuffle=$RUNS/shuffle-cpu.npz
python tests/compare_overlay.py --baseline $RUNS/baseline-cuda.npz --identity off=$RUNS/off-cuda.npz --identity logging=$RUNS/logging-cuda.npz --identity dt0=$RUNS/dt0-cuda.npz --identity c1=$RUNS/c1-cuda.npz --active dt=$RUNS/dt-cuda.npz --active constant=$RUNS/constant-cuda.npz --active shuffle=$RUNS/shuffle-cuda.npz --atol 0.002 --rtol 0.04
```

后两条比较命令在实际执行时使用了完整绝对路径。9 项 unittest 通过，包含 2 卡 `NamedSharding` 跨 shard 置换、多重集合保持、固定点/实际换位比例、同 seed/step 重放与不同 step 变化、配置校验、free-nats 顺序和局部梯度。第一阶段五项测试保留且通过。

CPU 与 CUDA 各模式的 off、logging、Dt alpha=0、constant c=1、shuffle alpha=0 分别对未改动基准有 **789 个共享数组逐位相同**。三种有效模式各有 **490 个未改动数组逐位相同**，且 `rep_active_frac=1`、活跃位置实际权重小于 1、超额 KL 减少、rep loss 与原始梯度确实变化。CPU 的基准总 loss 为 8.60686588；Dt、constant、shuffle 分别为 8.59959602、8.59713554、8.59956837。对应活跃权重均值 0.70070767、0.60000000、0.70070761；加权超额 KL 减少量 0.04276655、0.05730557、0.04304455。shuffle 6 个位置实际换位 5 个，权重多重集合与 Dt 相同，但位置对应不同，`cov(v,Rrep)` 从源权重的 +0.00011125 变为 -0.00016675。这些数值证明 fixture 中 gate 有效作用，不代表效果收益。

CUDA 的基准总 loss 为 8.60598278；Dt、constant、shuffle 分别为 8.59873867、8.59630013、8.59870911。三组的 `rep_active_frac=1`，活跃权重均值约 0.700287、0.600000、0.700287，shuffle 换位比例 5/6。bfloat16 分支重组最大绝对残差为 0.01487、0.01893、0.01533；最大总梯度变化为 0.01503、0.01799、0.01547。大残差集中在绝对值约 2.4 的 bias 梯度，符合 bfloat16 量化尺度；严格的 float32 容差在 CUDA 上明确失败，随后用上述已记录的后端容差通过。CPU 的相应最大残差约 `1e-6`。

补充序列 fixture：`replay_context=0, free_nats=0, imag_last=3` 的 continuation、`replay_context=1` 的内部 reset/terminal，各自对基准核对 Dt/shuffle 或 Dt 的分支与梯度；活跃比例分别为 1、2/3。连续两个 chunk 的 fixture 在第二块使用前块 carry、无 `is_first`，baseline 与 shuffle 都完成更新且参数有限。另以 `free_nats=1000` 构造全不活跃区，Dt 的标量 loss 改变，但 `repgrad/`、`grad/` 及 `updated/` 的所有数组与基准逐位相同；`rep_active_frac=0`、活跃均值 NaN、`excess_reduction_mean=0`，说明这种情况下 gate 未改变训练梯度。

两块 A100 上完整 shuffle agent 使用 `train_devices=[0,1]`，完成初始化、loss、原始梯度、optimizer 更新，所有必要值有限；完整训练路径的 `shuffle_moved_frac=5/6`。单元测试另外核对了跨 shard 的源位置与目的位置，不是每卡局部打乱。

预热后同一极小 CPU fixture 的 100 次更新耗时：off 0.3810 秒、logging 0.4189 秒、Dt 0.3987 秒、constant 0.4135 秒、shuffle 0.4161 秒，相对 off 分别增加约 10.0%、4.6%、8.5%、9.2%。进程峰值 RSS 依次为 704244、731660、731020、723440、740660 KiB。命令为上述 runner 加 `--modes off,logging,dt,constant,shuffle --bench_updates 100`，逐模式元数据见运行目录下 `bench-cpu/*.npz.json`，完整输出见 `bench-cpu/runner.log`。这些是小模型单次测量，不用于预测正式模型吞吐；进程 RSS 包含编译和快照开销，不等于模型显存。

## 剩余风险与真实任务交接

- 本阶段只有极小 debug 连续动作 fixture、单步或短序列工程验收，没有在四个真实 DMC 任务上训练 Dt/constant/shuffle，没有 M2 效果结论。此处的 Dt、w、活跃比例和换位比例也不能外推到正式模型。
- bfloat16 的总梯度分解须用低精度容差；CPU/float32 给出更强的局部数值检验。全局置换可能触发跨卡通信，实际模型规模下的吞吐和显存需要另测。诊断不替代模式漏检、干扰、延迟线索的任务级设计。
- 进入真实任务**工程集成试跑**可使用已验收提交、四个已确认的 clean 视觉 DMC 任务和独立 `/data/Policy_Discrepancy/runs` 目录，运行前记录 Git SHA、完整实际配置、环境版本、seed、服务器/GPU，并检查 `train/dt/` 活跃作用字段。试跑结果仅用于检查接口、吞吐、显存、日志与可复现性。
- 正式多 seed 比较仍须先冻结模型规模、baseline、校准窗口/seed/统计量及 c 匹配规则、评价指标、训练/评估 seed、预算、停止/续跑和 alpha/c 及效果判据。条件 shuffle 须等更强 M2 协议给出分层语义再实现。
