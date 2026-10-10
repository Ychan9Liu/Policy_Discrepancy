# DL 验证日志与证据索引

版本 DL-engineering-r1 / DL-protocol-r1，2026-10-10。整体目标**未完成**：本地工程验证进行中；真实模型、独立辨别、活跃压缩、替代解释与正式候选审计仍需实际 GPU/独立数据证据。

## 版本和同步边界

项目根 `D:\program\PD\dmr3`；当前隔离工作树 `C:\Users\97370\.codex\worktrees\1f9a\dmr3`，分支 `codex/dl-prevalidation`。DL-r0 归档提交 `2b9641d`；合并基准完整 SHA `ae8372e42a815824d2e4ec3a527bfb985a81972b`，含 main `595e982f1a3939775879728009e2105ee516e86d`。下列初步证据在合并基准**加未提交修改**取得，各文件哈希存小型证据，不把 HEAD 当作全部实测代码。

最终 clean 提交后须记录真实测试 SHA及结果；文档后续提交与测试代码的差异单独列出。当前没有向服务器同步 DL，没有启动 GPU；不更新原运行中 checkout，也不覆盖 canonical main 的既有本地修改。main/GitHub 状态须由实际 Git 查询确认。

## 环境与本地证据

隔离环境为 `%LOCALAPPDATA%/Codex/dl-validation-env`，Windows/Python3.12.14、JAX/jaxlib0.4.33、NumPy1.26.4、elements3.22.2、ninjax3.6.3、optax0.2.5、chex0.1.90、portal3.8.1、dm-control1.0.48、MuJoCo3.15.0；关键版本对齐服务器 lock，`pip check`通过。Windows Python3.12 与预期 Linux Python3.11 仍有差异，不能替代服务器验收。未更改系统/服务器环境；进程 DLL 路径补充只存在该外部隔离环境。

| 证据 | 已观察到的结果 | 限制 |
| --- | --- | --- |
| DL 最小探针、TwoHot、梯度、mask/terminal、条件置换 | 多轮 CPU fixture 已通过；包含两 CPU 逻辑设备上的跨 shard 对照 | fixture 不证明真实任务有效 |
| 完整 tiny Agent 旁路 | off/alpha0/rho0/logging/strengthκ0 的 617 项参数/梯度/loss/latent 和两次优化更新精确一致；非法有效 reward 明确失败 | 实测小模型，真实 size50m 待验；最终提交须重验 |
| 诊断工具 | 12 项 CPU 测试通过，含真实 TwoHot、11,573 参数 tiny Agent 的共享 h、冻结参数与 clean history 随采样位置不变 | 非真实 size50m checkpoint，不含真实 segmentation |
| quadruped 物理复原 | 3 seeds×4位置×7候选，共84项 H10重复检查，完整积分状态/任务/RNG/counter一致，最大误差0；float32 H1标签一致 | CPU、禁止渲染；只验物理分支与编码，不验 gate/return收益 |
| DL 短训练计划和契约 | 初步9项 fixture通过；prepare生成18条等待计划，不构造 agent或启动训练 | draft为dirty SHA，会被run拒绝，须真实 StageC 后在clean SHA重建 |

初步诊断证据根（Git ignored）：`analysis/outputs/dl-prevalidation/diagnostic-cpu-r1/`，含 `tests-frozen.log`、`physics-final.json`、`cpu-validation-frozen.json`。最后一项保存诊断/测试/物理脚本 SHA256、日志 hash和未提交标识；失败/旧轮结果未覆盖。完整 Agent / 最终提交验收的日志与数量以下次真实生成的索引为准。

### DL 已发现的工程失败和修复

- 静态 gate=1 仍执行乘法会产生后续 optimizer 末位差；改为真正的 alpha0/rho0/strengthκ0 旁路，没有放宽精确比较。
- JAX ordered callback 在 Dreamer transfer guard 下创建 host token；改用 unordered invalid-signal callback，实际非法训练 fixture 仍明确失败。
- 物理 float64 reward 与采集 float32 标签误作同精度比较；分开物理重复误差与 float32 编码一致性。
- 按诊断 batch4 的 clean carry推进会使历史随机流受选点影响；每帧统一 batch1 advance，诊断 carry丢弃并实测 invariance。
- P 索引移动可能只是互换相等权重；新增实际权重变化和释放量检查，未改变门槛求通过。
- 本轮 Windows regression 发现旧 `freeze_c.py` 的 JSON LF/CRLF 转换导致旁边 checksum 不等。已在 main `595e982` 的独立源码归档复现同一失败；DL-engineering-r1 改为写入实际被 hash 的 UTF-8 bytes。没有改变 c 公式、统计、容差或 Linux 字节；修复后协议14项/4subtests通过，完整回归仍须确定提交重验。

## GPU 和真实方法验收

sv3 原 SSH 入口 `js3.blockelite.cn:18600` 初期连续只读访问在认证前关闭，失败证据 `analysis/outputs/dl-prevalidation/bootstrap/sv3-ssh-20261010-145416.txt`。15:30北京时间恢复；15:31:14快照实际 hostname=lyg0326、GPU4–7均A100-SXM4-80GB/81920MiB、0MiB/无计算图形进程；0–2其他环境进程为未知作业，不干扰。原 checkout干净 d9be5574878d2c3f117cc6667be844f9a527eb84；Python3.11.16、pip check通过，/data约1.1TiB可用。尚未启动本轮 GPU 进程或训练，真实 checkpoint及CUDA/EGL映射待核查。空闲仅是该时点快照，不是持续占用保证；每次启动带时间重查。

最低一张独占 A100 可串行推进；分配的四张用于独立任务并行，无需多卡模型。优先单卡小型 pilot 测资源，再将两个 checkpoint 的采集/score分别排程，工程验收使用另卡，条件短训练最多四条并行。历史 quadruped_walk采样峰值7.89GiB、预热24.72–24.88动作/s不含本轮新探针。六组×3seed×100k=180万动作，旧吞吐外推约20.4 GPU小时纯训练；四卡约5–6小时纯训练排程，诊断/编译/评价另计。四卡预留一天仅是安排建议，不是实测 ETA或资源占用承诺。

### DL 真实设备绑定初验

15:36:53–56北京时间在 sv3物理GPU4 / UUID `GPU-9c070d5a-ac68-4c26-9522-a790ad874b23` 执行既有 `scripts/probe_formal_devices.py`。代码从完整 SHA `ae8372e42a815824d2e4ec3a527bfb985a81972b` 导出，helper SHA256 `3954d6051287141c7bf27b6ce68da8a813f1b798357c216f44a1680bbaf54194`；训练动作/Agent更新为0。PID1248755 的 CUDA计算UUID和 `nvidia-smi pmon` 的 C+G均只在物理GPU4，exit0；probe进程已结束，未停止其他进程。这仅验计算/渲染设备绑定，不是模型训练或方法证据。

证据：服务器 `/data/Policy_Discrepancy/runs/dl-prevalidation-bootstrap-20261010-ae8372e/gpu4-device-probe.json`，本机 `analysis/outputs/dl-prevalidation/bootstrap/gpu4-device-probe.json`。环境为单UUID `CUDA_VISIBLE_DEVICES`、`CUDA_DEVICE_ORDER=PCI_BUS_ID`、`MUJOCO_EGL_DEVICE_ID=4`、`MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`、进程CUDA0。XLA报告driver CUDA12.8比PTX编译器12.9.86旧，关闭并行编译；不改驱动/环境，实际耗时需包含该影响。GPU5–7绑定尚待各自实际探针。

## 待完成清单

1. 确定提交并完成本地全 Agent/回归与独立 runner审查，记录 SHA、实际命令与证据 hash。
2. 恢复 sv3：实时核查物理 GPU4–7、UUID/CUDA/EGL、Git/环境/checkpoint及磁盘，隔离同步确定版本。可立即完成的本地准备不依赖连接。
3. 真实 size50m GPU探针/梯度/optimizer验收，两个 checkpoint各32独立episode，真实 segmentation像素和语义视觉验收，物理复原验收。
4. 按冻结8/24分割审计独立标签覆盖、增量辨别、背景误保护、cuecost、方差限制、活跃压缩，以及校准 S/P/W有效性；失败留证据、另版修订，不在盲测调阈值。
5. 上述通过才运行18条100k短训练并按训练 seed审计对照、资源与退化；最后逐项裁决五个原目标。缺证据不写“可正式实验准备就绪”。

继续本次已授权任务；无需新建或自动消息其他 chat。恢复入口信息属于访问条件缺失，不是重新申请已有算力权限。候选科研设计与真实实证严格分开。
