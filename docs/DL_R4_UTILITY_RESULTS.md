# DL-r4 真实动作/记忆诊断运行记录

状态：**运行中，未验收，无科研裁决**。更新：2026-10-15 15:37北京时间。
科学规范 [DL_R4_UTILITY.md](DL_R4_UTILITY.md)，版本 DL-r4-utility-r1；正式M2不变。

## 版本与工程证据

- 主实现 `d61d8ae45b380dd4e54907e5669ce3cb731749f3`：sv3 clean Linux六组51项全部通过32.629秒，pip check0，完整34.724秒/0GPU。
- 实际执行/最终相关测试 `5ce1d3724ee8c5d01f09c4d687100a2f42f6ac83`：相对d61仅runner和独审各2行关闭超时管道，以及规范2行证据；没有模型、科学协议或随机流变化。sv3相关19项全过3.341秒/完整3.672秒，原ResourceWarning消失。其余未改模块的51项证据来自d61，不冒称5ce重新全测。
- 本机六组50项49pass/1Linux skip以及新增最终独审10项已通过；fixture中的stub Agent/RGB与真实tiny模型、真实CPU物理范围分别记录，不能当N1M方法验收。
- 代码仅同步sv3隔离checkout `/data/Policy_Discrepancy/dl-r4-code-20261015-d61d8ae`（目录名为初始实现，实际HEAD为5ce）；canonical仍d9be5574878d2c3f117cc6667be844f9a527eb84，sv1/sv2源码不改。
- 5ce已推研究分支codex/dl-prevalidation；main未合并，最后核验main595e982f1a3939775879728009e2105ee516e86d。后续报告提交相对actual只改docs，最终SHA以终点Git收据为准。

## 实际资源、输入与范围

sv3/lyg0326，物理GPU4，A100-SXM4-80GB，UUID `GPU-9c070d5a-ac68-4c26-9522-a790ad874b23`。
仅本对象一张卡；GPU5–7不预占。15:09:40.665775 fresh XML+pmon确认4–7各0MiB/无作业，
0–2有未知作业，未干预。随后零动作/零更新CUDA/EGL probe通过，probe阶段约3.67秒，成本另记。

CUDA_VISIBLE_DEVICES为上列UUID，进程内CUDA0；MUJOCO_EGL_DEVICE_ID=4，CUDA_DEVICE_ORDER=PCI_BUS_ID、MUJOCO_GL/PYOPENGL_PLATFORM=egl，XLA_PYTHON_CLIENT_PREALLOCATE=false。
15:10:36.937448 worker实际compute UUID和EGL C+G均仅GPU4，采样2575MiB，非全程峰值或最低显存保证。
版本/环境完整数据在model_authenticated.json；driver提示CUDA12.8比PTX compiler12.9.86旧，XLA关闭并行编译，暂未见因此失败，未修改驱动或环境。

输入 `/data/Policy_Discrepancy/runs/dl-r4-utility-inputs-20261015-0655/`：agent.pkl/config.yaml按原SHA认证，来源为sv2已验收N1M，513050093/4927字节；不在Git。完整params/optimizer/norm/counters load/save typed digest与294 leaves/params digest核验已通过并写model_authenticated.json；这不是独立重演训练。

实际任务 quadruped_walk，冻结size50m/seed0、vision64/camera2/proprioFalse/repeat1；新env/eval seeds4096–4127、cal0–7/held8–31，14普通位置D-top2主对象＋固定512辅助、三臂qq/pq/pp、16roots、H100；原H1按source共同raw和下一实际F64 reward评分。诊断config关闭训练gate/probe、prealloc/precompile/profiler，不更新模型或改变原checkpoint参数。完整源config和实际flat config均认证留档。

主完整入口21000秒硬限；source含建环境/关闭累计1800秒，模型资格1200秒，labels/oracle/physics/cached-policy累计18000秒；不是整个科研预算或完成ETA。父每15秒XML采样GPU4，仅允许worker PID；未知作业出现则停止本对象自己owned worker，不杀他人或换卡。

## 进程、命令与产物

15:10:12.809502北京时间启动parent247014/PGID247014，原/proc starttime407058309；
worker247226/PGID247226/PPID247014，starttime407058446。
15:12:06.883578二者均实际活跃，未有supervisor_terminal/worker_complete。
episode_000的source.pkl/signals.pkl已生成，尚未整episode完成或独审；不能据文件存在宣称科学验收。

**15:17:09.221456后续事实**：原parent/worker的starttime、PGID、cmdline均一致，worker R活跃，
episode0已完成主生成：147份branch/null，14700真实branch步及等量cached-policy/physics复放，
source14×100 F64/exact state参考通过；source累计9.827秒、labels累计322.840秒。episode1已创建，
cal尚未封口，held目录未生成，主terminal/worker_complete尚无。此为主程序工程检查，不是独审或科学裁决。
GPU4仅本worker247226/C+G、2587MiB；5–7各1MiB但XML+pmon均无作业，本对象未用，GPU4尚未释放。
现场小收据inputs/status-20261015T0716.json（文件名预置，实际检查时间以内容15:17:09为准）。

**15:37:15.241654后续事实**：原parent247014/starttime407058309与worker247226/starttime407058446、
PPID/PGID/cmdline一致且S/R活跃，episode0–4均有complete，episode5开始；尚无calibration_frozen、
held目录或terminal，未见工程错误。GPU4仍仅本worker C+G/2587MiB，5–7各1MiB但无作业；
GPU0–3出现未知作业，未干预。现场收据inputs/status-20261015T073716.json。
本轮并行的 [DL_FORMAL_READINESS.md](DL_FORMAL_READINESS.md) 仅0GPU审查/人工接口反例，
没有读取中途科学值、修改实际5ce或本运行的科学规范。

**15:40:23持续运行与Git限制**：只读监测确认原PID/starttime/PGID/cmdline一致、worker R，
episode5文件仍增长，cal/held/终态未改变，无工程错误。最新进度报告本机a559663797dc3ac2385cedea04f5a17de77c1ea9
推送遇connection reset及GitHub443连接失败，重试仍失败；最后成功远端核验研究分支为
95bc9d4812e5248c7a3db58e00f08f50c4f9986c，包含新的准入审查，main最后595e982未合。
后续本机终点报告也属于待补推，不把旧远端快照称现状；实际5ce已先推且sv3作业不受影响。

主目录 `/data/Policy_Discrepancy/runs/dl-r4-utility-20261015-5ce1d37-attempt01`。
inputs中的launch-5ce1d37-attempt01.json、resource-5ce1d37-attempt01.json及probe证据保存完整启动和mapping；
Linux检查log/JSON、transfer_verified.json、startup_resources-5ce1d37.json也在inputs。
本机小收据镜像在 `analysis/outputs/dl-r4-utility/preparation-20261015/`，不镜像checkpoint/rollouts到Git。

```bash
/data/Policy_Discrepancy/envs/dreamer/bin/python scripts/dl_r4_utility.py run \
  --expected-git 5ce1d3724ee8c5d01f09c4d687100a2f42f6ac83 \
  --checkpoint /data/Policy_Discrepancy/runs/dl-r4-utility-inputs-20261015-0655/agent.pkl \
  --config /data/Policy_Discrepancy/runs/dl-r4-utility-inputs-20261015-0655/config.yaml \
  --output /data/Policy_Discrepancy/runs/dl-r4-utility-20261015-5ce1d37-attempt01 \
  --resource-record /data/Policy_Discrepancy/runs/dl-r4-utility-inputs-20261015-0655/resource-5ce1d37-attempt01.json
```

## 验收与后续

当前没有独立信号增量、压缩/保护、性能结果，不读held科学值作中途选择。
须继续核原PID身份和终态；单次SSH观察超时不视为训练停止，不重启。parent本身有全入口ALRM及owned group清理。
完整生成后才用同一实际SHA运行CPU独审：

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu MUJOCO_GL=disable \
  /data/Policy_Discrepancy/envs/dreamer/bin/python analysis/dl_r4_audit.py \
  --run /data/Policy_Discrepancy/runs/dl-r4-utility-20261015-5ce1d37-attempt01 \
  --expected-git 5ce1d3724ee8c5d01f09c4d687100a2f42f6ac83 \
  --output /data/Policy_Discrepancy/runs/dl-r4-utility-20261015-5ce1d37-attempt01-independent.json
```

独审全入口1800秒，核全部hash后才解可信pickle，独立官方CPU source/branch物理、F64/fsum、三臂与cal封口；不独立重演模型/head/NJ子key/render，统计公式复用已测试模块。有限测量/覆盖/增量不足不追加有利样本或进入保护训练。任一失败保存完整attempt，新修复使用新SHA/new attempt，不覆盖或拼接。
完整方法仍需active rep损伤/保护、强度/shuffle/reward-only及跨seed学习证据，目标保持active。
