# DCS 仅背景环境准备与验收

**2026-11-06更新**：用户已在新批次确认Vision20 dynamic、baseline/原Dt、seed0、DAVIS train60/val30、video_alpha1，以及14任务上游地面规则/另6任务0.3；当前sv2八条缩小预算完整agent CUDA验收通过，正式队列和容量安排待答复。补充EGL检查发现额外渲染前后低位RGB不保证逐像素一致，原严格检查保持失败；纹理、物理及RNG恢复证据与GPU图像边界分开记录，不能把本文的CPU逐位证据推广到GPU。最新代码、运行与限制见 [DCS动态批次报告](DCS_DYNAMIC_VISION20.md)。以下为2026-11-05历史环境准备记录。

日期为2026-11-05。用户已确认准备 clean、静态背景、动态背景三种可切换条件。本次交付环境实现、素材审计和工程检查入口。新增含噪训练的任务矩阵、强度、素材划分、seed、预算及服务器/GPU尚未声明。已有 clean v1 及其扩展配置保持原决定。

代码与实际测试版本为 `cdeddec66c7a5c3987c9ee54640391a4e8e52fcd`，父版本为 `9c0a44b55c18866150a054090da2e978e74400e5`。本机源码检查与CPU验收完成后记录证据，后续交付提交仅补文档。同步范围仅本机Git，本轮未连接、安装或部署服务器，未核查服务器GPU或改变已有作业。

## 实现和语义

`embodied/envs/dmc.py` 原文件保持不变。新增 `dmc_background.py` 适配器和 `dcs_background.py` 背景组件，由 `dreamerv3/main.py` 环境工厂按配置选择。clean 仍构造原DMC，并移除新增配置项后传入原参数；不构造背景RNG、不读取素材、不改变天空或地面。原训练算法、梯度、回放、计步与评价runner源码保持不变。

| 条件 | reset | 后续环境动作 |
| --- | --- | --- |
| clean | 原环境 | 原环境 |
| static | 独立RNG选择视频和一帧 | 同一episode内保持该帧 |
| dynamic | 独立RNG选择视频、起始帧和方向 | 每次底层 `dm_control.step` 前进一帧，沿用上游边界反向规则 |

背景包装位于 FromDM 和 ActionRepeat 之前。repeat=2时，一个外层动作通常推进两帧，底层终止时提前结束；reset不会作为训练动作。背景文件按名称排序，DAVIS的补零编号给出时间顺序。额外render、读取诊断或生成报告不推进视频。没有引入颜色或相机包装器。

上游固定为 Google Research `5b09c22d73a9d35eb6c5d2a99b95677a45053466`，原始 `background.py`、`suite_utils.py` 与Apache-2.0许可证保存在 `third_party/distracting_control/`，背景源码SHA256为 `1a5f95fa3b53dabe7111549a5fec38fabab17256bab35f9f5e28cf756119e1d9`。本地标识为 `pd-dcs-background-v1`。

明确的移植差异包括本地文件访问、确定的帧顺序、Pillow bilinear缩放替代TensorFlow resize、最多缓存两张已缩放纹理，以及非法素材/纹理布局立即报错。alpha混合始终使用初始clean天空，避免跨reset反复混合产生漂移。此实现是DCS仅背景适配，不声称与原TensorFlow版本逐像素一致，也不是完整背景/颜色/相机联合DCS协议。新代码不依赖TensorFlow。上游随机采样和动态边界已在无需缩放的fixture上直接对照。

`render_background_frame(video, frame_index)` 在固定物理状态下临时替换纹理并返回图像，退出或抛异常均恢复原纹理和背景位置。它必须在该环境暂停step时调用，不支持并发操作同一环境。它不推进物理状态、不采样动作、不改agent history。配对干预的科研设计和模型推理仍属后续工作。

## 配置入口

三个环境overlay为 `dcs_clean`、`dcs_static`、`dcs_dynamic`。共同设置 `image=True`、`proprio=False`、`use_seed=True`，任务仍使用 `dmc_walker_walk` 等原任务名。overlay只配置环境，不选择训练模型/组别/预算。与原 `m2_v1` 等 `protocol_*` 入口组合使用含噪条件会在构造agent前拒绝，原正式准备校验同样拒绝含噪条件。

| 配置字段，前缀 `env.dmc.background.` | 默认值或要求 |
| --- | --- |
| `mode` | clean / static / dynamic，默认clean |
| `dataset` | 非clean必须显式给出 `DAVIS/JPEGImages/480p` 根目录 |
| `videos` | 非clean必须选 `train`、`val`，或逗号分隔的具体视频名称 |
| `num_videos` | -1表示使用上述完整选择；正整数取该列表前N个，不随机抽子集 |
| `video_alpha` | 上游组件默认1.0；非clean要求有限且大于0、不大于1；正式值仍须声明 |
| `ground_plane_alpha` | -1表示上游按domain规则；其他值须在[0,1]且有限 |
| `seed` | -1或显式uint32根种子，与环境seed共同派生，不直接作为全局随机种子 |
| `cache_size` | 默认2张纹理，仅影响缓存，不改变采样与帧序列 |

上游地面规则是reacher=0、walker/cheetah=1、cartpole/finger/ball_in_cup/humanoid=0.3。Vision20中14任务可使用这些规则。另6任务为hopper_hop、hopper_stand、quadruped_run、quadruped_walk、acrobot_swingup、pendulum_swingup，必须显式指定地面透明度。本次跨20任务工程fixture为这6任务使用0.3，**不是正式科研决定**。地面透明度改变可见地面，记录时不能省略这一因素。

背景RNG为独立的 NumPy RandomState/MT19937。种子取NUL分隔字符串 `[pd-dcs-background-v1, environment_seed, background.seed]` 的SHA256前4字节，按大端转uint32。环境seed仍走原工厂/调用者的映射；给定不同训练/评价环境seed即可得到独立背景流，不额外消耗环境、全局NumPy、agent、report或shuffle随机流。新含噪训练协议须声明各阶段环境seed映射。

`background_state()`记录派生seed、原环境/root seed、素材文件列表hash、有效alpha、episode、视频/帧名、方向、步数和完整MT19937状态。它是诊断快照，不是可恢复的整个环境/回放checkpoint。当前没有新增环境恢复承诺，原runner拒绝不一致续跑的限制保持。

## 本机素材和运行方式

证据根为 `analysis/outputs/dcs-background-preparation-20261105/`。隔离Python在该目录 `venv/Scripts/python.exe`。数据在 `assets/DAVIS/JPEGImages/480p/`，来自官方480p trainval归档，归档832766765字节，SHA256为 `e3d0b5b77c3d031b000a19e0e25e3e2cac65d183755601bc2cf066df1a2aa492`。已逐文件hash并解码6208张RGB图像，训练60视频4209帧、验证30视频1999帧，视频集合无交集。此处train/val是可用素材集合，不表示已经批准新实验的划分。素材及大产物均由Git忽略。

素材审计入口如下。output须为新目录，已存在时拒绝覆盖。

```text
python -m scripts.prepare_dcs_background --dataset <DAVIS/JPEGImages/480p> --output <新的审计目录>
```

它生成 `train-assets.json`、`val-assets.json` 和 `environment-plan.json`，记录逐帧hash、素材检查、20任务参数缺项以及 `authorized_to_train=false`。运行依赖见 `env/requirements-dcs.txt`，服务器原依赖快照已经含所需Pillow；本轮只安装了本机独立环境，未改变服务器环境。

下面是环境工厂使用示例，seed和素材仅为工程示例，不包含训练调用。将overlay换为 `dcs_clean` 或 `dcs_dynamic` 即切换条件。

```python
from scripts.prepare_m2_v1 import resolve
from dreamerv3.main import make_env

config = resolve([
    '--configs', 'dcs_static', '--task', 'dmc_walker_walk',
    '--env.dmc.background.dataset', '<DAVIS/JPEGImages/480p>',
    '--env.dmc.background.videos', 'bear,bmx-bumps',
    '--env.dmc.background.seed', '19',
])
env = make_env(config, 0, seed=7)
```

真实环境检查入口为 `scripts.probe_dcs_background`，要求显式选择后端。此次本机使用隔离Mesa 26.2.6 MinGW x64，真实renderer为 `llvmpipe (LLVM 23.1.4, 256 bits)`，全部使用CPU。DLL只在该进程中显式加载，没有系统级安装。Mesa归档SHA256为 `da0f71d9181b672c1918f3c342f11118c687f6059a46b16502845160733cc388`。

```text
python -m scripts.probe_dcs_background --dataset <DAVIS/JPEGImages/480p> --videos bear,bmx-bumps --all-vision20 --unmapped-ground-alpha 0.3 --steps 16 --backend glfw --software-gl-dir <assets/mesa-mingw/x64> --output <新的检查目录>
```

Linux可选 `--backend osmesa`；CUDA/EGL验收使用 `--backend egl` 并先按AGENTS核查获分配的物理卡及设备映射。本轮没有执行服务器/EGL检查。

## 验收与证据

实际测试SHA为 `cdeddec66c7a5c3987c9ee54640391a4e8e52fcd`。测试时训练/环境/脚本/测试源码与该提交一致，用户原有文档草稿保留。Python3.12.14、NumPy1.26.4、dm-control1.0.48、MuJoCo3.15.0、Pillow12.3.0、JAX/jaxlib0.4.33，`pip check`通过。与服务器Python3.11存在平台差异，不能据本机检查宣称服务器完整agent通过。

- `regression-cdeddec.log`记录56项通过、无跳过，耗时24.369秒。覆盖新增9项背景测试、原probe/overlay梯度、free-nats、全局shuffle、协议计数/去重/恢复、DMC16和Vision20准备检查。设置 `JAX_PLATFORMS=cpu`、`MUJOCO_GL=disable`、`XLA_FLAGS=--xla_force_host_platform_device_count=2`，跨shard测试实际在两个CPU逻辑设备执行。
- `preparation-cdeddec/`记录全部素材内容hash、完整图像解码与任务环境计划。
- `render-vision20-cdeddec/`通过原生产 `make_env` 工厂执行20任务的clean/static/dynamic，以及带额外配对诊断的dynamic，耗时107.109秒；参照为父版本原DMC加相同外层动作包装。每次18条观测包含2次reset，16次动作。逐位检查clean图像、固定动作下全部条件的物理状态/奖励/边界/任务RNG；静态背景在episode内保持、动态帧确实推进，图像变化非零；诊断后图像和背景RNG与未诊断运行逐位一致。
- `render-boundary-cdeddec/`以cartpole_balance、repeat=2、504次外层调用、只在起始显式reset的工程fixture，检查真实episode结束、后续自动reset、repeat累计奖励与背景推进，耗时87.187秒。五种检查路径各观察到1次真实episode结束，动态背景的episode计数也在自动reset时递增。这个fixture的调用数不是正式训练动作预算。
- 所有数值一致性使用精确数组相等，不采用事后放宽容差。扰动作用检查要求实际图像坐标不同率大于0，不设置科研效果阈值。三条件真实图像位于每任务 `conditions.png`，物理/图像轨迹为对应NPZ，完整背景RNG轨迹为JSON。
- 已人工查看walker_walk、reacher_hard和quadruped_run三条件图，主体及目标仍可见；视觉查看限于这些样本。`unchanged-training-source.json`记录6个原算法/环境/runner文件与父版本逐字一致的hash；`local-environment.lock`保存本机依赖快照。`DELIVERY.json`记录最终文档提交和同步范围，证据文件清单为 `evidence-sha256.json`。

复现回归命令如下。

```text
python -m unittest -v tests.test_dcs_background tests.test_rep_probe tests.test_rep_overlay tests.test_protocol_v1 tests.test_dmc16_preparation tests.test_vision20_seed1
```

初始失败日志保留，包括配置fixture未给任务、先前直接改DMC触发原源码不变检查、Windows渲染器查询绑定问题。最终用独立环境适配器保留原DMC/runner/算法源码，原源码不变测试原样通过；渲染器查询绑定到明确加载的Mesa库。没有放宽算法测试或数值容差。初次安装portal版本不存在也已改为锁文件中的3.8.1，仅发生在新的本机venv中。

## 剩余条件

本阶段完成本机环境准备与固定动作CPU渲染验收。未执行含噪模型训练或固定策略含噪评价，未运行新的完整agent/CUDA验收，也没有抗干扰或方法有效性结论。新的正式训练/评价矩阵及背景强度须由02与用户明确，服务器和物理GPU由用户分配。14任务可用上游地面规则，另外6任务的正式透明度仍待决定。

下一步继续合并窗口03，在已分配资源上做真实CUDA/EGL与完整agent短程验收，记录设备一致性、背景素材hash、吞吐及随机流隔离。科学协议与任务矩阵确认后再准备正式运行；本次环境接入不解除新含噪实验的协议缺项。
