# DL-native-render-identity-r1：零动作渲染身份诊断

## 状态与范围

**Agent 在既有完整预验证授权内采用的必要工程诊断，尚未执行。** 本文不是正式 M2 协议，也不是方法有效性、独立第二信号或选择性保护的实验。root 负责冻结提交、Linux 工程验收、实际资源核查与外部 launcher；本文件和实现的 CPU fixtures 不构成服务器/GPU或真实渲染验收。

必读 `AGENTS.md`、`docs/DL_REP_NATIVE_ASSAY.md`、`docs/DL_REP_NATIVE_RESULTS.md`、`docs/DL_VALIDATION_LOG.md`。实现为 `scripts/dl_native_render_identity.py`，fixtures 为 `tests/test_dl_native_render_identity.py`。起草只读审查基准 HEAD 为 `422ae48e9f1c4728c3e5cfe57a48ce6e2e12b10d`；实际执行 SHA 须由 root 在提交并验收后另记，不能以此起草 SHA 声称脚本已执行。

原 native r1 的三个组各32 episode 已终止，但 statistics 因同 seed 的初始 RGB hash 不同失败。相同 integration/task/reset 收据不足以通过原配对要求。本诊断**不读取旧 episode、checkpoint 或回报，不补拼、不放宽、不翻案，也不启动 observation guard**。诊断中出现合法 RGB 差异是可保留的结果；完成状态与图像是否相同分开。

## 已知源码事实与未证实解释

sv3 的实际安装为 `dm_control 1.0.48 / MuJoCo 3.15.0`，公共前缀为 `/data/Policy_Discrepancy/envs/dreamer/lib/python3.11/site-packages/`。

| 已只读核验的对象 | 证据 |
| --- | --- |
| task seed 与模型编译的顺序 | `dm_control/suite/quadruped.py:99–101` 先 `Physics.from_xml_string(xml, common.ASSETS)`，后 `Move(...,random=seed)`；`base.py:48–50` 为 task 创建独立 NumPy RandomState |
| 程序生成的 skybox | `suite/common/skybox.xml:3–4` 为 gradient、`mark="random"`；quadruped XML 引入该文件 |
| 星点随机流 | wheel 未附 C++实现文件；只读安装库未剥离静态符号及反汇编可见 `mjCTexture::Compile→randomdot`。`randomdot@0x3fb730` 在内部栈上初始化 PRNG，明确种子 `0x2a=42`，该函数无 `rand/srand` 调用。库中 `rand` 调用位于 `mju_standardNormal`。这是该安装库的静态证据，不是此次图像故障复现 |
| 原 snapshot 的边界 | `mujoco/include/mujoco/mjtype.h:508–532` 的 `mjSTATE_INTEGRATION` 为 mjData 状态，不含 mjModel 的 `tex_data`、材质、相机、灯光、visual 参数或 GL context |
| 固定相机与画图链 | `embodied/envs/dmc.py` 固定 camera2；官方 `mujoco/engine.py:818–838` 经 `mjv_updateScene→mjr_render→mjr_readPixels`，`533–557` 延迟建立 GL/MuJoCo context |

不能从 skybox 的 `mark="random"` 直接推出全局 C RNG 根因；**不得盲加 libc.srand、替换纹理或选取匹配旧图片的 seed**。CPU 编译模型、派生变换、scene、纹理上传、GL context/draw/readback 都仍可能涉及未测差异；当前没有证据将其称为随机图形噪声。

实施硬绑定安装库 SHA256 `d3a1476dd52e220031aca40b2b00f69d1febce2ee85c57d5015f68b0cc72afa9`，以及 quadruped/XML/三个 common assets/core/engine/EGL 源码的既有 byte pins。脚本中的 `PINNED` 为完整清单；实际路径、版本与 hash 保存到各 worker 收据。版本或字节不符即停止，不能现场改 pin 后宣布沿用本对象。

## 唯一固定对象

| 项目 | 采用值 |
| --- | --- |
| worker | 两个新 owned worker，顺序0、1；串行且不共享卡 |
| 每 worker | 两个 fresh 官方 `suite.load('quadruped','walk',task_kwargs={'random':1024})` 环境，顺序0、1 |
| 正常接口 | `FeedbackEnvironment` 原 production wrappers；action repeat1、64×64 uint8 RGB、camera2、proprioFalse |
| 每模型 | 一次正常 `FeedbackEnvironment.reset()` 得到首帧；随后两次原 `physics.render(64,64,camera_id=2)`；不调用 suite.step 或 actor |
| 总量 | 4 reset、12实际 RGB、0 suite transition、0 actor、0 Agent 实例/调用/更新、0 checkpoint/旧episode读取 |
| 资源 | 仅 sv3 物理 GPU4，UUID `GPU-9c070d5a-ac68-4c26-9522-a790ad874b23`，单卡独占 |
| 硬预算 | parent 入口开始、标准库其它 imports 之前起计180秒，包含 hash/imports/metadata helpers/两worker/env/render/统计/写盘/退出；单卡最多0.05 GPUh |

180秒是硬上限，不是预计耗时或吞吐保证。首个合法差异不停止后续固定模型/渲染，也不新增 seed、环境、render、action、worker 或预算。输入坏合同、源版本不符、占用冲突、进程/计数不全、提前/超时退出均不能认证完整。失败尝试保留，新尝试使用新目录。

worker 不实际创建 CUDA compute context。`JAX_PLATFORMS=cpu` 在 worker 依赖 imports 前固定，原 wrappers 可以导入框架但不构造 Agent；实际 `verify_runtime(compute=False,graphics=True)` 必须确认无该 worker CUDA context、实际 EGL 进程在物理4且无别的作业。外部已绑定 probe 的 CUDA/EGL 同卡证据仅说明该 probe 的历史映射，**不能写成诊断 worker 已实际执行 CUDA**。

## 实际证据与记录方式

每模型保存官方 `physics.model.to_bytes()` MJB、公开数值 model 字段（包括 tex/mat/cam/light/vis，保留 dtype/shape/原 bytes）、完整初始 integration/task RNG/clock/reset/FromDM done snapshot，以及每次渲染的派生 body/geom/site/camera/light transforms。

runtime 只读 hook 包装官方 `Camera.update` 和 `Camera.render`，不改变 camera、scene flags、GL 调用或输出数组。它在实际 `mjv_updateScene` 之后、实际 draw 之前读取 active geoms、lights、两个 scene camera、scene option/perturb/render camera 的公开数值字段。未知、opaque 或非有限的可选 introspection 字段记为 `unknown/unmeasured`，不填零、不假造 dtype/bytes；坏的真实 snapshot、RGB或reset合同仍严格失败。完整 scene/shader/context/upload/framebuffer/readback 输入始终没有独立认证。

正常 render 外层分别比较渲染前后完整 snapshot、MJB、已测 model/derived typed 字节。原缓冲区旧内容 hash、写后 buffer hash、实际返回 RGB hash、返回值与 buffer 是否共享内存均记录。每次立即保存独立 RGB copy；随后渲染检验之前的 live view 是否改变，同时保持已保存 copy/hash。旧 buffer 是可能未初始化的 uint8 内容，只用于定位覆写/alias，不要求旧 buffer 跨调用或跨 worker 相同。

模型之间比较：四个初始 snapshot、各模型首帧的 MJB、typed model/derived/scene manifest、unknown覆盖与 RGB。每模型比较首帧对后两帧；不新增 segmentation/depth、背景修改、重新创建 GL context 或渲染修复试验。

## 分类、完成条件与限制

| 输出分类 | 允许的解释 |
| --- | --- |
| `measured_CPU_numeric_input_difference` | 已测 model/derived/scene 数值输入存在差异；记录具体层/字段，不直接说明为何不同或哪个差异导致 RGB |
| `MJB_byte_difference_without_measured_numeric_difference_unresolved` | MJB 序列化不同但已测数值相同，可能涉及未测字段或 padding；不能冒称已定位真实视觉根因 |
| `measured_input_coverage_difference_unresolved` | introspection 覆盖/unknown不同，输入等价尚未建立 |
| `same_measured_CPU_inputs_RGB_difference_unresolved` | 已记录 CPU输入相同而 RGB 不同，context/upload/draw/readback 或其它未测输入是后续定位候选；不称全渲染输入等价或图形噪声 |
| `same_measured_inputs_and_RGB` | 此有限新对象的已测输入和 RGB 相同；不能据此否定旧完整 native worker 的问题，因为未复演旧 Agent/CUDA/加载历史 |

所有 MJB、typed model、typed derived、typed scene 的 equality 分开输出，不能把仅 MJB 不同混成视觉数值差。完整执行要求两 worker正常 exit0/terminal、全部4模型和12RGB、精确零动作计数、最终源/Git/产物 hash不变，以及 parent与外部 CLI 全入口均未超过180秒。合法图像差异可以 `complete_verified=True`，它不代表等价或根因已经解决。

CPU fixtures仅验证数值分类、typed dtype/shape/signed-zero、snapshot合同、RGB copy/alias、未知字段、bounded owned process与 late发布。它们不构造官方环境，不渲染，不访问真实 checkpoint/RGB，不提供 GPU复现证据。后续 root 的 Linux CPU验收仍然不是实际渲染验收。

## 入口、产物与验收链

```text
python scripts/dl_native_render_identity.py
  --output /data/Policy_Discrepancy/runs/<新诊断目录>/<新attempt>
  --expected-git <已提交、Linux工程验收且checkout clean的完整SHA>
  --resource-record <root新核独占GPU4、既有device probe已绑定的resource JSON>
```

内部 `--worker-index 0/1` 只接受真实 parent PID/env/budget/独立owned组认证，不能作为绕过 parent预算的独立入口。parent与 workers 使用同一 whole deadline。ALRM/TERM在 Popen及owned登记期间阻塞，child恢复原 signal mask；metadata child在worker内继承其phase组，parent只清理自己创建的组。超时 receipt只能留下 pending/late/provisional 文件，不能留下可验收的成功 completion。

新目录保存 `budget.json/plan.json/worker{0,1}-resource.json/worker{0,1}-exit.json/worker{0,1}-complete.json/result.json/supervisor_complete.json`及各日志。每模型保存一个 snapshot.pkl、MJB、含三次真实CPU字段的 NPZ，以及三份 RGB.npy/debugPNG，总四snapshot、十二RGB。generated pickle由可信 owned worker产生；parent只hash，不解 snapshot、RGB或 MJB，不把 CPU helper审计说成独立renderer重演。

root 的外部 CLI terminal需明确 `actual_git_sha/worker_pid/worker_terminal/exit_code/timed_out/elapsed_cli_seconds`，其中 PID绑定诊断parent、exit0、terminalTrue、timed_outFalse、`0<elapsed<=180`。不得对缺字段默认False。`verify_completion(directory,external_cli_terminal)`核完整退出与hash，不重新执行环境或decode数组；root另认证真实launcher字节/命令、stdout与实际src/resource flags。

产物留在独立服务器 runs 目录；若镜像到本机，只使用 Git已忽略的 `analysis/outputs/dl-native-render-identity/<执行SHA>/<attempt>/`，不提交RGB、snapshot、MJB或其它二进制产物。此文未做当前GPU占用核查，未启动任何worker。

首选下一任务仍是完成这个身份诊断及根因定位。只有根据实际证据冻结工程修复并验收后，才能以新完整尝试恢复原 native r1；不能把此零动作诊断代替完整 native评价或五项预验证目标。
