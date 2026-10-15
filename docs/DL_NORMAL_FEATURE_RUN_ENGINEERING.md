# 正常共同历史 reference0 特征执行入口

2026-10-15，`DL-normal-feature-run-r1`。本文件记录已委托实现的必要工程接口；
**未采用实际特征执行的 GPU 卡、UUID、秒数或启动安排**，也不采用 T4096、训练、
native 评价或方法性能判据。入口实现于 `scripts/dl_normal_feature_run.py`，
只加载原 N1M reference0。来源定义仍为 `docs/DL_NORMAL_READOUT.md`，
逐帧模型接口仍为 `docs/DL_NORMAL_FEATURE_ENGINEERING.md`。

## 固定范围与认证次序

1. CPU 父进程从完整 R5 原始 run、原 producer launch/terminal、成功 prior physics
   audit report/launch/terminal 的五个显式 path/rawSHA 调用真实
   `source.authenticate_and_index`。完整512条、32个 family、前8 fit/后24 held 的
   ID/顺序/原 RGB、原 raw、物理 prior 及完整库存均先认证，再允许可信 pickle 解码。
   不接受 partial 来源、旧 latent/carry 或按 D/e/Y 选择样本。
2. **必须**提供成功的512条当前任务标签和外层真实退出凭据，分别由
   `--labels/--labels-sha`、`--labels-terminal/--labels-terminal-sha` 显式绑定。
   外层版本为 `DL-normal-labels-launcher-r1`，字段为 `labels_output`、`success=true`、
   returncode0、whole900内、原身份已退场、所有 owned group 已清理、无 external failure。
   此 terminal 仍不能自证 controller 最终成功退出；execution-plan 另强制
   `labels_outer_execution={path,bytes,sha256}`，绑定 root 实际观察的外层 exit receipt。
   该文件须明确成功 labels 自己的 Git（不是将来 feature Git）、实际 ssh/options/sv3/
   project Python/controller/expected-git 命令、returncode0、whole≤900、GPU false。
   它的 `log_sha256` 属于外层 SSH 输出，不拿内部 `labels.log` 的 hash 代替；本入口不据
   该单独字符串声称重新认证完整外层日志。该 raw pin 加入 immutable 与初末闭包。
   核标签 source ID、prefix、fit/held、当前 obs/snapshot、F64原字节和原 query/index/labels
   Git blob；每条提取前再绑定 `reward_f64[t-1]`、`obs[t].reward` 和静态 MJB。
   这里只消费先前成功标签的证据，不重新执行物理，不把任务标签称 actor-critical 真值。
3. CPU 父进程封口 source index、labels pins、原模型输入、全部当前 Python/YAML/lock
   及必要 norm 的真实 Git blob 后，只产生一个独立 session 的 owned GPU worker。
   worker 的直接父进程、原 PID/start/PGID/SID、原 budget SHA 和剩余 whole deadline 必须一致。
   CPU 父进程不导入 JAX；CPU 审计库会清空 CUDA 变量，因此不在 GPU worker 内再次调用它。
4. worker 单次加载原 checkpoint SHA
   `c6a7d92e24669a515bb44693faf73e442d9868f9fc814eec0e22e069aa594d24`、原 config SHA
   `877f8470cb4f7f45d143e3099fbd3f55a828991afe33779b7b04c4264f6c8377`。
   对实际解 pickle 的同一 byte buffer 核原 pin；294叶、原 Python int counters、完整状态
   `4e0d1a0e1de806bf8e4863278cabd171ed0d208dddc1fea18b8b54d68ca6383f`、参数
   `cffc13c0dcdfbd0c8b0e5c503c9d6251f4b1550c08b85a8742a638655d66d372` 均须匹配。
   原 policy copy 开启且与当前 params 相同；seed0/BF16、NumPy1.26.4/JAX0.4.33/
   threefry2x32/partitionable false/x64 false 不变。无训练或环境创建。
5. 每条来源真 reset，从 `obs[0]` 连续到 `obs[t]`，只向 outgoing history 的 prev raw
   写入 `raw[:t]`；每个模型自己的 h/z 独立演进。原 EVAL seed/counter 和模型原两次
   nj.seed 不变；官方 unimix q/p 概率查询不消耗 RNG。**每帧**真实调用 public policy，
   要求 native raw 和完整 after-carry byte exact，然后才做 source raw host override。
   当前 `obs[t]` 必须调用 `observe_current`，没有 normal `raw[t]`、没有可继续推进的
   source history after，不使用干预动作。完整来源 projector 会验证全数组合法性；
   这不把干预动作作为模型的当前来源命令或特征输入。
6. 每条保存 `current.npz`、`history.json`、`complete.json`。当前张量为原 BF16 h4096、
   q/p logits32×32、F32 q/p probabilities32×32、F32 native raw12、三个原 uint32 key。
   BF16 按 uint16 原 bits 保存，schema 保留逻辑 dtype、shape、raw SHA 和 typed SHA。
   history 包含所有 t+1 帧的源绑定、两种 carry hash、native raw、keys、公共 oracle 和计数。
   reset/finish 保存完整 state、policy copy、counters 的 readonly seals；不发布旧 carry 特征。
7. worker 真实退出且原 owned session 已清理之后，CPU 父进程用**显式原四 producer 参数**
   重认证完整来源、prior、库存和五 external pins，再核 labels、原模型输入、当前源码。
   所有512条 artifact/typed tensor/帧收据及完整输出库存双遍一致才写
   `generation_complete.json`；`supervisor_terminal.json` 仍不能替代外层实际 parent exit0。
   CPU 闭包还逐值串起 reset.initial-carry→首 before、每 history-after→下一 before、
   source raw hash→下一 source previous-raw，核当前 outgoing 两 hash 为 None，并将当前
   NPZ 的 root/latent/actor 三 key 与末帧 history 对齐。重签普通文件 SHA 不绕过这些约束。

## 执行计划、设备与预算边界

入口没有默认 GPU 执行秒数。`--execution-plan` 和 `--execution-plan-sha` 是必需项，
schema 为 `DL-normal-feature-run-execution-r1`，必须明确：

- `git_sha`：实际确定、clean、完整40位 feature runner Git；
- `role='reference0'`、`complete_units=512`；
- `logical_server='sv3'`、实际 `physical_gpu` 和单一 `gpu_uuid`；
- `whole_seconds`：以后由 root/用户采用的一次整入口硬上限，必须为正整数。
- `labels_outer_execution`：上述成功标签 controller 的实际外层退出文件 path/bytes/rawSHA。

不能把本接口存在、正整数 plan、旧6h建议或历史单次推理资源当作启动授权。
`ENTRY_STARTED` 在首个 import 后立即计时，所有 CPU、hash、解码、模型初始化、逐帧
oracle、保存和退出闭包都计入同一 whole；子进程继承原绝对 monotonic deadline，不能重置。
初始/末遍 CPU 来源认证各受自身900秒边界和全入口剩余预算限制，不是额外 GPU 配额。
**实际外层还必须在 Popen 前启动已采用 whole hardwatchdog**，覆盖 imports、读取并解析
execution-plan 到内部 timer 安装之前的 bootstrap 段；程序没有以未采用数值代替这一要求。
外层需验证真实 parent exit0、whole wall、supervisor/generation raw pins 与 owned closure。
本轮未实现或运行该外层，没有 actual execution plan。

仅 compute-only：`CUDA_VISIBLE_DEVICES=<单一UUID>`、`CUDA_DEVICE_ORDER=PCI_BUS_ID`、
`JAX_PLATFORMS=cuda`、`MUJOCO_GL=disable`，两项 EGL 变量 unset。
启动模型前重新读 idle，初始化后核真实 compute UUID，父进程约15秒采样；先持久保存
可用 XML PID/type 摘要及 raw pmon，再因未知 PID、非 C 类型或错卡失败。
不创建 EGL probe、环境或 GL context，`egl_uuid=None/graphics_verified=False`。
cleanup 使用已审 CPU auditor 的原 PID/start/PGID/SID ownership fences，只清本次 group，
不会停止发现的未知 GPU 作业。资源冲突、超时、缺件或 oracle 不一致保留新尝试证据，
不自动重跑、不续拼、不降低 oracle 精度、不以 partial 发布可读出成功。
初始来源认证尚未返回时，timeout handler 仅从 `sys.modules` 取得已加载
source.AUDITOR，委托同一 fenced registry 清理已有 metadata session；handler 不做新 import。
这补上 parent 尚未赋全局 AUDITOR 的窗口，不绕开原 PID 复用/foreign SID 拒绝规则。

未知量包括完整 N1M 在每帧 exact oracle 下是否全部通过、单次完整运行用时和显存峰值。
有限 CPU fixture 的用时不能估计实际 GPU ETA，也不能证明整入口预算足够。

## 本机有限验收

命令：

```text
C:/Users/97370/AppData/Local/Codex/dl-validation-env/Scripts/python.exe -m unittest tests.test_dl_normal_feature_run -v
```

9项、34.446秒、8项通过/1项 Linux 专用 skip、0GPU；同时通过 `py_compile`。
上述本机测试时尚未提交，测试基线 HEAD 为
`47db6265311bc04dbdcc79439842eb8b6e707097` 加本轮新 runner/test；不是某个已提交
Linux/GPU版本的验收。首个6项过程曾因 fixture 把来源 raw[t] 改成 NaN 被原 projector
全数组 finite 检查正确拒绝；改为有限不同动作，原来源检查没有放宽。

本次实际测试文件 raw SHA256：runner
`7571f6226e36629b8e2dbdc60a312dcf095e940732c95026303637e5ef4b9f38`，test
`96dfd3e7ed8ef1f3e817818262ccc786b64f56ec1df113a416f15fa7b640cb29`。

证据范围严格分开：真实小 BF16 图像64/action12 Agent 在**人工生成**的64条来源命令、
65帧观测上执行全部65次真实 public/native oracle；核当前 raw[t] 不作为特征输入、
BF16 原 bits 保存、完整 counters/state readonly、错误 source ID 和 oracle 篡改拒绝。
另外是**人工构造**512项 artifact 图的闭包测试，其 clock0/1 明确不满足真实 source plan，
不调用真实来源认证或512模型前向；它只检查 publication graph、末文件篡改和 partial 拒绝。
新增重签 latent-key 和 carry 断链反例，普通 raw/spec pins 同步更新也必须拒绝。
成功 labels metadata 的来源/退出/Git authority 也明确 mock，未当作真实标签。
初始 auth 未返回时的 mocked registry 使用真正已审 group_alive，核 PID 复用和 foreign SID
不发 SIGKILL。新增真实独立 metadata session 加 SIGALRM 的 Linux 反例在本机 Windows 跳过，
只有未来确定提交 clean Linux 实际执行该项后才算该真实进程路径验收。

本轮实现/测试没有真实 R5/N1M/任务标签执行、无 SSH/GPU/physics/render/学习/读出/科学裁决。
成功标签的实际外层小 metadata 仅按 root 指定用于核对输入 schema；真实标签执行由 root
另行取证，本轮没有将其用于模型提取。
真实 N1M 的 public/native byte exact 若失败，必须拒绝并保留工程证据，不调容差。
future T endpoint 即使复用 `extract_prefix` 核心，也必须另行认证 checkpoint、完整 state、
训练来源和 endpoint 协议；本入口不把 arbitrary endpoint 或 task label 认证为 actor 利用或因果关键性。

root及独立Agent已只读复核上述三处修复，runner/test原字节同上述pins；
下一确定提交的clean Linux六模块测试须实际执行Linux专用alarm反例，仍不构成GPU提特征启动。

## 确定提交的实际Linux CPU验收

实际/测试Git `2a15c91aad16c0e3151fde3943e356dad1fcb69d`，sv3 fresh clean目录
`/data/Policy_Discrepancy/dl-normal-feature-run-code-20261015-2a15c91`。
六模块（feature_run/features/live_policy/source_index/task_query/labels）36项全过、无skip，
unittest45.954秒、整入口47.492050930秒/300秒，真实root SSH exit0/48.078秒；
pending-authentication真实独立metadata session加SIGALRM反例实际执行通过。
256份实际Git源码和七份额外规范的working bytes前后核同、最终Git clean、owned均空，0GPU。
真实小BF16模型及generated65帧、人工512图、mock labels的上述证据边界均保留；
本次未读取真实R5/运行真实N1M、未执行真实512提取、物理或方法检验。

证据在`/data/Policy_Discrepancy/runs/dl-normal-feature-run-cpu-20261015-2a15c91/attempt01/`，
本机`analysis/outputs/dl-normal-feature-run-cpu-2a15c91/`。terminal1224B rawSHA
`7a96338386e8cf49753a8ea1c00602696c20d68efb2c2cca2487fbef84ca1306`；
test log7930B `8a3f4d5b2b1dfbd85a2eb3a8eb19ba0fcc59d2b1026a6877fc2fb1fd6de268f7`；
source-binding83612B `87dd6ab6ce089c272541467abc64d78bd85fcc5e5c16e01237398ec971eb3656`。
外部controller `4840f7f0bd01308b86127a517e3ac7e612347984361da512a7950b084a0c3d3d`，
source bundle `149bdf289e24da5cbde5f4668d542af720699bd8b17f81e0e707aca248b47e65`；
SSH外层log `582a357fd35c685f16420b6de469fe5069b9a2fea9c0af141f3a9279e07664c3`。
此成功不采用任何GPU执行计划或whole预算；仍须先冻结实际计划、bootstrap外部watchdog和资源核查。
