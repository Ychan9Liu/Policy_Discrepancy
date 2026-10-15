# 正常共同历史特征接口

2026-10-15，`DL-normal-features-r2`。Agent 在完整预验证范围内采用的必要工程接口；
不采用 T4096、训练/特征实际执行预算、native 或性能门。不读取真实 R5，不运行真实 N1M。

实现为 `scripts/dl_normal_features.py` 的 `FrozenNormalFeatures`，无来源/模型文件加载器、
采集器、环境、标签器、统计或 GPU 分配。调用层负责完整 source、原输入、模型身份、
确定源码、设备、预算和最终产物闭包；本模块所有收据都明确 `source_authenticated=False`
及 `real_N1M_authenticated=False`。默认核 h4096/32×32几何，这不是 N1M 权重认证。
tiny fixture 必须显式 `fixture=True`，不能静默改生产几何。

接口次序为：

1. 构造时接收已加载的冻结 BF16/seed0/image64/action12 Agent；公共 policy copy 必须与
   当前 params 逐值一致，pending 为空，单一共同 train/policy device，无 shardmap。
2. `reset(obs, eval_seed=..., source_id=...)` 要求真正 reset 观测、reward0，并调用真正
   `Agent.init_policy(mode='eval')` 得全零初态。没有接受旧 latent/carry 的参数。
3. `step(obs, source_raw, counter, source_prev_raw=...)` 要求连续 counter 和原 F32 raw12；
   `source_prev_raw` 必须与内部已绑定的上一条 source raw 逐字节相同。首帧 prev raw 为0。
   当前观测及 source raw 真实性仍由调用层认证；这个一致性检查不会识别伪造的来源值。
4. 每步以来源原 EVAL seed/counter 沿模型自己的 h/z 执行原 native posterior policy。
   namespace 为 `DL_NORMAL_FEATURES_U1`，不把新 tag 混入原 `_eval_seeds` 公式，也不改旧
   R4/R5 namespace、oracle 白名单或任何算法源码。
5. 对同一 pre-carry、冻结 RGB 和 root 真的调用公共 `Agent.policy`；先要求 native raw 和
   完整 after-carry byte exact、恰消费一次 eval counter，再还原原 seed/counter。之后才在
   host 上仅将 outgoing carry 的 prev raw 改成 `source_raw`，h/z 不改。返回 native after
   和 history after 两个不同对象与收据，不能把后者称 native 闭环轨迹。
6. 概率在单独无 RNG、无 state 写入的 pure JIT 中计算；q/p 共用当前模型自己的 h，
   p query 不进入 q 历史。采用官方
   `jax.nn.softmax(model.dyn._dist(logits).output.dist.logits, -1)`，包含 unimix .01。
   输出原 BF16 h/logits、实际 F32 q/p 概率、实际 root/latent/actor key；不使用 mode、
   单次 onehot sample 或裸 raw-logit softmax 代替分布特征。
7. `finish()` 对完整 params/optimizer/norm/targets/counters 和 policy copy 核首末 hash；
   每调用还核原 buffer 身份、未被 donation 删除、配置/counters/pending 及内部 carry。
   返回条件历史封口后才能 reset 下一条历史。

## r2：末帧仅提取特征

真实 R5 处理前来源只有 `obs[0..t]` 和 `raw[0..t-1]`；`raw[t]` 属于随机干预，不能作为
当前帧提取后继续绑定的普通来源命令。r1 的 `step` 必需 `source_raw`，因此不足以直接处理
这个边界。r2 增加 `observe_current(obs, counter, source_prev_raw=...)`；不伪造或读取 `raw[t]`，
不改变已有 `step` 的来源命令覆盖语义，也不采用新来源、T、科学判据或执行预算。

状态机为 `idle → reset/active → observe_current/feature-only-final → finish/idle`；
active 可先执行任意已有合法 `step`。current 与 step 共用完全相同的 native q posterior、
官方 q/p query 和公共 oracle 前向；仍核连续 counter、真实首帧标记、已绑定 source prev raw，
并拒物理 terminal/last 帧。current 不接收当前 source raw，返回
`source_raw_available=False, source_raw=None, history_after=None, history_after_sha256=None`，
`next_prev_raw_bound=False, host_source_override_performed=False, feature_only_final=True`。
当前观测、原 key 和前一 source raw 进入历史封口；不将预测的 native raw 写成来源命令。

current 可保存该模型真实 `native_after` 和 hash，内部只保留其副本用于 finish 完整性检查；
它不是有下一来源命令的 history after。此后只允许 `finish`，step、再次 current 或提前 reset
均拒绝。finish 明写 `final_carry_kind='feature_only_native_after'`、
`final_carry_sha256=None`，另报 `final_native_after_sha256`，并标当前末帧没有 source raw、
下一 prev raw 未绑定。每条 history 的 `source_command_frames` 只数真正 step 命令；
总 `steps` 数包含已观测的 feature-only 帧。下一次 reset 将 episode 标记重新置为 active。

真实 tiny BF16 反例在相同 prefix 后，将 current 与提供另一条 source raw 的 step 只比较共同
前向的 h/q/p/logits、keys、native raw/full after，要求逐字节相同；current 不需要最后 raw。
另核错 counter/prevraw/last、末帧后继续推进均拒绝，返回副本篡改不改内部状态，reset-only
current 可 finish、下一 history 标记已清除，完整原 Agent state/policy/counters 始终不变。

本窄接口每步都执行 public oracle；它没有采用未来批量执行时的抽查策略或总预算。
新增输出可能改变独立 native JIT 融合，oracle 逐字节失败必须拒绝，不能放宽精度通过。
源码 hook 仅在进程内 tracing 范围使用并在异常时还原，接口为单线程调用。

有限测试 `tests/test_dl_normal_features.py` 使用真实小 BF16 图像64/action12 Agent、
人工生成两条两步历史。验证公共 raw/full after 在 source override 前一致；仅 source raw
差异改变下一步真实 h；相同 mode 可有不同概率，官方 unimix 不等于裸 softmax；错 reset/
counter/prevraw/内部 carry、公共输出/计数篡改和 NJ state 写入均拒绝；完整原态不变。
测试不认证实际 N1M、来源真实性、任务标签、读出质量、学习、native 性能或方法有效性。

实际生产还需要来源/完整退出认证、source frame/prefix 身份与特征/标签对齐、官方 CPU
全部物理复放标签及外部产物闭包。这些由调用层另行实现；本模块不以几何或 fixture
通过取代它们。实际模型特征执行及 T 仍须相应采用与资源安排。

## r1 有限 CPU 验收（只适用于旧版）

本机新三项测试全部通过，5.565 秒，无 skip、0GPU；使用
`C:/Users/97370/AppData/Local/Codex/dl-validation-env/Scripts/python.exe`，
本次未设置或需要 `DL_DLL_DIRECTORY`。独立源码审查未见改变上述工程含义的阻断。

actual/tested **c84995c5cbf243b7182a1c21f687dc159f695d70**，sv3 fresh clean
`/data/Policy_Discrepancy/dl-normal-features-code-20261015-c84995c`。
新 features 与原 live_policy 两模块共五项全部通过、无 skip：unittest 22.008 秒，
测试进程22.145737461秒，含 clone/checkout/source binding 的完整入口23.219545016秒/300秒；
真实 SSH exit0/23.781秒、0GPU、owned空。248份源码和本规范的实际 working bytes
在测试前后均匹配真实 Git blobs，最终 Git clean；未移动原R5 producer1584或C代码8d63。

证据 `/data/Policy_Discrepancy/runs/dl-normal-features-cpu-20261015-c84995c/attempt01/`，
本机ignored镜像 `analysis/outputs/dl-normal-features-cpu-c84995c/attempt01/`。
`linux-tests.log` SHA256 `07dfb5f438407f51655d2b2713dbbf3cc94fb0fbe6d8e3353b7b605052b697c3`；
`source-binding.json` SHA256 `7f86ab9ff76020f3b0da07e371ff502ae04f3996313d097595cd8f79d92b6ff0`；
`terminal.json` SHA256 `21b79f1c1b40cbdd9897bdd65dadcf490d819e74a1b3151f460b12f35e83264c`。
测试中的原接口回归与真实tiny BF16不是真实来源/N1M/读出/性能验收，不激活T或额外GPU预算。

以上 `c84995c` 证据只认证旧 r1 字节，不覆盖本次 r2 current 修复。新版需新的确定提交、
fresh clean Linux 有限 CPU 验收与对应 source/norm raw Git blob 封口后，才可报告新版工程通过。

r2 本机有限 CPU 补证：新 features 三项与旧 live_policy 两项共五项全过，无 skip，12.135 秒。
真实 tiny 子进程报告共同 current/step 前向一致、无末帧 raw 伪造、末帧后推进拒绝及逐 episode
标记重置，完整原态不变；source/N1M/GPU 标记均 False。没有读取真实 R5、运行真实 N1M、
训练或新增科研裁决；固定提交与 Linux 补验待 root 完成，旧 c84995c 运行目录保持不动。
