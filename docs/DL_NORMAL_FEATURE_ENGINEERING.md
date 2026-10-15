# 正常共同历史特征接口

2026-10-15，`DL-normal-features-r1`。Agent 在完整预验证范围内采用的必要工程接口；
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
