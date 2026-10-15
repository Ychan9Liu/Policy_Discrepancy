# DL-NM1 C：产物与独审接口

2026-10-15，DL-NM1-cal-artifacts-r1。Agent 在完整预验证授权内采用的工程契约，
配合 `DL_NM1_TRIAL.md` 中已采用的 C；不采用 T，不改变科学量、门或预算。
尚无真实 C 结果。本契约须随实际执行提交以真实 Git blob 认证。

## 边界与输入

只执行 128 批只读反事实，不累计学习，不回写 replay。主入口含 import、输入认证、
JIT、计算、写盘与清理共 3600 秒；CPU 全对象独审 900 秒。固定 sv3 物理 GPU5，
UUID `GPU-79704918-e096-92b4-252c-e3244497a8b3`，不渲染、不初始化 EGL。
采用 C 自有 `DL-NM1-compute-resource-r1` 资源认证：CUDA UUID/物理5/PCI_BUS_ID及
JAX_PLATFORMS=cuda，MUJOCO_GL=disable，移除 MUJOCO_EGL_DEVICE_ID/PYOPENGL_PLATFORM。
EGL UUID 为 None、graphics_verified=False/不适用；不要求也不新建 EGL probe。
原 `dl_resources` 的 CUDA/EGL 认证语义不改；C 只复用真实占用查询和 idle/年龄检查，
在导入前核 fresh≤120秒、导入后重新实际查 idle，模型后核真实 compute UUID 及 owned 进程。
R5 权威终态及完整独审或失败封存完成后，重新核空闲才启动；没有加卡或自动延时。

checkpoint 原字节 SHA256 为
`c6a7d92e24669a515bb44693faf73e442d9868f9fc814eec0e22e069aa594d24`，
config 为 `877f8470cb4f7f45d143e3099fbd3f55a828991afe33779b7b04c4264f6c8377`。
原库存 SHA 为 `3339b71ca033d5fc065130751fe163aca05a684b2f14f570ab2018219a3a706d`。
完整 post-load state 为 `4e0d1a0e1de806bf8e4863278cabd171ed0d208dddc1fea18b8b54d68ca6383f`，
params 为 `cffc13c0dcdfbd0c8b0e5c503c9d6251f4b1550c08b85a8742a638655d66d372`，294 叶。
实际 load 后 updates=batches=opt.step=249493、actions=62563、eval_actions=1001、
eval_seed=1289196729。原 checkpoint counters 必须原样保存；post-load batches 按真实
`Agent.load` 从 updates 恢复，不能把保存时 batches 误报成实际值。

replay 可采用显式 `DL-replay-stage-r1` 迁移收据；原库存不改。生产入口同时绑定原库存
和 stage 原字节 hash、源/目标实际主机与目录、1280 NPZ 的源前后/目标前后全量库存。
迁移仅改变物理存放位置，不声称恢复在线 sampler 或原环境；严格 CRC/schema/链认证仍执行。
原主机名必须真实查询，不从候选名称推断。

## 完整产物图

生产者与独审者共同固定精确字段和文件路径，写入各自源码与 fixture；以下语义不能缺失。
所有 run 内文件须在无循环的闭包中逐一绑定 raw SHA256 和 bytes。拒 symlink、额外文件、
缺项、重复 JSON key、非有限 JSON 数字及不完整批；日志/资源/启动观察可放 run 外另行封口。

- 顶层预算、实际配置、完整源码 blob 清单、输入 pin、父/worker PID+start_ticks/PGID、
  原始启动资源认证及后续占用记录、完整计划和初态/末态/cache 收据。
- 128 个 `batch_000` 至 `batch_127`；每批完整 16×65 原输入、全部 65 stepid、实际 prepare
  counter/key、context/consec 收据；16×64 信号；原梯度与真实更新证据；完整 B/N replay 输出；
  reference/production forward 对齐收据；对照权重和 P 随机性证据；每批 complete 最后写入。
- 系数封口、原始与实际 F32 分层边界、完整资格汇总；generation complete 绑定全部固定顶层
  文件及有序 128 批闭包。父 terminal 在 worker 退出和 owned cleanup 后写真实退出码、耗时与
  generation pin；外部 launcher 再绑定父 terminal。失败不得伪造成功闭包。
- 独审输出在 run 外 fresh 目录，先验 raw 完整性及权威退出后，才解码可信数组。

初态保留全部 294 叶的路径/shape/logical dtype/typed hash、完整 state 与 counters；
保留 encoder `enc/` 与 posterior `dyn/obs...` 的实际 F32 原参数。
每批保存这些路径的 B/N 完整参考梯度和原 `_train` scratch 真实新参数 F32 原值；
其它模块保留全叶 hash/有限性/位移收据，不冒称独审重算未存的全状态 Adam。
B/B 重复及 rho0/B 完整 params/carry/replay/共有指标 byte equality 必须由实际结果核验并封口。
所有输入、梯度、参数和 carry/replay 必须有限；只有预声明的诊断不可用比值可以结构化为 None。

NPZ 不允许 object/pickle。BF16 用 uint16 原位 bits，并记录 logical dtype、原 shape 与原 typed hash；
不可把 NumPy 的 void/V2 猜成 BF16。独审通过 bits<<16 解成 F32 仅作有限性/差值计算，
byte identity 比较原 bits。F32 参数/梯度直接保存 F32。数组 schema 与 NPZ 一起进入闭包。

## 原始信号与参考出口

`Counterfactual.reference` 使用独立完整 loss/`nj.grad` JIT；增加显式 reference-only 出口
`outs['nm1_reference']={K,E,rep_before,weight}`。K 是同一 loss 内 RSSM 的原 `probe.rep_raw`，
E 是该实际 F32 K 上 `max(K-1,0)`；rep_before 是覆盖前原 free-nats rep，weight 是实际该参考权重。
B 为 ones，N 为实际 overlay 权重。禁止从加权/取 floor 后 loss 反推 raw K。
`Agent.loss(...,dl_reference=False)` 默认分支不额外构图或输出；原生产 `_train` 不改。
该参考出口可能影响独立参考 JIT 融合，梯度不称生产内部逐叶 bit-gradient。

数学模块的每批 signals 字典固定为：`K,E,D,e,f,v_N` 为有限 F32[16,64]，
`m,A` 为 bool[16,64]，`stepid` 为 uint8[16,64,20]，`chain` 为 int32[16,64]。
chain 按原库存有序 chains 映射（0..15），独审从原 UUID/chain 图恢复。
K/e/f/D 使用同一个 N 参考 forward；m/A 与真实裁 context 后 flags 对齐。
不能 CPU 重算 KL/exp 冒充原 GPU 数值，CPU 只核实际原值的规定算术。

## 系数和资格的共享调用接口

生产纯 NumPy 模块 `analysis.dl_nm1_calibration` 提供：

`freeze(signals, expected_batches=128)` 返回 JSON 可写对象：`version,batches,positions,
aggregate64,S,W,bins`。S/W 含 `calibrated,reason,coefficient_f64,coefficient_f32,denominator_f64`；
不可用系数为 None。bins 含 `calibrated,reason,eligible_visits,D_quantiles_f64,K_quantiles_f64,
D_f32,K_f32`；无活跃位置时为空边界、calibrated=False，不捏造分层成功。
实际边界为 F64 linear quantile 后 cast F32、再次 unique 的值；P 在 GPU F32 以 side=right 使用。

`summarize(signals,effects,controls,frozen,expected_batches=128)` 返回可写资格及测量。
effects 每批有 `encoder_gradient_changed_elements,encoder_step_changed_elements,
posterior_gradient_changed_elements,posterior_step_changed_elements` 四个非负 int，
从保存的原始张量数值差恢复，不能用 hash 或 signed zero 代替。
controls 每批固定键 `v_S,v_W,v_P`：有效项是实际 GPU F32[16,64]，未校准项为 None。
未校准 S/W 或无活跃分层 P 时，NPZ 不存该权重，record 明写 None 和原因；不是文件缺损。
若 bins 无效，不调用 P 随机/置换生产操作，但仍可保存固定 key/priorities fixture 证据。
资格阴性应得到完整 `audit_integrity_passed=True,scientific_qualification=False` 的可能结果，
不通过 engineering raise 隐藏不利科研结果。

freeze 的 F64 乘积/fsum、实际 F32 权重释放、不同 stepid/链覆盖、按批同模块 gradient/step
非零、pooled P 效力和 S/W/P 释放门沿科学规范。独审不得 import 此生产裁决模块，
必须独立实现并比较全部标量及资格；无 denominator 或零 N 释放时比值 None、资格 False。
fixture 可显式改变 expected_batches，仅作数学测试，生产入口只能用 128。

P 必须存真实 key(uint32[2])、priorities(F32[1024])、perm(int32[16,64])、group/counts 和完整 v_P。
运行断言 JAX0.4.33、threefry2x32、threefry_partitionable=False、x64=False，不修改设置求通过。
原 key root0/tag0x444C/opt.step249493 在全部 C 批相同；实际 P 调原 conditional_shuffle。
独审用独立 NumPy Threefry2x32/fold_in/uniform 与 stable group 排序重建实际 perm，
先由真实 CPU JAX fixture 核实现；不称重执行原 GPU NN。

## 原态与独审范围

共同 Agent 全部 params 的原 buffer 身份、未被 donation 删除、实际 counters/config/pending
每调用核验，完整 state 在首末核；fresh scratch 才 donation。cache 首末全 hash，相同原始文件
再末遍全 hash；不要求每个 scratch 重扫 30GiB，不允许手清 pending 或合成写回。
初始资源认证在慢 replay 导入前检查；父每 15 秒写真实占用后再 guard，导入后重新真实核空闲，
不得伪改原资源记录时间。仅清理本次记录身份的 owned 进程。

独审在完整字节闭包和已知 checkpoint SHA 通过后，可用仅 NumPy 类型白名单的 restricted
unpickler 解析此可信自有原 checkpoint，核 294 叶与原始/实际 counters、初态原值及 full hash；
拒 JAX/model/任意类，不为载入成功扩大白名单。此为数据解码，不是 NN/Adam 重执行。
独审恢复精确 PCG64 计划、999984 合法起点次序、全65跨度/context、seed/m/A、P 与原张量差值。
没有原始全张量的 B/B/rho0 全状态 identity 属认证生产收据，局限必须单列；不能写成独立 NN 验收。
