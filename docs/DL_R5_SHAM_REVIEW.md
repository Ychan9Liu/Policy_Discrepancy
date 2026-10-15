# DL R5 sham拒绝：只读原因审查

2026-10-15，`DL-R5-sham-review-r1`。本页是Agent内部只读审查，不改变原结果、门、数据或预算，
不采用新的采集/学习。原执行Git `1584ef31c322fc3371ea9ce220c06fe7e654b23a`，
成功完整物理/证据独审Git `277c18001d353d9101a3622959351d3994699adb`。
已冻结highD主检验未支持、sham p=0.0045995400459954、implementation_concern=true；
原结果及raw pins见[实际结果](DL_R5_R2_RESULTS.md)。本次没有重拟合、删除family或重算另一主判据。

## 已验证事实

`scripts/dl_live_policy.py::LivePolicy.first`将同一unit/bundle的qqA/qqB都映射到同一个`qq`路径；
底层first是纯host组装，选择相同q_after/q_raw和该unit/counter的root/latent/actor keys，不新增NN/nj.seed。
不同unit本来有不同key，不能将此写成所有qq单元共用一个key。
`scripts/dl_live_family.py::run_unit`先完成current/derive/H1 shadow/public oracle及pretreatment封口，再reveal label；
label不进入policy root公式，当前D/e/K/r_t不是在本unit label后选择。
之后H100统一normal q，每unit新initial carry与fresh官方Env/reset，不从上unit carry起步。
公共oracle临时修改eval seed/counter后finally还原，params/optimizer/norm/policycopy/counters有完整fence，
只读ninjax路径拒绝write/create；未发现QQ别名直接影响下一policy/seed/Agent状态的确定源码通道。

每family确实复用Agent/JAX进程、编译/allocator/driver上下文、若干issued/sealed/oracle/accessed计数表、
CPU shadow进程和已收集结果列表。显式元数据用于认证/计数/落盘，未发现其反馈进入下一prefix/seed的读路径。
这些状态并未全部作为GPU/render运行时状态受到逐位fence；因此源码检查不能证明完整运行时无干扰。

另有确定的关闭语义缺口：`scripts/dl_production_source.py::ProductionEnvironment.close`只调用`self.env.close()`；
生产Wrapper的`__getattr__`转发到DMC，而`embodied/envs/dmc.py::DMC`未重载close，
最终是`embodied/core/base.py::Env.close`的pass。adapter.close不能证明suite/physics/EGL被显式关闭。
这不证明实际泄漏、跨unit影响或sham原因；GC及上下文析构的实际行为本轮没有重执行验证。
本轮不修改既有执行source，不以此覆盖失败、回溯修正数据或宣称旧RGB无效。

## 检验的确切范围与不能推出的结论

预定sham检验在固定held highD、固定QQ membership下，独立重抽A/B符号10000次，seed20261031，
双侧absolute统计、加一p值；23A/13B不要求重新抽样时保持计数。
原假设是这些单位QQA/B sharp equality及别名互换不会影响其它outcome/选择mask。
这个随机化诊断不要求误差正态或将同family的单位看成独立测量误差。
不得改为family-sign、计数固定置换或新阈值来消除已见拒绝。

仅“其它pq/pp臂可能改变后续硬件历史”不足以解释该条件QQ检验失效，因为重抽A/B时这些臂和QQ membership固定。
若要提出标签相关干扰，需要证据表明QQA/B别名本身改变实际运行、其它outcome/后续mask，或独立分配假设失效。
与A/B独立的渲染波动或共享噪声只贡献方差，不能单独当作这一通道的证据。
一次预定检验仍可能偶然拒绝；现有512及源码检查不能区分偶然拒绝与未观测的标签相关运行时影响。
因此**根因未知**，不将implementation concern解释为已证bug、已证GL因果或已证无害。

## 建议，未采用

若后续确需定位，先做有限关闭/释放工程验证和同bundle两QQ别名组装身份核验；这只验证接口。
再另行事前固定有限native reset→prefix→H100诊断，在X/key先封口后独立随机化QQ别名，
交叉共享worker与fresh独立worker，逐步记录RGB→NN→raw→physics首差及真实资源/clock。
不按已见异常单元挑样、不自适应扩样；所需规模、功效、资源和whole预算尚未采用。
可重复的随机化alias×context效应只能支持新条件下的共享通道，不唯一识别GL，也不能唯一追溯历史根因。
fresh化后p不显著不足以证明修复，不能据此翻案原结果。

当前C是另一独立的普通压缩干预/对照校准，不需要把历史sham先解释为无害；C通过也不能倒推sham无问题。
完整原R5的处理前历史可作固定共同reference，不能称fresh blind或策略效用真值。
下一先完成真实C和正常标签/特征闭包；若要依赖R5效用作因果真值，则上述concern必须保持为限制。
