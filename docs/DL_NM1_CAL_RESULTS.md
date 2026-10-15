# DL-NM1 C 实际执行与资源记录

2026-10-15，版本DL-NM1-cal-r1，状态：**首次尝试工程失败，未完整生成或独审**。
这是Agent在既有完整预验证授权内采用的128批cal-only，不是累计学习或正式实验；
后续T4096/整阶段6GPUh仍为建议，尚未采用。

实际运行/有限CPU测试Git为`8d63fff8b15a49c7e2e7bc84fd0f9c3481e6acb0`，
clean代码`/data/Policy_Discrepancy/dl-nm1-cal-cpu-code-20261015-8d63fff`，
本次报告文档不移动运行源码。该版本142项clean Linux有限CPU测试通过，见[工程证据](DL_NM1_ENGINEERING.md)。
执行规范[DL_NM1_TRIAL.md](DL_NM1_TRIAL.md)、产物/独审契约[DL_NM1_CAL_ARTIFACTS.md](DL_NM1_CAL_ARTIFACTS.md)保持原字节。

## 实际输入与固定配置

- 原N1M seed0/size50m/BF16，仅视觉64×64；checkpoint为原249493更新初态，不新增训练seed。
- 原checkpoint/config完整SHA及原state身份沿[已认证输入](DL_NM1_ENGINEERING.md)，不采用新模型。
- 原1280NPZ、16源链、1M动作的私有replay，完整迁移见[迁移记录](DL_REPLAY_STAGE.md)；
  stage SHA `00063544c10facde529f82fac07e759a2d44306aedd77924f30ff185e51519f8`。
- 固定128独立抽样批，每批16×65 context、16×64 loss；PCG64计划root20261101/tag0x4E4D4341。
  每批从相同完整初态执行B/B重复/rho0/N的真实scratch更新及B/N独立完整参考梯度；丢弃返回更新，
  committed optimizer/replay updates应均为0，无累计4096学习。
- normal-only A沿原H1m并排除current is_first；原raw K、e与f定义、alpha20/rho.5/free-nats1、
  原完整loss/梯度开关不改；S/W/P资格和128位置/8链/16批作用、1%/10%/50%门沿预先采用规范。
  不新增constant c或正式评价计划；S/W系数须依128完整真实张量计算后独审，不填默认数。

## 前驱完整审计及真实封存

原R5 producer1584完整512/32；成功physics auditor为
`277c18001d353d9101a3622959351d3994699adb`，actual exit0/655.220153340秒/1800秒。
report SHA `144c4490e69283ab14ad1a96fd8cbd9c47945875797a91ac6d420dade9e216cd`，
外层terminal SHA `d6164a94bdb7b9a95579454d539643d705f808d798d8f44043dad3d0101c8614`。
科学主检验未支持且sham concern，见[结果](DL_R5_R2_RESULTS.md)；C不以该科学阳性为前提，
也不因此取得保护或方法性能支持。

封存helper SHA `6436ce13847f7da5aa96530bc6a03c5101056829807ad3320f255749a6b9736c`，
两遍完整库存和两次原parent/worker/shadow/CPU auditor退役核验；不解码NPZ/pickle或读科学字段。
server完整15.583822330秒/900秒，真实SSHexit0/16.172秒，0GPU；execution success=true/failure=null。
最终retirement SHA `e8b76009e759c1d22d6770498a121abb7b7c0f22c469b87f2503fc60db168595`；
只有确认真实退出和完整execution收据后才把final SHA传C，不用pending自证退出。
本机收据`analysis/outputs/dl-nm1-cal-preflight-8d63fff/retirement-277c180/`。

## 资源、一次启动与最新快照

固定sv3/物理GPU5，NVIDIA A100-SXM4-80GB，UUID
`GPU-79704918-e096-92b4-252c-e3244497a8b3`；一张独占卡，授权4–7不等于四卡并占。
CUDA_VISIBLE_DEVICES设该UUID、CUDA_DEVICE_ORDER=PCI_BUS_ID、JAX_PLATFORMS=cuda；
无环境/渲染，MUJOCO_GL=disable，MUJOCO_EGL_DEVICE_ID/PYOPENGL_PLATFORM均移除。
原CUDA/EGL成对接口不用于此compute-only阶段，实际CUDA绑定仍由C收据检查。

22:28:20–24启动前完整XML/pmon，物理4–7均无进程/各0MiB，原C checkout仍8d63且clean。
冻结external launcher版本DL-NM1-launcher-r1，SHA
`53edf01e00a99bfe1c56ad99b7376e37c9e678eeded0ec0a9e7df6709536a30b`。
22:28:41.740151一次启动parent467533/PGID=SID467533/start409689202，outer467517；
全入口硬限3600秒，outer3610秒仅用于失败捕获/清理，不增加成功预算。独立CPU全入口另900秒。
launch SHA `71df2eba2e729c2e9b7d1d15bbfa660b04a7b2099adbfb6637ad34a6591dd740`。

运行目录`/data/Policy_Discrepancy/runs/dl-nm1-cal-20261015-8d63fff/attempt01/`，
外部权威/日志`/data/Policy_Discrepancy/runs/dl-nm1-cal-inputs-20261015-8d63fff/`，
本机观测`analysis/outputs/dl-nm1-cal-8d63fff/`，现有handle94939；不因观察返回session而重启。
22:31:13–17原parent仍活跃、正在CPU导入/核验，物理4–7无可见GPU进程/各0MiB；
GPU5仍为本入口保留，不能把未初始化CUDA的间隔当作任务释放。4/6/7未由本次C使用。
资源快照`analysis/outputs/dl-nm1-cal-preflight-8d63fff/resource-20261015T143129909781/`。

当前C峰值显存、实际吞吐、完整资格与系数均未测定/验收；旧诊断约2.53GiB不构成较小卡兼容保证。
已完成R5实际2.580220占卡小时；C新增上限1占卡小时，另CPU封存/独审不占GPU。
T的6GPUh只是未采用的整阶段建议，不是每臂6h，不是成功ETA或全部研究剩余预算。

## 当前验收边界与下一步

原8d63有限工程测试和前驱完整性已通过；真实C首次尝试失败，尚未证明真实N1M干预有效、对照匹配或方法有效。
当前chat保留首次失败、完成下述修复和确定提交Linux验收后，在相同物理5/原3600秒/原科学配置下
使用新的attempt标识从头重跑；不恢复或补齐原partial。完整外层真实退出/终态/清理和完整128产物后才启动900秒CPU独审；
不可仅凭parent returncode0或中间部分批次宣布通过。失败保留新attempt证据，不改门或延时兜底。
独审分别报告生产真实位移与独立参考梯度，不能把后者叫生产内部逐叶bit-gradient；
只验证参考S/W/P权重释放，不冒称各臂真实学习位移。T采用前须结合C实测与完整source/native工作量
核算整体预算，并满足[T解释及对照资格条件](DL_NM1_T_READINESS.md)。goal保持active。

## 22:34:08首次尝试工程失败及必要修复

原8d63/attempt01外层actual SSH exit1/328.578秒；实际parent exit1/326.670301574秒，
external_failure=null表示监督器没有额外故障，不表示worker成功。supervisor worker_returncode=1、
failure=`C worker failed/incomplete`，generation_complete不存在，原parent与owned组已结束/清理。
22:35:06–11 fresh完整XML/pmon：物理4–7均无进程/0MiB，GPU5已释放。
旧handle94939现已终态，不再对它续跑或重复启动。

worker已完成replay导入和计划生成、初始化真实size50m CUDA Agent；在第一个B的postload完整身份门失败，
尚未执行init_train/首批collect_batch或128校准，不能将此工程失败写成方法阴性。
两个独立源码审查与纯CPU最小重现确认一个充分的表示错误：
官方`Agent.save()`已device_get所有params并将五个counters显式转为Python int；
`host(agent.save())`再次将这些int变为零维ndarray。`tree_digest`有意区分两种类型，
所以数值完全相等、参数子树及parameter_digest完全相同仍会破坏完整typed hash。
同样错误也在末尾原态fence，且零维counter随后不能直接写JSON。

必要修复标记`DL-NM1-state-serialization-r1`：仅在两处完整save身份边界保留官方树及原标量类型，
共享精确身份核验；不改FULL_STATE/PARAMS/COUNTERS/STEP、共享host/tree_digest、数据、NN、配置或科学门。
旧aggregate检查在第一个条件短路，真实参数哈希等剩余条件尚未被该失败证据证明；
修复后须仍逐项严格认证，若再失败则另存证据，不能以已定位这一错误声称其余身份都已通过。
旧142项有限测试未走真实load_variants入口，新增回归必须覆盖官方tiny save/load身份及拒绝counter类型/值、参数修改。
修复本机测试、确定提交clean Linux补验及新的实际版本/启动尚待完成。

原失败完整目录保留；小镜像`analysis/outputs/dl-nm1-cal-8d63fff/failed-attempt01/`。
outer terminal20556字节，SHA `f98157c6247fd7c31d8b13c425209de83b8259fbc848d38781873f8fee33f49f`；
supervisor SHA `4a3cbc689648938b00b81742ede3cb7902130a9c1b8d6984ad85ffa428f8daca`；
worker.log SHA `a6b47a27d7d747aaf3ca473f5c782efaff2b7bb14414f701f7b41ca36b0fd090`；
main.log SHA `0bf6745dc082d88025599451631c156909939b85fbf92c40758401cd6e89fcec`。
实际runtime_before未产生，不能冒称其完整CUDA运行后核验已通过；真实模型初始化日志仅证明到达该加载路径。
