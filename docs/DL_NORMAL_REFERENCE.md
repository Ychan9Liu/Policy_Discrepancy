# DL-NM1 正常参考来源

版本DL-NM1-normal-reference-r1，2026-10-15。**Agent在完整预验证授权内采用的条件来源规范**，
尚未实现或运行；不采用NM1学习预算、组数或方法有效结论，不改R5的任何字段/7200秒/旧失败。
只在R5全对象未完整或未通过独审时启用本独立来源；条件仅依赖完整性，不依赖R5科研结果正负。
完整独审R5若可用，优先复用其处理前真实历史；不从失败R5 partial拼接本对象或将其升级为完整来源。

## 必要性及明确不同的对象

现有f2d24da完整生产来源来自300k模型，可作N1M探索性off-policy参考，但不是N1M正常访问分布。
N1M原R4失败取证没有全程最终模型state seal，不能直接称完整认证来源。
本对象用同一冻结N1M正常posterior policy采集新reference，不含branch、H1/shadow、MC或segmentation。
一family一条policy流、14时刻，与R5每unit独立policy流/一个随机时刻不同，不能替代R5效用统计或翻案旧失败。

## 固定输入与采集

固定N1M checkpoint/config原始SHA分别为
`c6a7d92e24669a515bb44693faf73e442d9868f9fc814eec0e22e069aa594d24`和
`877f8470cb4f7f45d143e3099fbd3f55a828991afe33779b7b04c4264f6c8377`。
任务dmc_quadruped_walk、size50m/BF16、vision64/camera2/proprioFalse/repeat1，模型root seed0，
原完整params/optimizer/norm/targets/counters只读。加载后必须与完整原state一致，policy副本同步并封口。
采集合法推进eval counter/seed，结束后完整还原并核最终fullstate和policy副本，不默认清moment或改变训练state。

一次Agent load/JIT，串行32个fresh官方suite/production wrapper/Env/EGL，各自reset/close：
family f0..31，env_seed=24576+f，eval_seed=0x4E4D3100+f，实际公共eval policy连续counter0..999。
root沿原公开_eval_seeds(configseed0,EVAL_TAG,eval_seed,t)，不另造抽样或混入family/arm新key。
每条完整1000实际动作、1001帧，固定14个普通t=64,128,...,896；fit family0..7=112状态，held8..31=336状态。
不替换early terminal/失败family，不为覆盖或收益增添seed/位置；每episode为一个统计单位。

存真实RGB/obs F32 reward/flags、suite F64 reward/discount、raw F32、actual applied F64、完整integration/task RNG/clock/reset/wrapper done snapshot，
以及静态完整MJB/camera身份和原始字节SHA。reset无奖励明确编码0及来源；终点raw/applied零占位action_executed=False，不映射/执行。
在0、全部14个t及999保存实际public key/counter、pre-policy carry/raw/after-carry收据，
在冻结RGB上按原key重演公共policy oracle、恢复eval计数；不能独立重渲染并要求另一图像作输入。
一次oracle通过不证明renderer逐位复现，RGB来源和完整state只读各有独立范围。

## 完整性、资源及退出

所有图/原始SHA/来源/config/模型与完整state seal、32/32 family、资源和权威exit0完整闭合才可用。
先认证文件完整图、路径、原始hash和终态，再解可信项目pickle；不以complete字段自行证明成功。
来源校验需真实Git blob绑定、clean确定SHA、固定原输入前后SHA、实际fullstate/policycopy不变及合法counter恢复。
GPU仅sv3物理4，在原R5终态并确认释放后重新核空闲/独占与CUDA/EGL同卡；不预占5–7或与原R5共享。
GPU父入口固定1800秒，包含import/load/JIT/32个Env/所有I/O、seal及清理；这不是完成ETA。
超时仅终止本次owned组、保留失败、新attempt，不延预算、补seed或拼接旧数据。

## 独立CPU全部物理与标签审计

固定900秒/0GPU，官方suite按每family声明seed真正reset；核静态完整MJB/camera及state0精确同source。
逐stored-applied控制重演全部32×1000步，独立核raw→actual production映射、F64/F32 reward/discount/flags、
完整integration/task RNG/clock/reset及expected wrapper done字节；不通过任意snapshot restore引入旧1e−8容差。
CPU不渲染、不调用模型，不把stored-control审计称完整闭环NN/render重演或独立执行FromDM。

到14个t沿当前已复放CPU状态，直接只读查询torso_velocity()[0]和torso_upright()；
核实际官方Move/Physics、desired_speed=.5、official reward component product及当前F64来源reward。
query前后完整snapshot/integration/task RNG字节不变；query自身无step/control/reset/render。
保存448条原F64标签及source frame/snapshot/完整prefix身份，不生成critical/应保护标签。
变异不足或速度奖励饱和时报告覆盖限制，不换标签、clip、挑有利位置或重新称blind。

此来源通过只支持后续共同真实历史读出的输入条件；完整学习干预/native及S/P/W资格仍待独立冻结和真实证据。
