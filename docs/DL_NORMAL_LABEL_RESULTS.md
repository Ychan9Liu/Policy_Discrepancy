# DL 正常来源当前任务标签：真实完整CPU验收

2026-10-15，版本`DL-normal-labels-r1`，**完整512条验收完成**。
这是Agent在完整预验证授权内采用的固定来源标签准备，不是机制、信号或模型效果阳性。
实际/测试Git为`47db6265311bc04dbdcc79439842eb8b6e707097`，运行在sv3，0GPU，
clean代码`/data/Policy_Discrepancy/dl-nm1-repair-code-20261015-47db626`；
必要工程与此前本机/新174项Linux有限验收见[工程记录](DL_NORMAL_LABEL_ENGINEERING.md)。

## 固定对象与真实执行

原完整R5 producer仍`1584ef31c322fc3371ea9ce220c06fe7e654b23a`，原run：
`/data/Policy_Discrepancy/runs/dl-r5-live-r2-20261015-1584ef3/attempt01/`。
成功prior physics audit为`277c18001d353d9101a3622959351d3994699adb`，
原producer launch/terminal及prior report/terminal/launch共五份原始pins强制显式输入，
完整前后原run库存、当前代码和prior五份复用依赖均认证；没有依据R5科学正负换来源。

固定32family×16unit=512个原有序source ID；前8family/128条fit，后24/384条held。
512次fresh官方suite/reset，沿记录的raw[:t]独立映射为实际applied动作并逐步官方复放，
共247872个prefix步骤，全部integration/task RNG/clock/reset状态和F64奖励/F32观测编码exact匹配。
仅在当前t查询torso-local velocity_x（主任务量）、upright及奖励分量，不restore/额外forward/render。
每行绑定当前obs[t]、snapshot[t]与reward[t−1]；不读取/伪造正常raw[t]。
同实例query前后snapshot、task RNG和静态MJB不变，两种snapshot hash协议分别保存，不能混比。

22:57:35.934235启动实际labels parent497085/PGID=SID497085/start409862629；
23:10:53.734796外部controller已写成功终态，内入口797.244413620秒，
controller整入口至末验证/清理797.825030724秒，实际root SSH observer exit0/798.329秒，均小于整体900秒。
parent已gone、owned组清理完成、external_failure=null，完整512输出已发布。
该900包含source/index、所有prefix、末遍完整认证及落盘，未叠加第二份source budget、未延时或拼partial。
model/render/显式restore/显式forward调用均0，不使用D/e/Y生成或筛选标签。

实际controller `DL-normal-labels-launcher-r1` rawSHA
`a0d66f8233d2493fa65ea8029cf3145eac1596769595b37503ff38a23415e732`，
其中只增加了标签工程规范绑定，采用前已静态审查；没有GPU或新的模型状态。

## 原始证据与退出层次

实际结果目录`/data/Policy_Discrepancy/runs/dl-normal-labels-20261015-47db626/attempt01/`，
本机完整小镜像`analysis/outputs/dl-normal-labels-47db626/attempt01/`。

| 文件 | bytes | 原始SHA256 |
| --- | ---: | --- |
| labels.json | 1539648 | `5db0529317c8c9c55674aee80df1ef0258c6bcf4e407ce1b4c74373ec0f46890` |
| terminal.json | 49321 | `cc0ec4728020290693f866b429de06a47eff3fcd917ba044cad310fee82c4361` |
| launch.json | 43141 | `332d10eda5f8ae5361a67baba4fbc1025d1c18cac5bc3595d79ba0c49e583a22` |
| labels.log | 273 | `276237942bda98424265f1861e35dcc92acf3e188f97513fb7934012a299ea79` |

terminal的returncode/parent_identity是被controller监督的真实labels脚本，不自证controller自身最终退出。
root先核实际SSH exit0及完整耗时，才接纳该输出；后续feature入口还须绑定这个外层事实。
已把实际observer收据、stdout和工具源码原字节复制到独立inputs目录，均再次核rawSHA，
不修改已发布labels/原R5图：

- `/data/Policy_Discrepancy/runs/dl-normal-labels-inputs-20261015-47db626/outer-execution.json`，
  588B，SHA `0e443503674477bbc8a2331380498ad589fdf69e18fc0d5f206620ffe27a18ab`；
- 同目录outer-execution.log，323B，SHA `2445c7f95789cc8493fa7d41bb29a4108663b72738b2dcfac6f09d81b06a8908`；
- 同目录outer-observer.py，1218B，SHA `6cdc35b08b3175fe20d1575ea737597956e7c701a833e740ee555698a154bd2f`。

## 验收范围与下一步

本次是原完整来源的真实当前任务量标签验收，超出仅generated64步/mock512图的有限工程范围；
没有重新执行原RGB renderer、NN或随机subkey，不证明RGB逐位可重渲染。
torso-local速度不是世界坐标速度，也不是动作正确性/应保护的真值；upright仍是辅助。
原512已经公开，固定8fit/24held是来源内部隔离，不是整个研究的fresh blind或512个独立训练seed。
没有拟合读出或证明q信息增量、压缩损伤、gate选择性、保护或native性能。

下一当前chat完成真实C及独审，同时审查正常reference0特征入口；
该入口必须消费本成功labels及其实际外层执行pin，逐source/当前时刻对齐并完成真实N1M公共oracle。
其GPU whole预算和实际设备需事前采用，不能自动沿用未采用的T6h；C仍按原GPU5/3600秒运行。
后续固定读出按[已采用数学定义](DL_NORMAL_READOUT.md)，不得根据已开标签选择特征族/主标签/阈值。

第二Agent独立本机复核全部rawpins、259份实际source_files的47db Git blobs、3份当前模块和5份prior依赖，
原audit inventory canonical SHA、512条trajectory pin/source ID/固定PCG64计划及row-binding digest一致。
3072个F64 hex与JSON逐字节吻合，奖励乘积/F32编码、query前后和RNG前后不变及两种hash格式全部闭合。
这是对已执行收据和本机镜像/真实Gitblob的独立复算，不是该Agent又重扫服务器或第二次复演247872物理步。
