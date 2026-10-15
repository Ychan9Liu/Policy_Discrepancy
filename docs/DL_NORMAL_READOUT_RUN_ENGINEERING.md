# 正常 reference0 读出：只读 CPU 工程接口

2026-10-16 当前索引，`DL-normal-readout-run-r2`。本文件说明实现与有限工程证据；不新增研究判据、
训练组、T 或真实读出执行授权。数学定义仍以 [DL_NORMAL_READOUT.md](DL_NORMAL_READOUT.md) 为准。

## 当前状态：F1失败，F2未启动，reader r2实际Linux13通过

**实际/测试提交** `6654f40bcf927a5eb47e7ce8523a48e88d8c17aa` 已完成sv3 fresh clean Linux一次整入口验收：
reader9+math4=13全过、无skip，七个必需用例在真实日志逐项ok；unittest14.664秒、tests14.851875407秒，
controller16.049491666秒/whole300，root实际SSHexit0/17.156秒/external_failure=null，非320秒成功扩限。
258源码+9规范的267个不同文件逐实际Gitblob初末一致，owned双registry空，GPU0。
root实际SCP完整10文件及两个Agent的本地镜像/Git复核通过，未重跑或读取真实feature/labels/NN。
这是mock provenance、BF16 bits及NumPy数学/接线的有限工程证据，不是真实F2或方法验收。

实际代码 `/data/Policy_Discrepancy/dl-normal-readout-code-20261016-6654f40`；
CPU run `/data/Policy_Discrepancy/runs/dl-normal-readout-cpu-20261016-6654f40/attempt01`；
本机 `analysis/outputs/dl-normal-readout-cpu-6654f40/`，包含实际部署/SSH和完整attempt01镜像。
root SSH receipt SHA `b92a149f357904353ea7c1e6783db4a5f1134d89ff5b8f311f17dfb30d812282`；
terminal2241B SHA `9599e6fc727779db576bd1fe350f2b915386cb5e0eea1750b597b25a9c64a1b3`；
tests log2694B SHA `1499f14175adac7a4638f83189a46d4afde6be497cff1aea073331cde902ddba`；
source-binding84946B SHA `989f37a7a046d264bec8054b9833fac6c60381d0b08c71197aeaf8d62a8f4540`。
launcher11788B SHA `70f0c8984a62be3739163b3a9062edd4e3a0062bf81e8bce76238d4859d9688f`；
bundle SHA `e5f14477259c6a3fbcaf9708c7fe317e53ca9e235abb058c99c7149a521df84a`，实际mkdir/SCP/远端readback均exit0。

01:48:19资源快照GPU5–7各9未知作业、GPU4空；F2/GPU6冲突，仍未上传/启动，等待用户调整独占资源。
C4已因物理5新未知作业失败，非科学裁决；新readout必须等待自己的完整F2成功，不能接失败C4或F1。

对应consumer6654/F2/GPU6的新CPU900外层已在本机准备并经独立只读peer，无确定接线阻断：
`analysis/outputs/dl-normal-reference0-readout-controller-r2/`，launcher43274B SHA
`359874d3e037d246057778f469c15c37fd1a2bab1d0393c19ff5fe151b92953c`，manifest SHA
`324335e0e4cd266580d6e362884de7907743635e85ea86df1ad6741d0cea09bd`。
原25generic AST、3f helper原bytes、8CLI/900/0GPU保留；264项外层来源覆盖reader260，五复用Gitblob仍等2a。
16个生成反例及正负数值闭包通过，均未fit/main/读取真实特征；准备不是实际读出或方法验收。
新inputs计划 `/data/Policy_Discrepancy/runs/dl-normal-reference0-readout-inputs-20261016-6654f40`，
新out `/data/Policy_Discrepancy/runs/dl-normal-reference0-readout-20261016-6654f40/attempt02`。
尚未stage/执行；未来F2完整成功terminal/root退出两个真实SHA仍mandatory runtime参数、当前未知不预造。
root还须保存读出自身实际SSH0及全入口≤900收据，内部controller terminal不能自证未来完整退出。

原F1物理GPU4在325/512条后因资源guard发现未知574111的C+G占用而失败，真实SSH退出1、
3553.406秒，小于12600秒；不是超时或feature科学阴性。无完整generation，325条partial不能
拼接、补齐或作为读出输入。未知进程身份和来源未确定，不归因、不干扰。完整失败保存与raw pins见
[DL_NORMAL_REFERENCE0_EXECUTION.md](DL_NORMAL_REFERENCE0_EXECUTION.md)。

已采用同一producer2a、同512/32family、128fit/384held及12600秒完整预算的fresh F2 attempt02，
仅在用户已授权sv3物理4–7内改到物理6，UUID
`GPU-70db6c66-9917-e0e2-44db-0fc80fcf41b9`；最多两卡、每卡一任务，C/probe5、F6。
F2外部控制器已在本机准备及只读复核，尚未上传或启动；实际启动仍须fresh资源核查。

reader r2仅改五项固定绑定：版本、F2 run、F2 inputs、物理6和UUID；producer2a/CODE、
五份复用依赖、全部认证函数和数学均不变。本机reader9+math4共13项全过、无skip，107.364秒，
仅生成/mock provenance与NumPy；不是实际feature/labels读出。入口raw SHA256
`8250e79feb17c788626edf7b66a0febadcf79a99343ae25a087ceb9923c5cd53`，test
`9030890aa0dcaead92e0f24111f930270feebd67cdb744e1aeae708c66e1de74`。
确定SHA的clean Linux验收已完成如上；不得把下述旧dfd验收记成r2验收。
旧dfd reader和已部署的e238读出控制器绑定F1/GPU4，**不得执行或复用于F2**。
新reader尚未执行真实读出，F2未来成功终态/真实外层退出pins尚未产生。

## 固定对象与命令

`analysis/dl_normal_readout_run.py` 只接受原始 reference0：producer actual/tested
`2a15c91aad16c0e3151fde3943e356dad1fcb69d`、sv3 物理6、完整512/32family、12600秒。
后续 B/N endpoints 需要各自完整来源协议，本入口不冒充通用 endpoint 认证器。
当前分析提交通过 `--expected-git` 单独记录，不能写成提取时实际提交。

所有参数必填，无默认 CPU 预算：

```text
python analysis/dl_normal_readout_run.py
  --features-run /data/Policy_Discrepancy/runs/dl-normal-reference0-features-20261016-2a15c91/attempt02
  --feature-controller-terminal <真实外部controller terminal路径>
  --feature-controller-sha <该文件完整rawSHA256>
  --feature-outer-execution <root真实SSH退出收据路径>
  --feature-outer-sha <该文件完整rawSHA256>
  --expected-git <当前分析完整Git SHA>
  --seconds 900
  --output <所有输入及代码目录之外的fresh目录>
```

这些F2真实特征终态及 SSH 退出收据尚未产生，本文件不填假 pin。既有完整CPU入口900秒采用
保持；新reader提交/Linux验收已完成，外层须采用相应新绑定及真实F2双pins。入口计时包括 import、认证、Git/raw读取、NumPy 拟合、写盘；不得延时、补 partial、
搜索超参或从输出符号调整来源。成功还需要实际外层退出0及完整 wall 在显式预算内；
`readout.json`/stdout 自身不能证明进程未来成功退出。

## 外部权威与完整图

先验证强制传入的两个 raw pins。controller terminal 必须为
`DL-normal-reference0-launcher-r1`，原 parent 实际0、success、gone/owned-clean、无错误、完整
reference0 scope，固定物理6/上述UUID/12600。root 实际 SSH 收据采用的字段契约为：

- `git_sha=2a15c91...`、`physical_gpu=6`、整数 `returncode=0`；
- `0 < elapsed_seconds <= 12600`、`external_failure=None`、
  `actual_outer_complete_within12600=True`；
- 精确 command 为 `ssh -o BatchMode=yes -o ConnectTimeout=10 sv3` 后接项目 Python 与
  `/data/Policy_Discrepancy/runs/dl-normal-reference0-inputs-20261016-2a15c91/launch.py`；
- 完整 `log_sha256`；`controller_terminal={path,bytes,sha256}` 必须与强制传入的终态 pin 全同。

该契约已由 root 采用；实际收据仍须真实产生。12630秒通信观察 watchdog 不改变12600成功上限。
只读 `/proc` 核原 PID 不存在且原 PGID 无存活成员；PID 复用或未知身份均拒绝，不发送信号。

任何 NPZ 解码之前先完整真实 `rglob`，严格等于 producer 的
`generation_complete.output_inventory` 加两个已独立 raw 认证的顶层文件
`generation_complete.json`、`supervisor_terminal.json`。只允许512个固定 unit 目录，
不接受额外文件、symlink、缺项、partial、不同 source/order 或失败外部退出。

原 `scripts/dl_normal_feature_run.verify_worker` 在写这两个终态文件前执行。
本入口使用显式 `WorkerView` 复用该阶段验证函数：仅从 `rglob` 排除上述两个**精确顶层相对路径**，
所有暴露路径与 bytes 仍是原始只读文件。子目录同名文件不会隐藏。调用前后重新核完整真实库存；
这不是“完整 run 原样通过旧函数”，也不修改或复制任何原始产物。

producer 的完整源码清单逐 Git2a 原 blob 认证；实际复用的 verifier/source/tree helper、
官方概率 wrapper 和数学模块当前 bytes 另须与2a整文件相同。当前分析全部 py/yaml/lock、
AGENTS 和必要规范逐当前 Git blob 初末认证并要求 exact clean checkout。

## 输入语义与输出

固定原 sourceID、plan、每 family16条、前8 family128 fit、后24 family384 held；
source index、history、每帧 public/native oracle、reset/carry/key 链、只读 state/policy/counters、
全部原 NPZ/schema/raw/typed hashes 和 current 非 terminal、无 source raw[t] 边界均须完整认证。
真实 labels 的独立成功 CPU/外层退出收据、源 frame、t/t−1、torso-local frame、
query/RNG 只读封口及 F64 JSON/hex 逐行一致；不运行 Physics 或重复产生标签。

h 从原 uint16 BF16 bits 经 `ml_dtypes.bfloat16` 解码，再转 F64；q/p 从原32×32 F32概率转 F64。
不 clip、再 softmax、使用 logits/mode/sample 或复用旧 latent。官方 unimix 概率来源通过实际
producer/wrapper 源码、输入身份、history/oracle 及 typed bytes 认证，不声称独立重执行 NN。

只调用已验 NumPy 数学模块：固定 ridge .1、每个 group 仅 fit RMS，h/hq/hp 各自 refit，
两列 target 保持物理单位（矩阵解等价各自独立 ridge），截距不罚。先 family 内均值，再 family 等权。
三个 held 配对比较为 h−hq、hp−hq、h−hp；正值有利于后者。固定10000次、seed20261040、
pointwise95%区间条件于固定 fit，不传播 fit-family/训练seed不确定性，也不自动多重校正。

fresh 输出 `readout.json` 加 `readout.npz`：包含全部 fitted heads（means/RMS、weights、intercept、
kernel 全谱/rank/阈值/df/退化状态）、每行 prediction/residual/target、fit/held 绝对 family 风险、
fit 描述性差值、held 差值/区间、source/order 与完整输入和代码 pins。
JSON 中每个数值数组引用 NPZ key、dtype/shape/raw SHA；NPZ 不含 object/pickle。

结果没有阳性阈值或方法裁决；不能推出 actor 已利用、关键信息已丢失、唯一因果中介、保护有效或性能收益。

## r1本机有限验证历史

新增测试只使用生成/明确 mock 的图与真实 NumPy 算术，真实 feature/labels 未读取、未拟合。
涵盖外部失败/超时/错设备/错 pin、partial/缺件/额外文件、被排除顶层终态篡改、
子目录同名冒充、view 前后真实库存变化、原 latent key 错误、BF16 signed-zero bits、
target/source错位、输入 hash 改动、固定 fit/held 拟合调用与完整数值产物闭包。
源/物理/NN 的真实认证仍依赖未来完整实际提取与外部退出证据；fixture 不能替代。

本机最终命令：`python -m unittest tests.test_dl_normal_readout_run -v`，9项全部通过、
无 skip，107.147秒/exit0。使用项目既有 validation Python，未安装依赖，未重新执行旧数学4项。
测试依据为当前工作文件；整文件 raw SHA256：
readout入口 `6f1d09877ad118c1e9f3a67825b432d69322990e89d85e2ebfb0954b5a4b3f47`，
test `7b6338bba4786d1195bb259bfe3aab249330ccd30f2732e7d930f08cbbd1a30d`。
上述本机测试时三文件尚未提交；本机没有 SSH/GPU/真实数据拟合。
root及独立Agent只读核认证顺序、严格phase view、数学/bit解码/输出与拒绝反例，未见阻断。
下一确定提交的reader9+既有math4项采用一次clean Linux CPU测试整入口300秒上限；
它不是实际读出执行预算，也不允许读取运行中的reference0或按其结果改数学/标签。

## r1确定提交的实际 Linux CPU 验收历史

actual/tested `dfd6756ad4f38f89de87e17bac7fd0192265979a`，sv3 fresh clean checkout：
上述reader9与原数学4项共13项全过、无skip，7个必需用例均在真实日志中为ok。
unittest14.966秒、命令15.148718191秒、整入口16.334839418秒/300秒，
root实际SSHexit0/17.391秒，无external failure，0GPU、无JAX/Physics/真实数据读取或拟合。
258份源码+九份规范的267个不同文件实际working bytes前后相同，逐dfd Git原blob认证，
checkout始终clean、全部本次owned registry为空；五份复用依赖逐整文件等于producer2a。
root与另一Agent独立核本机小镜像及Git blobs，未再次运行远端测试。

真实run `/data/Policy_Discrepancy/runs/dl-normal-readout-cpu-20261016-dfd6756/attempt01/`；
code `/data/Policy_Discrepancy/dl-normal-readout-code-20261016-dfd6756`。
本机小镜像 `analysis/outputs/dl-normal-readout-cpu-dfd6756/attempt01/`，
外层SSH日志与真实退出收据在其父目录。关键完整raw SHA256：

- terminal：`381c0168f353044a13ddb53ba677a5d2cc8c32a795afbabd91fecdacca5a5a11`；
- test log：`ef1b78c75110f9cdd8aea273b54139f56f2cf9bbf5c56b978f078c15b3e6c10f`；
- source binding：`7e5c4bb3c5641a6bcf4f5b776fd35f429935ebeee7ab159d49e734e15d132f8c`；
- binding before：`8bd426dacedc6ed7364aa5bfd8547db4ebb2840d43dae4a59358cd297dd5af94`；
- root实际SSH receipt：`84243783b4b07a6e1e870f3fd368ca859ce82f8784d762d261d90227fb5454c5`；
- root实际SSH log：`829a692f2ad89e2ba68b59f2ee5d290068a3be0c2598633e0d69931f6df03b65`；
- 部署记录：`6a4a255dccad6acdfbd9104591f420e1b9c1b043f94672c5814248c323282d67`。

部署mkdir/SCP/远端四文件sha256sum均真实exit0；外launcher11788B raw
`7ada9e599ebcf044bd1ac86463d6379346875be06b9dc1b97f2ec4b6e84443d9`，
observer `79a2bc43499954d2333417452577b1faa618ff3ac146ca8f3546e2fba2ddea15`，
原group helper `ec3e0c45e643db875fe7a15619060facc72828cf06c80da094621629db48f6f9`，
bundle29695291B `fcbb98ccb387a753e800ff20282b4e338bbc6607770669810d1e60251a7f271a`。
这些是generated/mock provenance、真实NumPy/BF16解码和phase-view有限工程证据；
不认证真实512特征、标签、NN/Physics或方法有效。

## 完整CPU入口900秒的采用与F1读出路径历史

2026-10-16，Agent在完整预验证授权内采用一次reference0固定读出，当时actual/tested为dfd，
**整个CPU入口900秒、0GPU**，覆盖bootstrap前计时、全图/源码/标签认证、原固定拟合、
全部数值产物、初末只读检查、落盘、owned清理及实际完整退出。旧600秒候选不采用。
当时原feature handle37919仍运行；其后已实际SSH退出1，因而该F1路径不能进入读出。
当前改为等待F2真实SSHexit0/whole≤12600、512完整图与原组退场均成立后提供两个真实raw pins，
不使用partial、内部成功替代真实退出、不因读出符号更换来源或数学。

当时指定的输出 `/data/Policy_Discrepancy/runs/dl-normal-reference0-readout-20261016-dfd6756/attempt01/`，
外部inputs `/data/Policy_Discrepancy/runs/dl-normal-reference0-readout-inputs-20261016-dfd6756/`。
这两处旧dfd路径不供F2执行；新consumer Git与fresh读出路径待root确定。
900秒是有限成本上限，非完整任务耗时保证；失败保留，不延时或拼接。
该采用不启动累计学习、不增加GPU，也不采用T4096/6h或新的研究效力门。

## r1外部900秒控制器部署历史（未执行，现不可运行）

外部launcher43274B rawSHA `e23811bfd67709081ef954f126ea66819066864b1db9b594217f9186c69c287e`，
原generic helper34052B `3f0b8b025128d3280df2120f3987a6f7ad8806054fab5ec2e57b6decfeca28f1`，
本机准备包 `analysis/outputs/dl-normal-reference0-readout-controller-r1/`。
以下仅保留当时准备/部署记录；该包仍固定旧F1/GPU4/dfd，不能用于F2。
25个通用函数AST等于已实际Linux检验的3f版本，不引入C-specific authority、资格或report判据。
16个纯生成元数据/数值闭包反例拒绝，正负contrast均可接纳；未运行真实拟合、NN或Physics。
root与第二Agent只读核真实dfd接口、完整2a schema、两未知真实pins、全图初末、全部数值descriptor及实际reader stdout report pin。
控制器认证264个Git来源，reader自身analysis_binding为260项且全被覆盖；
这与上述CPU验收258源码+九norm的267个不同文件是不同集合，不为凑数改变范围。

当时仅上述两个文件已实际部署到上节历史路径的旧dfd读出inputs，单次SSH原bytes写入并逐文件读回SHA全同，
远端reader仍dfd且clean；部署未执行main、读取真实feature或增加GPU作业。
实际部署收据 `analysis/outputs/dl-normal-reference0-readout-dfd6756/controller-deployment.json`。
CPU环境隐藏CUDA/禁用GL，固定PYTHONPATH为已认证CODE、BLAS/OMP线程1。
launcher两项必填 `--feature-controller-sha`、`--feature-outer-sha`；后者绑定新inputs内
`feature-outer-execution.json`的原root真实退出bytes，当时尚未产生；F1其后失败，不能提供成功退出bytes或填生成正例。
root另备实际SSH observer，同一900秒成功上限、925秒仅失败观察；真实完成时才产生
`analysis/outputs/dl-normal-reference0-readout-dfd6756/ssh-terminal.json`；该旧路径未执行真实读出。
内部terminal/读出报告不能证明自身未来完整退出；这些工程准备不表示真实读出或方法通过。
