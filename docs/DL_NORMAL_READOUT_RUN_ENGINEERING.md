# 正常 reference0 读出：只读 CPU 工程接口

2026-10-16，`DL-normal-readout-run-r1`。本文件说明实现与有限工程证据；不新增研究判据、
训练组、T 或真实读出执行授权。数学定义仍以 [DL_NORMAL_READOUT.md](DL_NORMAL_READOUT.md) 为准。

## 固定对象与命令

`analysis/dl_normal_readout_run.py` 只接受原始 reference0：producer actual/tested
`2a15c91aad16c0e3151fde3943e356dad1fcb69d`、sv3 物理4、完整512/32family、12600秒。
后续 B/N endpoints 需要各自完整来源协议，本入口不冒充通用 endpoint 认证器。
当前分析提交通过 `--expected-git` 单独记录，不能写成提取时实际提交。

所有参数必填，无默认 CPU 预算：

```text
python analysis/dl_normal_readout_run.py
  --features-run /data/Policy_Discrepancy/runs/dl-normal-reference0-features-20261015-2a15c91/attempt01
  --feature-controller-terminal <真实外部controller terminal路径>
  --feature-controller-sha <该文件完整rawSHA256>
  --feature-outer-execution <root真实SSH退出收据路径>
  --feature-outer-sha <该文件完整rawSHA256>
  --expected-git <当前分析完整Git SHA>
  --seconds <另行采用的完整CPU入口秒数>
  --output <所有输入及代码目录之外的fresh目录>
```

这些真实特征终态及 SSH 退出收据尚未产生，本文件不填假 pin。600秒只是待 root 采用的工程
预算候选。入口计时包括 import、认证、Git/raw读取、NumPy 拟合、写盘；不得延时、补 partial、
搜索超参或从输出符号调整来源。成功还需要实际外层退出0及完整 wall 在显式预算内；
`readout.json`/stdout 自身不能证明进程未来成功退出。

## 外部权威与完整图

先验证强制传入的两个 raw pins。controller terminal 必须为
`DL-normal-reference0-launcher-r1`，原 parent 实际0、success、gone/owned-clean、无错误、完整
reference0 scope，固定物理4/UUID/12600。root 实际 SSH 收据采用的字段契约为：

- `git_sha=2a15c91...`、`physical_gpu=4`、整数 `returncode=0`；
- `0 < elapsed_seconds <= 12600`、`external_failure=None`、
  `actual_outer_complete_within12600=True`；
- 精确 command 为 `ssh -o BatchMode=yes -o ConnectTimeout=10 sv3` 后接项目 Python 与
  `/data/Policy_Discrepancy/runs/dl-normal-reference0-inputs-20261015-2a15c91/launch.py`；
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

## 有限验证范围

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
