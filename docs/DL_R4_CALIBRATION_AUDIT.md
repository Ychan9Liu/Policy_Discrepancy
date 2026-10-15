# DL-r4 校准停止证据核验

版本 **DL-r4-calibration-audit-r1**，2026-10-15。Agent在原完整预验证授权内采用的失败取证与CPU复核；
不新增GPU、模型前向或训练，不修改DL-r4-utility-r1科学判据。实际生产版本始终为
`5ce1d3724ee8c5d01f09c4d687100a2f42f6ac83`；新审计代码须另记确定SHA。

## 已知终态与核验问题

原run `/data/Policy_Discrepancy/runs/dl-r4-utility-20261015-5ce1d37-attempt01` 已在cal8门停止：
8份complete及cal封口存在，null_passed=False、worker exit1，末异常为预定calibration null failure；
未生成held。15:54:43原parent/worker均消失、GPU4释放，15:56只读终态一致。
这些是主程序与现场收据事实，尚不是独立复放后的测量裁决。

本次问题限于：原planned selected状态的同一qq/pair0与qq_null是否确有超过原阈值的真实奖励轨迹差异，
且该失败证据能否被完整来源链与独立CPU物理复放认证。**不对未生成的24held计算信号增量或覆盖。**

## 取证与审计范围

1. 在原run外建立fresh外部capture。先确认原进程终止，核精确失败、八cal与无held/worker_complete/statistics；
   对所有既有run文件做两次稳定清单/哈希，显式保存目录和缺失对象。先封口capture本身的SHA再交独审。
   这是停止后取证，不伪称producer成功闭合；原数据和失败记录不写回。
2. 新clean审计checkout认证其自身SHA；原budget.source_files按生产5ce的Git blobs核验，区分run SHA和audit SHA。
   核terminal/log/resource/runtime/input pins、cal seal→8份manifest→全部文件hash图，全部hash通过才解可信pickle。
3. 全部预声明selected cal状态的pair0 qq与qq_null逐一核源状态、RGB/历史、bundle/keys、首动作/carry；
   使用原独立审计的官方CPU physics/restore复放stored-applied control、逐步F64 reward/fullstate及静态MJB/camera。
   用math.fsum/固定100分母重算每条回报，检查与原数组求和口径的一致性及原阈值`.0025`两侧分类。
   若数值舍入改变判门，独审拒绝发布裁决，外部CLI保留该失败并记为算术侧不一致；不能选择有利口径或声称已有结构化成功结果。
4. 认证失败与科学测量失败分开。只有链完整且至少一个预声明null真实超阈，才可写
   `audit_integrity_passed=True, qualification_passed=False, status=measurement-inconclusive`；
   始终`held_generated=False, held_increment_evaluated=False`。所有null连续差值联报，不只挑最大值。

原32EP成功审计继续要求worker_complete和完整held，不能放宽以接纳本次partial run。
新工具用针对来源篡改、未终止/held污染及真实CPU物理反例的有限测试，再在确定提交上实际执行。

## 不能据此推出什么

仅复放pair0 qq/null，不冒称其余15roots、pq/pp、全部source或全部分支已独审。
它不独立重算模型/head/D/K/NJ随机流或render，不重跑真实FromDM wrapper/训练；同一官方物理后端也不是第二次独立实验。
本cal失败没有正常worker最终全模型状态seal/policy receipt/runtime_final，现有各EP只读收据不能补造该全程成功闭合。
单个null重复仍不是全MC或渲染误差界；认证真实差异也不自动识别renderer、政策敏感性或长时放大中哪一个是根因。

原失败保留，不能增EP/root/H、换顺序、松阈值或重跑求过门。后续先依据认证后的首差位置与误差传播辨别下一必要干预，
另立有限对象和判据；无充分依据不启动新GPU。正常active压缩/保护因果链仍未完成，正式M2不变。

## 执行与结果

新工具为`scripts/dl_r4_failure_capture.py`与`analysis/dl_r4_calibration_audit.py`，各有独立tests。
Windows四组40项38通过/2skip，16.114秒；skip分别是Windows symlink权限和Linux进程组。
包括新metadata正负接线、真实CPU qq/null200步、.0025等号/1ULP及求和门侧、坏hash先于decode、
完整snapshot/首次差异索引、旧full auditor/runner回归。这是fixture证据，不是实际cal数据通过。
独立源码/schema交叉审未见阻断；已将算术门侧不一致的文档表述纠正为抛异常拒绝裁决。
下一在确定clean SHA的新sv3审计checkout执行Linux这40项，原生产5ce checkout不移动；
然后capture原run全部文件两次封口，并固定capture SHA运行CPU独审。
确定SHA、Linux结果、外部capture路径/SHA、CPU实际时长及科研范围内的结果将在完成后追加。

### 实际CPU执行与有限结论（2026-10-15 16:12）

实际/tested审计SHA：`4ddad2e9a92ddfe95979d104d0f004722c7f7dff`。
sv3新clean checkout `/data/Policy_Discrepancy/dl-r4-cal-audit-code-20261015-4ddad2e`；
Linux四组40项全部通过5.897秒，完整入口6.325646秒/0GPU，原生产checkout保持5ce clean。
没有Dreamer/embodied算法、原runner/统计/完整auditor修改。

新输出根 `/data/Policy_Discrepancy/runs/dl-r4-cal-audit-20261015-4ddad2e-attempt01`：

| 对象 | 结果/认证 |
| --- | --- |
| `capture.json` | 1205文件、5,264,343,594字节，两次库存/目录/hash稳定；外入口17.136417秒/exit0 |
| capture SHA256 | `df6b9f5789decab5931ade9f8cec57ae9b170090dca8bbffd94ac09ee4541658` |
| 外部pins | 原checkpoint/config的实际文件SHA仍为原c6a7d92e…/877f8470…，并保存launch/设备probe/resource原始文件hash；未执行模型 |
| `audit.json` | 内入口35.059434秒、外入口35.159777秒/exit0、0GPU，JAX未导入 |
| audit SHA256 | `7ff17776be2c7f0ee11912d8ed6ce3f1cd60509085c2dd9bab0dd79ed49eb1d6` |
| 独立范围 | 全部23个预声明pair0 qq/null、4600个官方CPU物理步，stored-applied/F64 reward/积分状态与snapshot链通过 |
| 裁决 | integrity通过、qualification失败、`measurement-inconclusive`；原attempt仍失败，held未生成/未评价 |

只有episode2/t64超原阈值：独立fsum平均绝对差`0.00319120049295762 > 0.0025`；
原np.sum差`0.003191200492957904`，门侧一致。其最大单步奖励差`0.18790843912517763`。
其余22个对照最大平均差`0.0013101225970738994`。不能删除这一状态、改为22/23通过或据此放宽阈值。

**已观察的存储序列顺序，不是根因裁决**：所有23对在frame1–5首次出现RGB字节差，首差最大绝对通道值均为1；
selected/action logits首差与该frame索引相同。17对后续latent/raw有差异，6对整个H100未出现latent/raw差；
有raw差的对照，其积分状态首差出现在相应动作后的下一frame。
失败的episode2/t64：frame2为13个像素/21通道元素的±1差，logit在action2首次不同，
latent/raw与F64 reward在action5首次不同，积分状态在frame6首次不同。frame0/current、action0/首执行动作均按0起算。
核验通过了相同完整初始snapshot/历史/当前RGB/bundle/keys/首动作，但未独立重算GPU网络或render。
因此支持“当前序列存在图像微差先于动作/物理分叉”的有限观察；
“图像扰动导致BF16 categorical选择改变并闭环放大”仍是有该顺序支持的工作解释，
具体renderer有效输入、驱动/GL状态等根因未识别，不能写成已修复。

独立小收据复核（stdlib/Decimal/hash，不重跑原physics/模型）确认了capture/audit/terminal的SHA对应、
23组唯一控制/4600步、唯一超门及全部门侧；原np.sum与fsum平均差最大约2.84224e−16，未翻门侧。
该二次复核只证明小收据内部算术/哈希/解释一致，真实CPU物理事实仍来自上述确定4dd入口。

主GPU成本仍仅原attempt的约.728554 GPUh（probe约3.67秒另记）；本取证与独审新增0GPU。
16:15:33.553567北京时间fresh XML+pmon确认sv3物理4–7无进程、0%/各4MiB，原PID均不存在；
原主telemetry采样峰2587MiB，不是最低显存保证。`resource-final.json`保存端点及两个clean SHA。
本机只镜像上述小JSON至`analysis/outputs/dl-r4-utility/calibration-audit-20261015-4ddad2e/`（ignored），
没有把checkpoint/轨迹/图像提交Git。

### 下一问题与替代方向（建议，未采用新GPU对象）

首选继续现有chat，先0GPU明确下一标签估计对象和资格判据：
若继续研究native H100闭环效用，需要事前定义并控制render/rollout变动对**MC均值**的影响及处理顺序，
不能把单次重复当全MC误差界，也不能只是改本次`.0025`门求通过。
替代方向是用冻结未来控制脚本的真实物理后果建立不依赖未来RGB的外部标签；
它只能支持该有限控制族的可访问任务效用，不等同native闭环性能，不能冒名替换本次主量。
两者均须说明对双信号辨别和实际保护机制分别能提供何种证据，以及怎样的反例会停止该对象。
在对象、必要性/充分性、固定新预算与判败条件明确前，不新增GPU实验。
普通replay文件仍在只提供后续机制对象的前置可行性，不构成B/N协议已采用或旧online流可恢复。

### 全部预声明null的连续值

下表仅为完整精度JSON的显示舍入；判门使用上述完整精度，不重新分类。

| episode | position | 平均绝对回报差 | 超原.0025门 |
| --- | --- | --- | --- |
| 0 | 64 | 6.49191215756e-05 | 否 |
| 0 | 448 | 0.000137157660116 | 否 |
| 0 | 512 | 0 | 否 |
| 1 | 128 | 0.000850342283527 | 否 |
| 1 | 512 | 9.41700884769e-05 | 否 |
| 1 | 640 | 0.00070754582971 | 否 |
| 2 | 64 | 0.00319120049296 | 是 |
| 2 | 320 | 0.000146232053782 | 否 |
| 2 | 512 | 1.93665882875e-06 | 否 |
| 3 | 384 | 0.000147055864391 | 否 |
| 3 | 512 | 0 | 否 |
| 3 | 576 | 0 | 否 |
| 4 | 192 | 0.000697004635468 | 否 |
| 4 | 512 | 0.00067105423298 | 否 |
| 5 | 192 | 0.000160868761688 | 否 |
| 5 | 256 | 0.000170708478666 | 否 |
| 5 | 512 | 0 | 否 |
| 6 | 64 | 0.00131012259707 | 否 |
| 6 | 320 | 5.04185036334e-05 | 否 |
| 6 | 512 | 0 | 否 |
| 7 | 64 | 0.00011351626131 | 否 |
| 7 | 384 | 4.37032717997e-06 | 否 |
| 7 | 512 | 0 | 否 |
