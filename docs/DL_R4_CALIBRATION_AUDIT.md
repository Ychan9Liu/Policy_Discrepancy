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
