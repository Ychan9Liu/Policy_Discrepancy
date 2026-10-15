# DL-r5 live诊断：一次工程r2完整重试

2026-10-15。科学对象仍为[DL-live-utility-r1](DL_R5_LIVE_TRIAL.md)，
**Agent在完整预验证授权及sv3物理4–7资源范围内采用工程版本DL-r5-engineering-r2**。
采用发生在未读取旧attempt任何中间D/e/Y、cal系数或科学统计的条件下；不是用户逐项确认、科研阳性或原失败通过。
本规范只覆盖下述一次新完整attempt的必要工程调整，尚未启动GPU。

## 工程冻结与资源范围

2026-10-15 19:11源码冻结：父入口、family工程层、独审及两测试文件；本机有限R5组合38项，
37通过、1项Linux进程组测试在Windows跳过（15.490秒），尚待提交后clean Linux复核。
两个独立只读审查确认run_unit AST与原f845/e237一致，policy/shadow/statistics、物理实现、
Dreamer agent/rssm/dt_latch及原科学规范没有改动；这是源码/fixture证据，不是真实512单元验收。
提交后须核父/独审真实Git blob清单逐值相等、新规范pin与对应blob一致，以及测试前后clean。

资源需求为**一张独占A100-SXM4-80GB串行**，物理4–7是授权范围，不是四卡并占。
本次主入口至多3.5占卡小时；旧同规格诊断采样峰约2587MiB只作实测参考，不是最低显存保证。
CPU库存、私有replay、tiny工程、ridge/bootstrap和独立物理复放不需GPU。
之后真实模型特征提取及正常压缩学习仍需单卡，但未采用的NM1组数/学习预算不能充作整轮总核时承诺。

## 旧失败与工程调整的依据

原actual/tested f845de0cce297443c28761421ab91a7955b62553，原完整入口7200秒保持原文不改。
2026-10-15 18:42:31.635740北京时间parent exit1，外4044.184685秒（约1.123385占卡小时）；
family13终态报告资源守卫发现GPU4未知进程，主动只清理自有组，整体未完成。
13家族/208单元已由主进程接纳，磁盘有223个unit complete（含未完整family13的15个），
无generation_complete、全对象统计或独审，不能拼接/分析partial或将其作完整normal-reference。
失败证据与限制见[运行报告](DL_R5_LIVE_RESULTS.md)。

原资源守卫在保存冲突sample前抛错：最后持久化快照18:42:15仍仅本worker，
所以不能从稍后出现的未知PID反推具体触发PID；guard报告不是独立保存的冲突原始快照。
原worker_returncode=None因异常前赋值，终态/原PID退出及owned清理证据另核，不伪造退出码。

按13完整family/4044秒保守外推约9955秒；按已冻结名义policy工作量计约9172–9957秒。
这是只用预定计划和运行时间的吞吐推算，未使用D/e/Y，不是统计功效或成功ETA。
因此下一完整入口一次固定12600秒，约26%–37%工程余量；不能声称原7200通过，也不能再超时后逐次加时求成功。

## 本次允许且必须显式绑定的变化

- 物理GPU固定为已授权sv3 **5**，UUID `GPU-79704918-e096-92b4-252c-e3244497a8b3`。
  曾在18:50见GPU4未知C/C+G作业（proc AccessDenied）；19:00:24 fresh XML+pmon4–7均空是后续快照，
  不代表未知进程归属或持续空闲。新启动前仍重新核GPU5独占及CUDA/EGL实际同UUID。
- **新整个主入口12600秒**，含所有import/load/JIT、32fresh family、所有unit/统计/输出和owned清理；
  不是每family12600。独立CPU全对象审计仍1800秒/0GPU，设备probe另记成本。
- 新Git提交、新clean隔离checkout、新run/attempt目录，从完整原N1M初态重新生成全部512unit；
  不复用旧cal、任何旧unit/NPZ/pickle/renderer/carry或部分文件，不在旧run补齐。
- 新budget及独审绑定engineering_version和本规范的真实Git blob/SHA；原科学规范也仍绑定。
  旧f845执行checkout/原run保持不动，旧失败不由最新源码反向重命名为成功。
- 先持久化资源sample（含可用XML查询摘要和raw pmon）再判定陌生PID；冲突仍立即仅清本组，
  不停止/驱逐陌生作业、不共享卡。异常cleanup后记录实际child.returncode，保留guard failure。

## 科学内容和验收一项不改

原N1M/config pins、size50m/BF16/vision64/camera2/repeat1/root0、32family×16流、
所有env/eval/时间/随机label/root/NJ子key、原H1处理前CPU shadow、实际source r_t、三条qq/pq/pp路径、
H100/真实终止padding、8cal/24held和先封口顺序、所有ridge/.1/高D阈值/coverage、主量与二次基底残差对照、
family bootstrap/随机化检验/停止门及全部真实物理/完整state/来源/设备/退出独审契约均保持。
不加/减单元、按收益选位置、更换科学主比较或将sham精度标记改为主门。
仍无optimizer更新或正式训练。物理卡变化不是额外训练seed，也不构成renderer逐位重演。

完整生成/权威exit0且全512官方CPU独审通过后才作原已冻结统计裁决；不支持/不确定均按原范围接受，
不能再换seed/H/模型/标签追阳性。再次冲突、超时或工程失败保留全失败对象，
不自动第三次延时/改门；在原完整目标内重新论证必要下一动作。

此重试优先于独立source-only，因为后者不能替代尚缺的独立信号辨别；
R5科研阳性仍不是普通压缩问题的逻辑必要前提。[正常来源](DL_NORMAL_REFERENCE.md)是备用前置，
[NM1学习提案](DL_NM1_PROTOCOL_REVIEW.md)的U/cal/native/S/P/W仍未采用，不由重试自动启动。
