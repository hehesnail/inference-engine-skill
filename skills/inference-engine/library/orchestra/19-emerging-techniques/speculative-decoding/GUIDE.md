# Speculative decoding：正确性与收益

draft 提出候选，target 批量验证，再提交接受的 tokens。存在实现并不证明启用；非零 acceptance 也不自动证明更快。

## 必须保留的分布与状态
标准随机 speculative sampling 对候选 token 使用 min(1,p/q) 接受概率；拒绝后从归一化 max(p-q,0) 残差采样，而不是简单从 p 重采样。全接受时还涉及 target 的额外 token。p/q 必须对应各自实际采样变换后的分布。greedy、tree/Medusa、typical acceptance 等方案有不同契约，不能都称 lossless。

target/draft vocab、tokenizer、位置、KV/循环状态、grammar与RNG要一致提交或回滚。树形候选 mask 必须只见祖先；只截断某个 KV 长度不足以恢复混合循环模型。公式、具体反例和接线检查在 [接受与回滚专题](../../../../references/speculative-decoding-contracts.md)。

## 工程路线
按目标版本配置实际支持的 draft/EAGLE/MTP/ngram等路径；新 vLLM使用 speculative-config，不拼接旧 speculative-model API。Lookahead/Medusa 的示意算法不是可以直接 import 的统一第三方 API。训练 drafter 仅在任务需要时进行，loss/label shift与shape需实际测试。

## 性能
记录 draft、verify、reject处理、host调度、通信与额外内存；统计提出、接受、最终提交的 tokens，明确分母。acceptance threshold 与真实收益不保证单调。负载、并发、batch和输出长度改变收益；同seed随机输出不必逐token相同，概率等价与greedy一致的验收分别定义。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `19-emerging-techniques/speculative-decoding/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [lookahead](references/lookahead.md)
- [medusa](references/medusa.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
