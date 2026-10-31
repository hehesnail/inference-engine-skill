# LLaVA：图像特征与语言模型接线

固定checkpoint、vision tower、projector、tokenizer与conversation template。沿图像预处理、patch/tile、encoder、projector、placeholder展开和decoder定位，不能只验证文本生成API可调用。

## 多图与any-resolution
`process_images` 可能返回tensor或变长列表；按实际返回逐项放到设备，不能一律调用 .to。any-resolution等模型还依赖image_sizes。特征顺序、processor展开后的占位位置映射、文本token与batch重排必须一致（原始单个图像标记可以展开为多个视觉token），媒体token预算计入context。

不同generate实现返回prompt+completion或completion-only，需要按接口决定decode是否切去input长度。直接套通用HF切片可能丢掉全部答案。

## 使用与测量
CLI的模块和参数配对：某些`--query`例子属于eval.run_llava，不属于serve.cli。Gradio服务还可能需controller/model worker，不能只启动UI即认为模型已就绪。undefined ask_image/compare_images等只能标为伪代码，不能称可执行API。

冻结processor输出比较encoder/projector与decoder路径；性能包含哪些图像I/O/预处理/encoder阶段明确标注。详细多图cache、dtype与processor parity见 [多模态专题](../../../../references/multimodal-inference.md)。

## 来源与配套材料

来源：Orchestra-Research/AI-Research-SKILLs，固定快照 `773a52944ba4747a18bd4ae9ade53fff041adcbc` 的 `18-multimodal/llava/SKILL.md`。本页经过全文语义审阅与改写，保留推理工作所需机制和适用边界；上游原文保存在研究源码快照中。

配套资料按需查阅；除本次逐项记录的核对范围外，历史示例仍需与目标版本实现对照，不能自动认定可运行。

- [training](references/training.md)

完整历史正文及本次具体勘误保留在 [审阅材料](source-notes/guide-at-pin.md)，仅用于追溯；运行规范使用本页与专题。
