# 多模态推理：processor、状态与计时边界

用于 vision-language、语音识别与 diffusion 接入/优化。把输入变换、encoder、语言模型或 denoiser、输出处理分开定位；服务协议成功与模型数值正确分别验证。

## Vision-language

沿 `媒体 → processor → patches/tiles → encoder/projector → placeholder/embedding merge → decoder` 核对 shape、dtype、device 和顺序。相同原图不保证不同 processor 配置得到相同输入；记录 resize/crop、归一化、tile/patch 策略、image_sizes 和模板占位符。

- any-resolution 或多图输入可能返回不同长度的张量列表，不能无条件对 `process_images` 的返回调用 `.to()`，也不能遗漏模型需要的 `image_sizes`。
- processor 展开后的 embedding 占位位置与所选图像特征必须匹配；原始模板的单个 <image> 可以合法扩为多个视觉 token，部分模型还插入行/结束标记，按实际映射选择 feature 位置，不要求原始标记数等于 patch 数。多个请求或多图重排后，以样本身份检查映射，不能仅看总 tensor shape。
- encoder cache 需要把媒体内容、预处理参数、模型/adapter 身份纳入实际缓存语义。字符串 URL 相同不证明媒体未变；不同模板也可能改变 decoder 输入。
- context budget 包括媒体展开的 tokens，原始文件大小或图片数量不能直接替代 token 预算。报告 text/media prefill、decode 与端到端时长的边界。
- HF generate、LLaVA 自定义 generate 和引擎 serving 的返回可能分别含 prompt+completion 或只含 completion；依接口判定是否截去 input 长度，不机械切片。
- 项目 CLI 参数要检查所属模块；eval.run_llava 的 query 示例不能直接套到 serve.cli。Web UI 通常另需 controller/model worker。

现有 SGLang 接线、graph lifetime 和 processor parity 细节见 [专项路径](sglang-specialized-paths.md)。

## 语音

原始音频的采样率、声道、重采样/幅值、分段、VAD、语言和 task 都会影响结果。保留原始时长，并区分预处理、encoder、decoder、文本后处理。RTF = 墙钟秒/音频秒，需要说明是否包括加载、I/O、VAD、batch 和并发。

Whisper 使用窗口化处理，不意味着有任意长度或任意实时协议的保证；turbo 以转写为主，不能作为翻译任务的通用替代。faster-whisper 返回的 segments 是惰性迭代，实际计算可能发生在遍历时；单次 transcribe 的 generator 不自动构成在线流式服务。循环逐个文件处理也不是 batch 推理。流式实现要明确重叠窗口、跨段上下文、去重、时间戳和 final 状态，避免重复文字或漏字。

比较 WER/CER/翻译质量时固定数据切分与规范化规则；延迟需包含用户真正等待的阶段。不要把模型官方历史速度表当作当前卡上的实测结果。

## Diffusion

输入 seed、latents、scheduler、step 数、guidance、prompt/negative prompt、resolution、batch、VAE 和 adapter 共同决定实验。不同 scheduler 或少 steps 会改变生成轨迹和质量，不能把它们当作语义不变的 kernel 优化。

- CPU offload 在适当模型放置流程中启用，先把所有模型 `.to("cuda")` 再 offload 可能造成加载峰值；按 pipeline 支持的顺序核对。
- SDPA、xFormers 与 attention slicing 是不同路径，叠加可能减速或根本不生效。记录实际 backend；VAE slicing 主要沿 batch，tiling 主要沿空间分块，关注接缝与边界。
- fused LoRA 已合并进权重；卸载 adapter 模块不必然撤销 merge，需要对应 unfuse/恢复原权重流程。并发请求不同 adapter 时明确锁、复制或动态融合契约。
- BF16 的指数范围较大，不代表尾数精度更高。黑图先查 NaN/Inf、VAE dtype、数值溢出和输入，不能把关闭 safety checker 当作通用数值修复。
- graph 捕获需要覆盖真实 shape/step 控制与可变输入；同 seed 不是所有后端 bitwise 一致的承诺。对本次优化选择约定的 latent/中间输出/最终质量指标。

跨模态共同遵循 [状态生命周期](stateful-inference.md)、[attention 契约](attention-backend-contracts.md)、[量化契约](quantization-engine-contracts.md) 和 [数值验证](numerical-validation.md)。只展开任务涉及的模型与路径。
