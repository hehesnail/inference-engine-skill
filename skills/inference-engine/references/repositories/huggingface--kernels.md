# huggingface/kernels

固定提交：[633d1faef0caea5f2b9acf447e17ad5f840a52d1](https://github.com/huggingface/kernels/tree/633d1faef0caea5f2b9acf447e17ad5f840a52d1)。

先读本地适配：[构建与后端](../backend-kernel-development.md)。

当前采用 5/5 项。采用项正文逐篇语义审阅；相关源码和附属资源的核对范围单列，不能推广成整仓每行已验证。技术修订与限制见 [语义核对](../semantic-audit.md)。

| 专项 | 用途 |
|---|---|
| [cpu-kernels](../../library/hf-kernels/kernel-builder/skills/cpu-kernels/GUIDE.md) | SIMD separate TU、runtime dispatch、tail处理、NUMA及量化packing契约。 |
| [cuda-kernels](../../library/hf-kernels/kernel-builder/skills/cuda-kernels/GUIDE.md) | HF kernel-builder namespace/ABI3、C++dispatcher与fake实现接线，有真实guard/currentstream范例。 |
| [rocm-kernels](../../library/hf-kernels/kernel-builder/skills/rocm-kernels/GUIDE.md) | ROCm wave/布局/stride与真实Diffusers对照、尾部mask与cache计时。 |
| [triton-kernels](../../library/hf-kernels/kernel-builder/skills/triton-kernels/GUIDE.md) | 通用Tritonmask/stride/reduction/FP32accumulation、autotune键与GB/s/FLOPs定义。 |
| [xpu-kernels](../../library/hf-kernels/kernel-builder/skills/xpu-kernels/GUIDE.md) | XPU register压力估计与fallback、swizzle/descriptor选择、可复用trial管理。 |
