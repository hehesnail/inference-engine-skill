# CUDA IMA：本地操作说明

异步 CUDA 错误可能在后续同步或图启动时才暴露。保留首个异常、实际 binary/版本和触发输入；在已授权的隔离复现中，让驱动在异常点保存 GPU 状态。

为本次任务设定持久输出目录，再给复现进程设置：

```bash
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_SHOW_PROGRESS=1
export CUDA_COREDUMP_FILE="$RUN_DIR/cuda_coredump_%h.%p.%t"
export CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory'
```

裁剪选项可降低体积，但也省掉内存内容。需要检查具体数据时有针对性地调整范围。容器输出应持久保存到服务器。

```text
cuda-gdb
(cuda-gdb) target cudacore /absolute/path/to/dump
(cuda-gdb) bt
```

核对 faulting kernel、设备、grid/block/thread、PC 和调用栈，再对应实际生成代码。`CUDA_LAUNCH_BLOCKING=1` 对图内错误的定位有限，也会改变时序。debug 设置不带入正式性能对照。

依据：[vLLM 官方 CUDA core dump 指南](https://vllm.ai/blog/2025-08-11-cuda-debugging)。平台与驱动支持以目标环境为准。
