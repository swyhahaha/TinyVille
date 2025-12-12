vllm serve Qwen/Qwen3-8B \
--port 8000 \
--host 0.0.0.0 \
--dtype bfloat16 \
--tensor_parallel_size 1 \
--served-model-name Qwen3-8B \
--gpu-memory-utilization 0.8