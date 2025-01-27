# DeepSeek R1: Benchmarking Performance Across GPUs

## Table of Contents
1. [Introduction](#introduction)
2. [Benchmark Performance](#benchmark-performance)
3. [Hardware Compatibility](#hardware-compatibility)
4. [Apple M3 Max Performance](#apple-m3-max-performance)
5. [Conclusion](#conclusion)

## Introduction <a name="introduction"></a>
DeepSeek R1 is a state-of-the-art language model designed to deliver high performance across a variety of tasks, including mathematical problem-solving, coding, and general-purpose question answering. With its range of distilled models, DeepSeek R1 offers flexibility and efficiency, making it suitable for both research and practical applications. This article provides an in-depth look at the performance benchmarks, hardware compatibility, and specific performance metrics on Apple's M3 Max hardware.

## Benchmark Performance
DeepSeek R1 models have been rigorously tested against several benchmarks to evaluate their capabilities. Below is a summary of the performance metrics:

| Model                             | AIME 2024 pass@1 | AIME 2024 cons@64 | MATH-500 | GPQA Diamond | LiveCode Bench | CodeForces Rating |
|-----------------------------------|------------------|-------------------|----------|--------------|----------------|-------------------|
| GPT-4o-0513                       |  9.3             | 13.4              | 74.6     | 49.9         | 32.9           | 759               |
| Claude-3.5-Sonnet-1022            | 16.0             | 26.7              | 78.3     | 65.0         | 38.9           | 717               |
| OpenAI-o1-mini                    | 63.6             | 80.0              | 90.0     | 60.0         | 53.8           | 1820              |
| QwQ-32B-Preview                   | 50.0             | 60.0              | 90.6     | 54.5         | 41.9           | 1316              |
| DeepSeek-R1-Distill-Qwen-1.5B     | 28.9             | 52.7              | 83.9     | 33.8         | 16.9           | 954               |
| DeepSeek-R1-Distill-Qwen-7B       | 55.5             | 83.3              | 92.8     | 49.1         | 37.6           | 1189              |
| **DeepSeek-R1-Distill-Qwen-14B**  | 69.7             | 80.0              | 93.9     | 59.1         | **53.1**       | **1481**          |
| **DeepSeek-R1-Distill-Qwen-32B**  | 72.6             | 83.3              | 94.3     | 62.1         | **57.2**       | **1691**          |
| DeepSeek-R1-Distill-Llama-8B      | 50.4             | 80.0              | 89.1     | 49.0         | 39.6           | 1205              |
| DeepSeek-R1-Distill-Llama-70B     | 70.0             | 86.7              | 94.5     | 65.2         | 57.5           | 1633              |

The DeepSeek R1 models consistently perform well across various benchmarks, demonstrating their robustness and versatility.

## Hardware Compatibility
DeepSeek R1 models are compatible with a wide range of hardware configurations, ensuring flexibility for different use cases. Below is a summary of the hardware benchmarks:

Here is the updated table with the new column **1k tokens/$1** added:

| Model                      | GPU                    | VRAM       | Tokens/sec | Cost (USD) | 1k Tokens/$1 |
|----------------------------|------------------------|------------|------------|------------|--------------|
| DeepSeek-R1-Distill (14B)  | RTX 3090               | 24GB       | 58         | 800        | 72.50        |
| DeepSeek-R1-Distill (14B)  | RTX 4070 Ti Super      | 16GB       | 52         | 890        | 58.43        |
| DeepSeek-R1-Distill (14B)  | RTX Titan              | 24GB       | 44         | 789        | 55.77        |
| DeepSeek-R1-Distill (32B)  | RTX 4090               | 24GB       | 36         | 1900       | 18.95        |
| DeepSeek-R1-Distill (32B)  | RTX 6000 ADA           | 48GB       | 36         | 6199       | 5.81         |
| DeepSeek-R1-Distill (32B)  | RTX 3090               | 24GB       | 31         | 800        | 38.75        |
| DeepSeek-R1-Distill (32B)  | RTX 4070 Ti Super x 2  | 16GB x 2   | 26         | 1780       | 14.61        |
| DeepSeek-R1-Distill (32B)  | RTX A6000              | 48GB       | 23         | 3995       | 5.76         |
| DeepSeek-R1-Distill (32B)  | RTX Titan              | 24GB       | 23         | 789        | 29.15        |
| DeepSeek-R1-Distill (32B)  | Apple M3 Max 40 GPU    | 128GB      | 23         | 4000       | 5.75         |
| DeepSeek-R1-Distill (70B)  | RTX 4090 x 2           | 24GB x 2   | 19         | 3800       | 5.00         |
| DeepSeek-R1-Distill (70B)  | RTX 6000 ADA           | 48GB       | 19         | 6199       | 3.07         |
| DeepSeek-R1-Distill (70B)  | RTX 3090 x 2           | 24GB x 2   | 17         | 1600       | 10.63        |
| DeepSeek-R1-Distill (70B)  | RTX A6000              | 48GB       | 12         | 3995       | 3.00         |
| DeepSeek-R1-Distill (70B)  | RTX 8000               | 48GB       | 10         | 1950       | 5.13         |
| DeepSeek-R1-Distill (70B)  | Apple M3 Max 40 GPU    | 128GB      | 4          | 4000       | 1.00         |

**Note:**

- The **1k Tokens/$1** column represents the number of tokens generated per $1 spent, calculated as: **1k Tokens/$1 = (Tokens/sec × 1000) / Cost (USD)**.  
- This metric helps evaluate the cost efficiency of each hardware configuration.

These benchmarks highlight the adaptability of DeepSeek R1 models across different hardware setups, ensuring optimal performance.

## Apple M3 Max Performance
For users leveraging Apple's M3 Max hardware, DeepSeek R1 models offer impressive performance metrics:

| Model                                      | GPU     | VRAM   | Tokens/sec | First token latency |
|--------------------------------------------|---------|--------|------------|---------------------|
| DeepSeek-R1-Distill-Qwen-32B (MLX-4bit)    | M3 Max  | 128GB  | 19.00      | 0.67s               |
| DeepSeek-R1-Distill-Qwen-32B (MLX-8bit)    | M3 Max  | 128GB  | 10.57      | 0.30s               |
| DeepSeek-R1-Distill-Qwen-32B (Q4_K_M-GGUF) | M3 Max  | 128GB  | 15.93      | 0.73s               |
| DeepSeek-R1-Distill-Qwen-32B (Q8_0-GGUF)   | M3 Max  | 128GB  | 7.50       | 0.92s               |
| DeepSeek-R1-Distill-Llama-70B (MLX-4bit)   | M3 Max  | 128GB  | 9.30       | 6.74s               |

These results demonstrate that DeepSeek R1 models are well-optimized for Apple's M3 Max, providing efficient token generation and low latency.

## Conclusion
DeepSeek R1 stands out as a versatile and high-performing language model, suitable for a wide range of applications. Its robust performance across various benchmarks and compatibility with different hardware configurations make it a valuable tool for both researchers and developers. Whether you are using high-end GPUs or Apple's M3 Max, DeepSeek R1 delivers consistent and reliable results.

## References
- [DeepSeek R1 (Distill LLM) Performance Analysis: GPU and Apple M3 Max Benchmarks](https://www.reddit.com/r/LocalLLaMA/comments/1i69dhz/deepseek_r1_ollama_hardware_benchmark_for_localllm/)

