# vLLM
Here's how to get vLLM running on Strix Halo (Ryzen 395) and some basic benchmarks/methodology for comparing.

## Setup 
You can refer to [torch-therock](../torch-therock) folder to install ROCm and build our fast PyTorch.

Adn the vLLM build from source docs (they don't work but are a good starting point):
- https://docs.vllm.ai/en/v0.6.5/getting_started/amd-installation.html#build-from-source-rocm

As for the actual install:
- `00-setup-env.sh` - create mamba `vllm` env and dependencies (assumes you've have Torch and stuff built)
- `01-build-vllm.sh` - working vLLM build; lots of little details different from docs

vLLM gotchas
- `use_existing_pytorch.py` ofc
- amd_smi segfaults gfx1151 because ofc AMD thanks
  - need to patch vLLM to not use amdsmi
- `pip install "numpy<2"`
- Modify CMakeLists.txt for gfx1151
- docs say to use setup.py - doesn't work!
  - use `pip install -e . --no-build-isolation`

## Benchmarking

### How to run
Run server:
```
vllm serve meta-llama/Llama-2-7b-chat-hf --gpu-memory-utilization 0.75
```
- vLLM will default to 90% memory usage, which may be more than you want in terms of system stability depending on how high you've set your GTT
- 0.75 = just about bang on 90GiB on my setup (120GiB GTT)
- Testing done from first run (empty prefix cache)

For llama.cpp we run:
```
ROCBLAS_USE_HIPBLASLT=1 build/bin/llama-server \
             -fa on \
             -ngl 99 \
             -np 16 \
             -cb \
             --kv-unified \
             --cache-reuse 256 \ 
             --port 8000 \
             -m /models/gguf/Llama-2-7b-chat-hf.f16.gguf
```

Setup:
```
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# High performance mode
# amd_iommu=off
sudo ryzenadj --stapm-limit=120000 --fast-limit=160000 --slow-limit=140000
sudo tuned-adm profile accelerator-performance
```
- in vLLM GPU never goes about 95W
- llama.cpp can go higher (100W+ sustained)


Run test, eg:
```
python benchmark_serving.py --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 64 --model meta-llama/Llama-2-7b-chat-hf --max-concurrency 8
```

### vLLM Concurrency
As expected, increasing concurrency increases throughput.
- testing with the "standard" llama2 first, meta-llama/Llama-2-7b-chat-hf (this if FP16 - 13.5GB)

| Model                         | Concurrency | Duration (s) | Req/s    | Output Tok/s | Total Tok/s | Mean TTFT (ms) | Mean TPOT (ms) |
|-------------------------------|-------------|--------------|----------|--------------|-------------|----------------|----------------|
| meta-llama/Llama-2-7b-chat-hf | c=1         | 1295.40      | 0.05     | 8.61         | 22.36       | **121.23**     | **115.88**     |
| meta-llama/Llama-2-7b-chat-hf | c=8         | 225.80       | 0.28     | 49.41        | 128.28      | 536.74         | 135.23         |
| meta-llama/Llama-2-7b-chat-hf | c=16        | **155.97**   | **0.41** | **72.17**    | **186.35**  | 871.62         | 162.64         |


### vLLM Quant comparison
In theory, quants should perform better. RDNA3.5 does not have native FP8 support but *does* have native INT8 so I thought I'd try W8A8-INT8 (which on Nvidia is usually faster than FP8) but INT8 craters hard on RDNA3.5.
GPTQ (Q4) is the fastest.


| Model                                    | Size   | Concurrency | Duration (s) | Req/s    | Output Tok/s | Total Tok/s | Mean TTFT (ms) | Mean TPOT (ms) |
|------------------------------------------|--------|-------------|--------------|----------|--------------|-------------|----------------|----------------|
| meta-llama/Llama-2-7b-chat-hf (FP16)     | 13.5GB | c=16        | 155.97       | 0.41     | 72.17        | 186.35      | 871.62         | 162.64         |
| RedHatAI/Llama-2-7b-chat-quantized.w8a8  | 6.6GB  | c=16        | 407.27       | 0.16     | 28.66        | 72.38       | 2213.22        | 361.97         |
| TheBloke/Llama-2-7B-Chat-GPTQ            | 3.7GB  | c=16        | **85.72**    | **0.75** | **148.91**   | **356.67**  | **787.02**     | **90.31**      |


### vLLM vs llama.cpp
OK, but how does vLLM compare to llama.cpp? My expectation is that at c=1 llama.cpp would obviously perform better but that at higher concurrency maybe not? That seems to be the general result.
Note, not all requests completed for llama.cpp (36/64) - maybe a context issue, take these as just an eyball.

#### c=1 Comparison
At c=1, even FP16 vs FP16 llama.cpp is  a fair bit faster

| Backend   | Model/Quant                           | Duration (s) | Req/s    | Output Tok/s | Total Tok/s | Mean TTFT (ms) | Mean TPOT (ms) |
|-----------|---------------------------------------|--------------|----------|--------------|-------------|----------------|----------------|
| vLLM      | meta-llama/Llama-2-7b-chat-hf (FP16)  | 1295.40      | 0.05     | 8.61         | 22.36       | **121.23**     | 115.88         |
| llama.cpp | Llama-2-7b-chat-hf.f16.gguf           | 412.23       | 0.09     | 12.75        | 16.00       | 200.52         | **76.32**      |
| llama.cpp | llama-2-7b.Q8_0.gguf                  | **237.27**   | **0.15** | **24.74**    | **30.38**   | 146.13         | 39.16          |
| llama.cpp | llama-2-7b.Q4_K_M.gguf                | 237.19       | **0.15** | **24.75**    | **30.39**   | **145.18**     | **39.09**      |

#### c=16 FP16 Comparison
At c=16 for FP16, vLLM and llama.cpp are pretty close.

| Backend   | Model/Quant                           | Duration (s) | Req/s    | Output Tok/s | Total Tok/s | Mean TTFT (ms) | Mean TPOT (ms) |
|-----------|---------------------------------------|--------------|----------|--------------|-------------|----------------|----------------|
| vLLM      | meta-llama/Llama-2-7b-chat-hf (FP16)  | 155.97       | 0.41     | **72.17**    | **186.35**  | 871.62         | 162.64         |
| llama.cpp | Llama-2-7b-chat-hf.f16.gguf           | **79.95**    | **0.45** | 66.85        | 83.60       | **497.90**     | **175.86**     |

#### c=16 8-bit Quantization Comparison
At 8-bit, of course, w/ how badly W8A8-INT8 performs, llama.cpp crushes vLLM.

| Backend   | Model/Quant                          | Duration (s) | Req/s    | Output Tok/s | Total Tok/s | Mean TTFT (ms) | Mean TPOT (ms) |
|-----------|--------------------------------------|--------------|----------|--------------|-------------|----------------|----------------|
| vLLM      | RedHatAI/Llama-2-7b-chat-quantized.w8a8 | 407.27   | 0.16     | 28.66        | 72.38       | 2213.22        | 361.97         |
| llama.cpp | llama-2-7b.Q8_0.gguf                 | **35.48**    | **1.01** | **161.34**   | **199.08**  | **395.97**     | **71.05**      |

#### c=16 4-bit Quantization Comparison
While llama.cpp seems to pull ahead for Q4, I think it's close enough that you'd probably want to do a careful review.

| Backend   | Model/Quant                   | Duration (s) | Req/s    | Output Tok/s | Total Tok/s | Mean TTFT (ms) | Mean TPOT (ms) |
|-----------|-------------------------------|--------------|----------|--------------|-------------|----------------|----------------|
| vLLM      | TheBloke/Llama-2-7B-Chat-GPTQ | 85.72        | 0.75     | 148.91       | **356.67**  | 787.02         | 90.31          |
| llama.cpp | llama-2-7b.Q4_K_M.gguf        | **32.85**    | **1.10** | **178.68**   | 219.44      | **381.53**     | **64.94**      |


### Raw Results
If you want to dig in.

```
### meta-llama/Llama-2-7b-chat-hf (FP16 - 13.5GB)
# c=1
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7f0183dcde40>, seed=0, num_prompts=64, dataset_name='sharegpt', no_stream=False, dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', custom_output_len=256, custom_skip_chat_template=False, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, random_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_output_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, endpoint_type='openai', label=None, backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', max_concurrency=1, model='meta-llama/Llama-2-7b-chat-hf', tokenizer=None, use_beam_search=False, logprobs=None, request_rate=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |                                                                                                                                                                                                                                           | 00:15 elapsed, 1781:16:25 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 1
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 64/64 [21:35<00:00, 20.24s/it]
============ Serving Benchmark Result ============
Successful requests:                     64
Maximum request concurrency:             1
Benchmark duration (s):                  1295.40
Total input tokens:                      17809
Total generated tokens:                  11150
Request throughput (req/s):              0.05
Output token throughput (tok/s):         8.61
Total Token throughput (tok/s):          22.36
---------------Time to First Token----------------
Mean TTFT (ms):                          121.23
Median TTFT (ms):                        120.41
P99 TTFT (ms):                           129.47
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          115.88
Median TPOT (ms):                        115.68
P99 TPOT (ms):                           118.66
---------------Inter-token Latency----------------
Mean ITL (ms):                           116.15
Median ITL (ms):                         115.69
P99 ITL (ms):                            121.45
==================================================

# c=8
Namespace(backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', dataset_name='sharegpt', dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', no_stream=False, max_concurrency=8, model='meta-llama/Llama-2-7b-chat-hf', tokenizer=None, use_beam_search=False, num_prompts=64, logprobs=None, request_rate=inf, burstiness=1.0, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', custom_output_len=256, custom_skip_chat_template=False, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, hf_subset=None, hf_split=None, hf_output_len=None, top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None)
Starting initial single prompt test run...
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf RPS.
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 8
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 64/64 [03:45<00:00,  3.53s/it]
============ Serving Benchmark Result ============
Successful requests:                     64
Maximum request concurrency:             8
Benchmark duration (s):                  225.80
Total input tokens:                      17809
Total generated tokens:                  11157
Request throughput (req/s):              0.28
Output token throughput (tok/s):         49.41
Total Token throughput (tok/s):          128.28
---------------Time to First Token----------------
Mean TTFT (ms):                          536.74
Median TTFT (ms):                        484.12
P99 TTFT (ms):                           1420.17
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          135.23
Median TPOT (ms):                        135.23
P99 TPOT (ms):                           156.51
---------------Inter-token Latency----------------
Mean ITL (ms):                           135.60
Median ITL (ms):                         129.41
P99 ITL (ms):                            431.69
==================================================

# c=16
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7f0b51dade40>, seed=0, num_prompts=64, dataset_name='sharegpt', no_stream=False, dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', custom_output_len=256, custom_skip_chat_template=False, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, random_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_output_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, endpoint_type='openai', label=None, backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', max_concurrency=16, model='meta-llama/Llama-2-7b-chat-hf', tokenizer=None, use_beam_search=False, logprobs=None, request_rate=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |                                                                                                                                                                                                                                           | 00:15 elapsed, 1772:25:14 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 16
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 64/64 [02:35<00:00,  2.44s/it]
============ Serving Benchmark Result ============
Successful requests:                     64
Maximum request concurrency:             16
Benchmark duration (s):                  155.97
Total input tokens:                      17809
Total generated tokens:                  11256
Request throughput (req/s):              0.41
Output token throughput (tok/s):         72.17
Total Token throughput (tok/s):          186.35
---------------Time to First Token----------------
Mean TTFT (ms):                          871.62
Median TTFT (ms):                        593.93
P99 TTFT (ms):                           2771.88
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          162.64
Median TPOT (ms):                        159.98
P99 TPOT (ms):                           254.94
---------------Inter-token Latency----------------
Mean ITL (ms):                           158.53
Median ITL (ms):                         146.81
P99 ITL (ms):                            671.18
==================================================



# RedHatAI/Llama-2-7b-chat-quantized.w8a8 (6.6GB) c=16
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7f984baf5e40>, seed=0, num_prompts=64, dataset_name='sharegpt', no_stream=False, dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', custom_output_len=256, custom_skip_chat_template=False, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, random_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_output_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, endpoint_type='openai', label=None, backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', max_concurrency=16, model='RedHatAI/Llama-2-7b-chat-quantized.w8a8', tokenizer=None, use_beam_search=False, logprobs=None, request_rate=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |                                                                                                                                                                                                                                           | 00:38 elapsed, 4890:59:01 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 16
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 64/64 [06:47<00:00,  6.36s/it]
============ Serving Benchmark Result ============
Successful requests:                     64
Maximum request concurrency:             16
Benchmark duration (s):                  407.27
Total input tokens:                      17809
Total generated tokens:                  11671
Request throughput (req/s):              0.16
Output token throughput (tok/s):         28.66
Total Token throughput (tok/s):          72.38
---------------Time to First Token----------------
Mean TTFT (ms):                          2213.22
Median TTFT (ms):                        1386.26
P99 TTFT (ms):                           6955.75
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          361.97
Median TPOT (ms):                        351.24
P99 TPOT (ms):                           610.30
---------------Inter-token Latency----------------
Mean ITL (ms):                           349.49
Median ITL (ms):                         320.35
P99 ITL (ms):                            1538.99
==================================================


# TheBloke/Llama-2-7B-Chat-GPTQ (3.7GB)
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7f35fb055e40>, seed=0, num_prompts=64, dataset_name='sharegpt', no_stream=False, dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', custom_output_len=256, custom_skip_chat_template=False, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, random_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_output_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, endpoint_type='openai', label=None, backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', max_concurrency=16, model='TheBloke/Llama-2-7B-Chat-GPTQ', tokenizer=None, use_beam_search=False, logprobs=None, request_rate=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file you can ignore this message.
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |                                                                                                                                                                                                                                            | 00:04 elapsed, 677:18:01 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 16
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 64/64 [01:25<00:00,  1.34s/it]
============ Serving Benchmark Result ============
Successful requests:                     64
Maximum request concurrency:             16
Benchmark duration (s):                  85.72
Total input tokens:                      17809
Total generated tokens:                  12765
Request throughput (req/s):              0.75
Output token throughput (tok/s):         148.91
Total Token throughput (tok/s):          356.67
---------------Time to First Token----------------
Mean TTFT (ms):                          787.02
Median TTFT (ms):                        462.31
P99 TTFT (ms):                           2622.98
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          90.31
Median TPOT (ms):                        86.06
P99 TPOT (ms):                           219.29
---------------Inter-token Latency----------------
Mean ITL (ms):                           83.50
Median ITL (ms):                         71.74
P99 ITL (ms):                            608.18
==================================================


# fp16 llama.cpp - c=1
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7f79b7e95e40>, seed=0, num_prompts=64, dataset_name='sharegpt', no_stream=False, dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', custom_output_len=256, custom_skip_chat_template=False, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, random_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_output_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, endpoint_type='openai', label=None, backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', max_concurrency=1, model='/models/gguf/Llama-2-7b-chat-hf.f16.gguf', tokenizer=None, use_beam_search=False, logprobs=None, request_rate=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file you can ignore this message.
Merges were not in checkpoint, building merges on the fly.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32000/32000 [00:15<00:00, 2059.72it/s]
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |                                                                                                                                                                                                                                           | 00:09 elapsed, 7075:55:31 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 1
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 64/64 [06:52<00:00,  6.44s/it]
============ Serving Benchmark Result ============
Successful requests:                     36
Maximum request concurrency:             1
Benchmark duration (s):                  412.23
Total input tokens:                      1339
Total generated tokens:                  5256
Request throughput (req/s):              0.09
Output token throughput (tok/s):         12.75
Total Token throughput (tok/s):          16.00
---------------Time to First Token----------------
Mean TTFT (ms):                          200.52
Median TTFT (ms):                        178.87
P99 TTFT (ms):                           418.39
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          76.32
Median TPOT (ms):                        77.69
P99 TPOT (ms):                           80.92
---------------Inter-token Latency----------------
Mean ITL (ms):                           77.38
Median ITL (ms):                         78.71
P99 ITL (ms):                            82.60
==================================================


# llama.cpp - c=16
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7fcd4b809e40>, seed=0, num_prompts=64, dataset_name='sharegpt', no_stream=False, dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', custom_output_len=256, custom_skip_chat_template=False, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, random_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_output_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, endpoint_type='openai', label=None, backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', max_concurrency=16, model='/models/gguf/Llama-2-7b-chat-hf.f16.gguf', tokenizer=None, use_beam_search=False, logprobs=None, request_rate=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file you can ignore this message.
Merges were not in checkpoint, building merges on the fly.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32000/32000 [00:15<00:00, 2055.37it/s]
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |                                                                                                                                                                                                                                           | 00:10 elapsed, 7834:26:28 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 16
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 64/64 [01:19<00:00,  1.25s/it]
============ Serving Benchmark Result ============
Successful requests:                     36
Maximum request concurrency:             16
Benchmark duration (s):                  79.95
Total input tokens:                      1339
Total generated tokens:                  5345
Request throughput (req/s):              0.45
Output token throughput (tok/s):         66.85
Total Token throughput (tok/s):          83.60
---------------Time to First Token----------------
Mean TTFT (ms):                          497.90
Median TTFT (ms):                        389.52
P99 TTFT (ms):                           800.17
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          175.86
Median TPOT (ms):                        177.96
P99 TPOT (ms):                           202.39
---------------Inter-token Latency----------------
Mean ITL (ms):                           175.29
Median ITL (ms):                         174.34
P99 ITL (ms):                            308.21
==================================================

# llama.cpp Q8_0 - c=1
🐟 ❯ vllm bench serve --dataset-name sharegpt --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 64 --model /models/gguf/llama-2-7b.Q8_0.gguf --max-concurrency 1
INFO 08-31 14:22:17 [__init__.py:242] Automatically detected platform rocm.
WARNING 08-31 14:22:17 [rocm.py:29] Failed to import from amdsmi with ModuleNotFoundError("No module named 'amdsmi'")
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7f2c83e71e40>, seed=0, num_prompts=64, dataset_name='sharegpt', no_stream=False, dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', custom_output_len=256, custom_skip_chat_template=False, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, random_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_output_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, endpoint_type='openai', label=None, backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', max_concurrency=1, model='/models/gguf/llama-2-7b.Q8_0.gguf', tokenizer=None, use_beam_search=False, logprobs=None, request_rate=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file you can ignore this message.
Merges were not in checkpoint, building merges on the fly.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32000/32000 [00:15<00:00, 2056.19it/s]
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |                                                                                                                                                                                                                                           | 00:05 elapsed, 3724:27:50 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 1
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 64/64 [03:57<00:00,  3.71s/it]
============ Serving Benchmark Result ============
Successful requests:                     36
Maximum request concurrency:             1
Benchmark duration (s):                  237.27
Total input tokens:                      1339
Total generated tokens:                  5870
Request throughput (req/s):              0.15
Output token throughput (tok/s):         24.74
Total Token throughput (tok/s):          30.38
---------------Time to First Token----------------
Mean TTFT (ms):                          146.13
Median TTFT (ms):                        86.05
P99 TTFT (ms):                           613.21
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          39.16
Median TPOT (ms):                        39.57
P99 TPOT (ms):                           40.81
---------------Inter-token Latency----------------
Mean ITL (ms):                           39.69
Median ITL (ms):                         39.49
P99 ITL (ms):                            42.24
==================================================


# llama.cpp Q8_0 - c=16
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7f4b84635e40>, seed=0, num_prompts=64, dataset_name='sharegpt', no_stream=False, dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', custom_output_len=256, custom_skip_chat_template=False, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, random_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_output_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, endpoint_type='openai', label=None, backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', max_concurrency=16, model='/models/gguf/llama-2-7b.Q8_0.gguf', tokenizer=None, use_beam_search=False, logprobs=None, request_rate=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file you can ignore this message.
Merges were not in checkpoint, building merges on the fly.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32000/32000 [00:15<00:00, 2065.12it/s]
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |                                                                                                                                                                                                                                           | 00:05 elapsed, 3851:58:05 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 16
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 64/64 [00:35<00:00,  1.80it/s]
============ Serving Benchmark Result ============
Successful requests:                     36
Maximum request concurrency:             16
Benchmark duration (s):                  35.48
Total input tokens:                      1339
Total generated tokens:                  5724
Request throughput (req/s):              1.01
Output token throughput (tok/s):         161.34
Total Token throughput (tok/s):          199.08
---------------Time to First Token----------------
Mean TTFT (ms):                          395.97
Median TTFT (ms):                        299.71
P99 TTFT (ms):                           937.76
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          71.05
Median TPOT (ms):                        69.53
P99 TPOT (ms):                           114.77
---------------Inter-token Latency----------------
Mean ITL (ms):                           70.61
Median ITL (ms):                         63.51
P99 ITL (ms):                            368.20
==================================================


# llama.cpp Q4_K_M - c=1
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7f636a6c9e40>, seed=0, num_prompts=64, dataset_name='sharegpt', no_stream=False, dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', custom_output_len=256, custom_skip_chat_template=False, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, random_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_output_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, endpoint_type='openai', label=None, backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', max_concurrency=1, model='/models/gguf/llama-2-7b.Q4_K_M.gguf', tokenizer=None, use_beam_search=False, logprobs=None, request_rate=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file you can ignore this message.
Merges were not in checkpoint, building merges on the fly.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32000/32000 [00:15<00:00, 2071.24it/s]
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |                                                                                                                                                                                                                                           | 00:05 elapsed, 3798:57:47 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 1
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 64/64 [03:57<00:00,  3.71s/it]
============ Serving Benchmark Result ============
Successful requests:                     36
Maximum request concurrency:             1
Benchmark duration (s):                  237.19
Total input tokens:                      1339
Total generated tokens:                  5870
Request throughput (req/s):              0.15
Output token throughput (tok/s):         24.75
Total Token throughput (tok/s):          30.39
---------------Time to First Token----------------
Mean TTFT (ms):                          145.18
Median TTFT (ms):                        88.03
P99 TTFT (ms):                           617.56
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          39.09
Median TPOT (ms):                        39.35
P99 TPOT (ms):                           41.09
---------------Inter-token Latency----------------
Mean ITL (ms):                           39.68
Median ITL (ms):                         40.23
P99 ITL (ms):                            42.31
==================================================


# llama.cpp Q4_K_M - c=16
Namespace(subparser='bench', bench_type='serve', dispatch_function=<function BenchmarkServingSubcommand.cmd at 0x7f33ec7b9e40>, seed=0, num_prompts=64, dataset_name='sharegpt', no_stream=False, dataset_path='./ShareGPT_V3_unfiltered_cleaned_split.json', custom_output_len=256, custom_skip_chat_template=False, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, random_prefix_len=0, random_batch_size=1, random_mm_base_items_per_request=1, random_mm_num_mm_items_range_ratio=0.0, random_mm_limit_mm_per_prompt={'image': 255, 'video': 0}, random_mm_bucket_config={(256, 256, 1): 0.5, (720, 1280, 1): 0.5, (720, 1280, 16): 0.0}, hf_subset=None, hf_split=None, hf_output_len=None, prefix_repetition_prefix_len=256, prefix_repetition_suffix_len=256, prefix_repetition_num_prefixes=10, prefix_repetition_output_len=128, endpoint_type='openai', label=None, backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', max_concurrency=16, model='/models/gguf/llama-2-7b.Q4_K_M.gguf', tokenizer=None, use_beam_search=False, logprobs=None, request_rate=inf, burstiness=1.0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, append_result=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=False, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None, ramp_up_strategy=None, ramp_up_start_rps=None, ramp_up_end_rps=None, ready_check_timeout_sec=600)
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file you can ignore this message.
Merges were not in checkpoint, building merges on the fly.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32000/32000 [00:15<00:00, 2069.57it/s]
Starting initial single prompt test run...
Waiting for endpoint to become up in 600 seconds
 |                                                                                                                                                                                                                                           | 00:04 elapsed, 2970:11:54 remaining
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 16
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 64/64 [00:32<00:00,  1.95it/s]
============ Serving Benchmark Result ============
Successful requests:                     36
Maximum request concurrency:             16
Benchmark duration (s):                  32.85
Total input tokens:                      1339
Total generated tokens:                  5870
Request throughput (req/s):              1.10
Output token throughput (tok/s):         178.68
Total Token throughput (tok/s):          219.44
---------------Time to First Token----------------
Mean TTFT (ms):                          381.53
Median TTFT (ms):                        294.11
P99 TTFT (ms):                           909.76
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          64.94
Median TPOT (ms):                        62.65
P99 TPOT (ms):                           109.95
---------------Inter-token Latency----------------
Mean ITL (ms):                           64.65
Median ITL (ms):                         57.15
P99 ITL (ms):                            352.53
==================================================



```

Just for fun llama-bench:
```
🐟 ❯ build/bin/llama-bench -fa 1 -m /models/gguf/llama-2-7b.Q4_0.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 |  1 |           pp512 |        878.31 ± 4.11 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 |  1 |           tg128 |         39.33 ± 0.00 |

build: 4d74393bc (6327)

🐟 ❯ ROCBLAS_USE_HIPBLASLT=1 build/bin/llama-bench -fa 1 -m /models/gguf/llama-2-7b.Q4_0.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 |  1 |           pp512 |        972.75 ± 6.99 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | ROCm       |  99 |  1 |           tg128 |         39.23 ± 0.06 |

build: 4d74393bc (6327)

🐟 ❯ build/bin/llama-bench -fa 1 -m /models/gguf/llama-2-7b.Q4_K_M.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 7B Q4_K - Medium         |   3.80 GiB |     6.74 B | ROCm       |  99 |  1 |           pp512 |        873.74 ± 3.75 |
| llama 7B Q4_K - Medium         |   3.80 GiB |     6.74 B | ROCm       |  99 |  1 |           tg128 |         33.28 ± 0.01 |

build: 4d74393bc (6327)

🐟 ❯ ROCBLAS_USE_HIPBLASLT=1 build/bin/llama-bench -fa 1 -m /models/gguf/llama-2-7b.Q4_K_M.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 7B Q4_K - Medium         |   3.80 GiB |     6.74 B | ROCm       |  99 |  1 |           pp512 |        958.96 ± 6.76 |
| llama 7B Q4_K - Medium         |   3.80 GiB |     6.74 B | ROCm       |  99 |  1 |           tg128 |         33.30 ± 0.02 |

build: 4d74393bc (6327)

🐟 ❯ build/bin/llama-bench -fa 1 -m /models/gguf/llama-2-7b.Q8_0.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 7B Q8_0                  |   6.67 GiB |     6.74 B | ROCm       |  99 |  1 |           pp512 |        859.40 ± 1.58 |
| llama 7B Q8_0                  |   6.67 GiB |     6.74 B | ROCm       |  99 |  1 |           tg128 |         25.25 ± 0.01 |

build: 4d74393bc (6327)

🐟 ❯ ROCBLAS_USE_HIPBLASLT=1 build/bin/llama-bench -fa 1 -m /models/gguf/llama-2-7b.Q8_0.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 7B Q8_0                  |   6.67 GiB |     6.74 B | ROCm       |  99 |  1 |           pp512 |        944.45 ± 3.76 |
| llama 7B Q8_0                  |   6.67 GiB |     6.74 B | ROCm       |  99 |  1 |           tg128 |         25.24 ± 0.02 |

build: 4d74393bc (6327)

🐟 ❯ build/bin/llama-bench -fa 1 -m /models/gguf/Llama-2-7b-chat-hf.f16.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 7B F16                   |  12.55 GiB |     6.74 B | ROCm       |  99 |  1 |           pp512 |       1056.97 ± 5.37 |
| llama 7B F16                   |  12.55 GiB |     6.74 B | ROCm       |  99 |  1 |           tg128 |         13.91 ± 0.02 |

build: 4d74393bc (6327)

🐟 ❯ ROCBLAS_USE_HIPBLASLT=1 build/bin/llama-bench -fa 1 -m /models/gguf/Llama-2-7b-chat-hf.f16.gguf
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| llama 7B F16                   |  12.55 GiB |     6.74 B | ROCm       |  99 |  1 |           pp512 |       1189.21 ± 4.65 |
| llama 7B F16                   |  12.55 GiB |     6.74 B | ROCm       |  99 |  1 |           tg128 |         13.93 ± 0.01 |

build: 4d74393bc (6327)
```
