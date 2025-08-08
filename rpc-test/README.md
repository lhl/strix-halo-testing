# llama.cpp RPC Tests

For maximum compatibility you should use the same build# for all your RPC servers (add `-DGGML_RPC=ON` to your `cmake`)

In theory, each build can be for whatever backend (Vulkan, HIP, CUDA, SYCL, etc) you want.

## Running
One each machine, run something like:

```
$ build/bin/rpc-server -p 50053 -H 0.0.0.0 -m 112000 -c
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = Radeon 8060S Graphics (AMD open-source driver) | uma: 1 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 32768 | int dot: 1 | matrix cores: KHR_coopmat

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING: Host ('0.0.0.0') is != '127.0.0.1'
         Never expose the RPC server to an open network!
         This is an experimental feature and is not secure!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

create_backend: using Vulkan0 backend
Starting RPC server v2.0.0
  endpoint       : 0.0.0.0:50053
  local cache    : /home/lhl/.cache/llama.cpp/rpc/
  backend memory : 112000 MB
```

The rpc-server takes minimal memory, so you can run multiple servers on different ports.

The `-c` is important - it will cache tensors so they don't have to be constantly transferred every time.

## RPC Scaling 

Here's a simple test that shows how adding more nodes affects inference speed. Prompt processing mostly takes a one time hit, but token generation speed decreases pretty linearly with additional nodes.

In general, you should use the least number of nodes as possible (including avoiding RPC entirely when possible).

### 1
```
❯ time build/bin/llama-bench -m /models/gguf/gpt-oss-120b-F16.gguf                                                                         (base)
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = Radeon 8060S Graphics (AMD open-source driver) | uma: 1 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 32768 | int dot: 1 | matrix cores: KHR_coopmat
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss ?B F16                 |  60.87 GiB |   116.83 B | Vulkan,RPC |  99 |           pp512 |        404.67 ± 3.65 |
| gpt-oss ?B F16                 |  60.87 GiB |   116.83 B | Vulkan,RPC |  99 |           tg128 |         33.50 ± 0.03 |

build: 9515c613 (6097)

________________________________________________________
Executed in   53.52 secs    fish           external
   usr time   13.05 secs    0.39 millis   13.05 secs
   sys time    8.93 secs    1.32 millis    8.92 secs
```

### 2 
```
❯ time build/bin/llama-bench --rpc 192.168.128.12:50053 -m /models/gguf/gpt-oss-120b-F16.gguf                                              (base)
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = Radeon 8060S Graphics (AMD open-source driver) | uma: 1 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 32768 | int dot: 1 | matrix cores: KHR_coopmat
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss ?B F16                 |  60.87 GiB |   116.83 B | Vulkan,RPC |  99 |           pp512 |        374.82 ± 3.68 |
| gpt-oss ?B F16                 |  60.87 GiB |   116.83 B | Vulkan,RPC |  99 |           tg128 |         28.87 ± 0.11 |

build: 9515c613 (6097)

________________________________________________________
Executed in  102.58 secs    fish           external
   usr time   51.32 secs  804.00 micros   51.32 secs
   sys time    5.29 secs  528.00 micros    5.29 secs
```

### 3
```
❯ time build/bin/llama-bench --rpc 192.168.128.12:50053,192.168.128.13:50053 -m /models/gguf/gpt-oss-120b-F16.gguf                         (base)
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = Radeon 8060S Graphics (AMD open-source driver) | uma: 1 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 32768 | int dot: 1 | matrix cores: KHR_coopmat
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss ?B F16                 |  60.87 GiB |   116.83 B | Vulkan,RPC |  99 |           pp512 |        363.54 ± 3.60 |
| gpt-oss ?B F16                 |  60.87 GiB |   116.83 B | Vulkan,RPC |  99 |           tg128 |         23.02 ± 0.10 |

build: 9515c613 (6097)

________________________________________________________
Executed in  119.56 secs    fish           external
   usr time   52.60 secs  523.00 micros   52.60 secs
   sys time    5.29 secs  347.00 micros    5.29 secs
```

### 4
```
❯ time build/bin/llama-bench --rpc 192.168.128.12:50053,192.168.128.13:50053,192.168.128.14:50053 -m /models/gguf/gpt-oss-120b-F16.gguf    (base)
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = Radeon 8060S Graphics (AMD open-source driver) | uma: 1 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 32768 | int dot: 1 | matrix cores: KHR_coopmat
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss ?B F16                 |  60.87 GiB |   116.83 B | Vulkan,RPC |  99 |           pp512 |        368.13 ± 1.76 |
| gpt-oss ?B F16                 |  60.87 GiB |   116.83 B | Vulkan,RPC |  99 |           tg128 |         21.61 ± 0.05 |

build: 9515c613 (6097)

________________________________________________________
Executed in  100.66 secs    fish           external
   usr time   48.84 secs    0.00 micros   48.84 secs
   sys time    2.72 secs  947.00 micros    2.71 secs
```


## 4 Machine RPC Tests

- For Vulkan you will want to mmap otherwise your main node will use more memory and potentially OOM if you're squeezing a tight fit
- For HIP/ROCm you will absolutely want to disable mmap if you're using >50% of memory or you will die of old age before the model loads
- In the past you had to specify the local RPC server for llama-bench, but now you don't and it will automatically offload to the localhost server it looks like
- use '-v' to get visibility on what's going on

You can use small models for testing. You will [take a performance hit](https://github.com/lhl/strix-halo-testing/tree/main/llm-bench/gpt-oss-20b-F16#benchmark-results) as you distribute to more systems.

### gpt-oss 20b
```
❯ time build/bin/llama-bench --rpc 192.168.128.12:50053,192.168.128.13:50053,192.168.128.14:50053 -m /models/gguf/gpt-oss-20b-F16.gguf

ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = Radeon 8060S Graphics (AMD open-source driver) | uma: 1 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 32768 | int dot: 1 | matrix cores: KHR_coopmat
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss ?B F16                 |  12.83 GiB |    20.91 B | Vulkan,RPC |  99 |           pp512 |        853.47 ± 1.34 |
| gpt-oss ?B F16                 |  12.83 GiB |    20.91 B | Vulkan,RPC |  99 |           tg128 |         27.95 ± 0.23 |

build: 9515c613 (6097)

________________________________________________________
Executed in   37.91 secs    fish           external
   usr time   12.48 secs  353.00 micros   12.48 secs
   sys time    1.27 secs  458.00 micros    1.27 secs
```
- on a single machine pp512/tg128 is 1022/47

### gpt-oss 120b
One thing worth noting is that Vulkan does not to a great job distributing memory evenly. Here's how it distributes the layers (you should use `amdgpu_top --smi` to easily watch the processes and memory usage):

- cluster1: llama-bench -> rpc-server 10577 M 
- cluster2: 17268 M
- cluster3: 17268 M
- cluster4: 17268 M

```
❯ time build/bin/llama-bench --rpc 192.168.128.12:50053,192.168.128.13:50053,192.168.128.14:50053 -m /models/gguf/gpt-oss-120b-F16.gguf                                                        (base)
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = Radeon 8060S Graphics (AMD open-source driver) | uma: 1 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 32768 | int dot: 1 | matrix cores: KHR_coopmat
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss ?B F16                 |  60.87 GiB |   116.83 B | Vulkan,RPC |  99 |           pp512 |        367.39 ± 2.23 |
| gpt-oss ?B F16                 |  60.87 GiB |   116.83 B | Vulkan,RPC |  99 |           tg128 |         21.60 ± 0.13 |

build: 9515c613 (6097)

________________________________________________________
Executed in  138.95 secs    fish           external
   usr time   58.74 secs    0.00 micros   58.74 secs
   sys time    7.23 secs  723.00 micros    7.23 secs
```
- on a single machine pp512/tg128 is 431/34


### Tulu 3 405B
Run big dense models incredibly slow (There's zero point to this, IMO)
```
load_tensors: tensor 'token_embd.weight' (q4_K) (and 0 others) cannot be used with preferred buffer type Vulkan_Host, using CPU instead
load_tensors: offloading 126 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 127/127 layers to GPU
load_tensors:      Vulkan0 model buffer size = 43202.07 MiB
load_tensors: RPC[192.168.128.13:50053] model buffer size = 60767.75 MiB
load_tensors: RPC[192.168.128.14:50053] model buffer size = 62259.25 MiB
load_tensors: RPC[192.168.128.12:50053] model buffer size = 64445.50 MiB
load_tensors:          CPU model buffer size =  1127.32 MiB
| llama ?B Q4_K - Medium         | 226.37 GiB |   405.85 B | Vulkan,RPC | 999 |    0 |           pp512 |          5.51 ± 0.01 |
| llama ?B Q4_K - Medium         | 226.37 GiB |   405.85 B | Vulkan,RPC | 999 |    0 |           tg128 |          0.88 ± 0.00 |

build: 9515c613 (6097)

________________________________________________________
Executed in   31.48 mins    fish           external
   usr time  287.63 secs    0.00 millis  287.63 secs
   sys time   94.85 secs    2.17 millis   94.85 secs
```

### DeepSeek R1 UD Q2_K_XL
Run gigantic MoEs at ... well, still relatively unusable speeds...
```
load_tensors: tensor 'token_embd.weight' (q4_K) (and 0 others) cannot be used with preferred buffer type Vulkan_Host, using CPU instead
load_tensors: offloading 61 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 62/62 layers to GPU
load_tensors:      Vulkan0 model buffer size = 37403.07 MiB
load_tensors: RPC[192.168.128.13:50053] model buffer size = 62352.75 MiB
load_tensors: RPC[192.168.128.14:50053] model buffer size = 62352.75 MiB
load_tensors: RPC[192.168.128.12:50053] model buffer size = 53493.38 MiB
load_tensors:   CPU_Mapped model buffer size =   497.11 MiB
| deepseek2 671B Q2_K - Medium   | 211.03 GiB |   671.03 B | Vulkan,RPC |  99 |           pp512 |         40.20 ± 0.12 |
| deepseek2 671B Q2_K - Medium   | 211.03 GiB |   671.03 B | Vulkan,RPC |  99 |           tg128 |          7.98 ± 0.04 |
llama_perf_context_print:        load time =  760228.15 ms
llama_perf_context_print: prompt eval time =       0.00 ms /     1 tokens (    0.00 ms per token,      inf tokens per second)
llama_perf_context_print:        eval time =       0.00 ms /   641 runs   (    0.00 ms per token,      inf tokens per second)
llama_perf_context_print:       total time =  840390.42 ms /   642 tokens
llama_perf_context_print:    graphs reused =        621

build: 9515c613 (6097)

________________________________________________________
Executed in  841.13 secs    fish           external
   usr time  225.48 secs    0.44 millis  225.48 secs
   sys time  103.04 secs    2.40 millis  103.04 secs
```

### DeepSeek R1 UD Q2_K_XL - HIP
Now here's where we switch to HIP - the important thing to note is the memory distribution. Using HIP is significantly better when it comes to evenly distributing memory across nodes. Nope, I have no idea why.

```
❯ time build/bin/llama-bench --rpc 192.168.128.12:50054,192.168.128.13:50054,192.168.128.14:50054 -m /models/gguf/DeepSeek-R1-UD-Q2_K_XL/DeepSeek-R1-UD-Q2_K_XL-00001-of-00005.gguf -v

load_tensors: tensor 'token_embd.weight' (q4_K) (and 0 others) cannot be used with preferred buffer type ROCm_Host, using CPU instead
load_tensors: offloading 61 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 62/62 layers to GPU
load_tensors: RPC[192.168.128.12:50054] model buffer size = 49825.57 MiB
load_tensors: RPC[192.168.128.13:50054] model buffer size = 58684.95 MiB
load_tensors: RPC[192.168.128.14:50054] model buffer size = 58684.95 MiB
load_tensors:        ROCm0 model buffer size = 48406.50 MiB
load_tensors:   CPU_Mapped model buffer size =   497.11 MiB

| deepseek2 671B Q2_K - Medium   | 211.03 GiB |   671.03 B | ROCm,RPC   |  99 |           pp512 |         35.23 ± 0.13 |
| deepseek2 671B Q2_K - Medium   | 211.03 GiB |   671.03 B | ROCm,RPC   |  99 |           tg128 |          7.87 ± 0.01 |

build: 9a963895 (6110)

________________________________________________________
Executed in  594.00 secs    fish           external
   usr time  205.45 secs    0.41 millis  205.45 secs
   sys time   67.55 secs    1.32 millis   67.55 secs
```
- it also surprisingly loads faster?
- note mmap is on, but we're not yet crossing the 50% threshold


### DeepSeek R1 Q4_K_M

In theory this should fit, but this is where we crash w/ Vulkan (OOM on load) due to Vulkan's uneven allocation.

With ROCm we need to `--mmap 0` for loading speed, and that's fine it loads, but it still ooms (after loading) at `pp512`. Here are `pp128` numbers. Even with a 4X cluster, there's barely enough memory left over with context over. You could increase memory limits a bit and the Q4_K_XL is a bit smaller, but you're probably better off with the Q3_K_XL if you want to actually use it, although... it'll be slow.

```
❯ time build/bin/llama-bench --rpc 192.168.128.12:50054,192.168.128.13:50054,192.168.128.14:50054 -m /models/gguf/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00011.gguf -v --mmap 0 -p 128
load_tensors: tensor 'token_embd.weight' (q4_K) (and 0 others) cannot be used with preferred buffer type ROCm_Host, using CPU instead
load_tensors: offloading 61 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 62/62 layers to GPU
load_tensors: RPC[192.168.128.12:50054] model buffer size = 87857.67 MiB
load_tensors: RPC[192.168.128.13:50054] model buffer size = 103503.00 MiB
load_tensors: RPC[192.168.128.14:50054] model buffer size = 103503.00 MiB
load_tensors:        ROCm0 model buffer size = 90328.85 MiB
load_tensors:          CPU model buffer size =   497.11 MiB
load_all_data: device RPC[192.168.128.12:50054] does not support async, host buffers or events
......................load_all_data: device RPC[192.168.128.13:50054] does not support async, host buffers or events
...........................load_all_data: device RPC[192.168.128.14:50054] does not support async, host buffers or events
...........................load_all_data: using async uploads for device ROCm0, buffer type ROCm0, backend ROCm0
.......................load_all_data: no device found for buffer type CPU for async uploads
.
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | ROCm,RPC   |  99 |    0 |           pp128 |         27.38 ± 0.13 |
| deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | ROCm,RPC   |  99 |    0 |           tg128 |          6.66 ± 0.00 |

build: 9a963895 (6110)

________________________________________________________
Executed in  581.31 secs    fish           external
   usr time  287.49 secs    0.00 micros  287.49 secs
   sys time   73.37 secs  948.00 micros   73.37 secs

```

### Qwen3-Coder-480B-A35B-Instruct
This *should* work... but it doesn't.

Here's the problem and a patch for RPC and some big/wide models having issues (like Qwen 3 235B+): https://github.com/ggml-org/llama.cpp/issues/15055#issuecomment-3165296254

```
❯ time build/bin/llama-bench --rpc 192.168.128.12:50054,192.168.128.13:50054,192.168.128.14:50054 -m ~/Qwen3-Coder-480B-A35B-Instruct-GGUF/UD-Q6_K_XL/Qwen3-Coder-480B-A35B-Instruct-UD-Q6_K_XL-00001-of-00009.gguf  -v --mmap 0 -fa 1
load_tensors: offloaded 63/63 layers to GPU
load_tensors: RPC[192.168.128.12:50054] model buffer size = 103327.52 MiB
load_tensors: RPC[192.168.128.13:50054] model buffer size = 96933.77 MiB
load_tensors: RPC[192.168.128.14:50054] model buffer size = 98677.52 MiB
load_tensors:        ROCm0 model buffer size = 89250.46 MiB
load_tensors:    ROCm_Host model buffer size =   945.89 MiB
load_all_data: device RPC[192.168.128.12:50054] does not support async, host buffers or events
..........................load_all_data: device RPC[192.168.128.13:50054] does not support async, host buffers or events
........................load_all_data: device RPC[192.168.128.14:50054] does not support async, host buffers or events
..........................load_all_data: using async uploads for device ROCm0, buffer type ROCm0, backend ROCm0
.......................load_all_data: buffer type ROCm_Host is not the default buffer type for device ROCm0 for async uploads
...
llama_context: RPC[192.168.128.12:50054] compute buffer size =   265.51 MiB
llama_context: RPC[192.168.128.13:50054] compute buffer size =   264.51 MiB
llama_context: RPC[192.168.128.14:50054] compute buffer size =   264.51 MiB
llama_context:      ROCm0 compute buffer size =   308.75 MiB
llama_context:        CPU compute buffer size =     1.01 MiB
llama_context: graph nodes  = 3851
llama_context: graph splits = 5
attach_threadpool: call
set_n_threads: n_threads = 16, n_threads_batch = 16
Kernel Name: _ZL23flash_attn_tile_ext_f32ILi128ELi32ELi8ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiiiiiiiiiiiiiliiliiiiil
VGPU=0x559c73a321f0 SWq=0x7f7570e37000, HWq=0x7f745c100000, id=2
        Dispatch Header = 0xb02 (type=2, barrier=1, acquire=1, release=1), setup=0
        grid=[512, 8, 96], workgroup=[32, 8, 1]
        private_seg_size=2096, group_seg_size=36992
        kernel_obj=0x7f755e1ca6c0, kernarg_address=0x0x7f5e52601c00
        completion_signal=0x0, correlation_id=0
        rptr=22, wptr=33
:0:rocdevice.cpp            :3594: 34648731232 us:  Callback: Queue 0x7f745c100000 aborting with error : HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception. code: 0x1016

________________________________________________________
Executed in   17.06 mins    fish           external
   usr time  304.02 secs    0.05 millis  304.02 secs
   sys time  165.38 secs    1.03 millis  165.38 secs

fish: Job 1, 'time build/bin/llama-bench --rp…' terminated by signal SIGABRT (Abort)
```
