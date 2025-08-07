Use the same build

## RPC Server
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

## Testing

- You will want to mmap otherwise your main node will use more memory
- You only specify the other servers now even w/ llama-bench?
- use '-v' to get visibility

"load_tensors: tensor 'token_embd.weight' (q4_K) (and 272 others) cannot be used with preferred buffer type Vulkan_Host, using CPU instead"


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

### gpt-oss 120b
llama-bench -> rpc-server 10577 M 
17268 M
17268 M
17268 M
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

### Tulu 3 405B

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

# DeepSeek R1 UD  Q2_K_XL


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

# HIP 
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

Interesting, much more even w/ HIP...
Also loads faster, runs slower?


# DeepSeek R1 Q4_K_M
Crashes with Vulkan (OOM), uneven allocation. Works with ROCm (--mmap 0 for loading speed, doesn't use more memory):

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
- this *barely* fits - you should try Q4_K_XL or better, Q3_K_XL, and make sure you're at the max (120GB+) for your RPC memory limits
