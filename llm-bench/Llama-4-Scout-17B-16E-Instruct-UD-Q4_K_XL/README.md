# Benchmark Results
| backend           | hipblaslt   | -fa   | -b     | pp512   | tg128   | max_mem   |
|-------------------|-------------|-------|--------|---------|---------|-----------|
| llama.cpp-hip     |             |       |        | -       | -       | **0**     |
| llama.cpp-hip     | 1           |       |        | -       | -       | **0**     |
| llama.cpp-hip     |             | -fa 1 |        | -       | -       | **0**     |
| llama.cpp-hip     | 1           | -fa 1 |        | -       | -       | **0**     |
| llama.cpp-rocwmma |             |       |        | -       | -       | **0**     |
| llama.cpp-rocwmma | 1           |       |        | -       | -       | **0**     |
| llama.cpp-rocwmma |             | -fa 1 |        | -       | -       | **0**     |
| llama.cpp-rocwmma | 1           | -fa 1 |        | -       | -       | **0**     |
| llama.cpp-vulkan  |             |       |        | -       | -       | **0**     |
| llama.cpp-vulkan  |             |       | -b 256 | -       | -       | **0**     |
| llama.cpp-vulkan  |             | -fa 1 |        | -       | -       | **0**     |
| llama.cpp-vulkan  |             | -fa 1 | -b 256 | -       | -       | **0**     |
## Performance Charts

### Tokens/s Performance
![PP Tokens/s](pp_tokens_per_sec.png)
![TG Tokens/s](tg_tokens_per_sec.png)

### Memory Usage
![PP VRAM](pp_vram_peak_mib.png)
![TG VRAM](tg_vram_peak_mib.png)

## Detailed Sweeps

### PP sweep
| backend           | hipblaslt   | -fa   | -b     | 1   | 2   | 4   | 8   | 16   | 32   | 64   | 128   | 256   | 512   | 1024   | 2048   | 4096   |
|-------------------|-------------|-------|--------|-----|-----|-----|-----|------|------|------|-------|-------|-------|--------|--------|--------|
| llama.cpp-hip     |             |       |        | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-hip     | 1           |       |        | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-hip     |             | -fa 1 |        | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-hip     | 1           | -fa 1 |        | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-rocwmma |             |       |        | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-rocwmma | 1           |       |        | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-rocwmma |             | -fa 1 |        | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-rocwmma | 1           | -fa 1 |        | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-vulkan  |             |       |        | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-vulkan  |             |       | -b 256 | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-vulkan  |             | -fa 1 |        | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-vulkan  |             | -fa 1 | -b 256 | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
### TG sweep
| backend           | hipblaslt   | -fa   | -b     | 1   | 2   | 4   | 8   | 16   | 32   | 64   | 128   | 256   | 512   | 1024   | 2048   | 4096   |
|-------------------|-------------|-------|--------|-----|-----|-----|-----|------|------|------|-------|-------|-------|--------|--------|--------|
| llama.cpp-hip     |             |       |        | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-hip     | 1           |       |        | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-hip     |             | -fa 1 |        | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-hip     | 1           | -fa 1 |        | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-rocwmma |             |       |        | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-rocwmma | 1           |       |        | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-rocwmma |             | -fa 1 |        | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-rocwmma | 1           | -fa 1 |        | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-vulkan  |             |       |        | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-vulkan  |             |       | -b 256 | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-vulkan  |             | -fa 1 |        | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |
| llama.cpp-vulkan  |             | -fa 1 | -b 256 | -   | -   | -   | -   | -    | -    | -    | -     | -     | -     | -      | -      | -      |