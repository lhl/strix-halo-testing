# Benchmark Results
| backend           | hipblaslt   | -fa   | -b   |   pp512 |   tg128 | max_mem   |
|-------------------|-------------|-------|------|---------|---------|-----------|
| llama.cpp-hip     |             |       |      |     nan |     nan | 15889     |
| llama.cpp-hip     | 1           |       |      |     nan |     nan | 15921     |
| llama.cpp-hip     |             | -fa 1 |      |     nan |     nan | 15647     |
| llama.cpp-hip     | 1           | -fa 1 |      |     nan |     nan | 15679     |
| llama.cpp-rocwmma |             |       |      |     nan |     nan | 15889     |
| llama.cpp-rocwmma | 1           |       |      |     nan |     nan | 15921     |
| llama.cpp-rocwmma |             | -fa 1 |      |     nan |     nan | 15647     |
| llama.cpp-rocwmma | 1           | -fa 1 |      |     nan |     nan | 15679     |
| llama.cpp-vulkan  |             |       |      |     nan |     nan | 15936     |
| llama.cpp-vulkan  |             | -fa 1 |      |     nan |     nan | **15582** |
## Performance Charts

### Tokens/s Performance
![PP Tokens/s](pp_tokens_per_sec.png)
![TG Tokens/s](tg_tokens_per_sec.png)

### Memory Usage
![PP VRAM](pp_vram_peak_mib.png)
![TG VRAM](tg_vram_peak_mib.png)

## Detailed Sweeps

### PP sweep
| backend           | hipblaslt   | -fa   | -b   |   1 |   2 |   4 |   8 |   16 |   32 |   64 |   128 |   256 |   512 |   1024 |   2048 |   4096 | 8192           |
|-------------------|-------------|-------|------|-----|-----|-----|-----|------|------|------|-------|-------|-------|--------|--------|--------|----------------|
| llama.cpp-hip     |             |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 50.821442      |
| llama.cpp-hip     | 1           |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | **209.958664** |
| llama.cpp-hip     |             | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 40.299797      |
| llama.cpp-hip     | 1           | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 97.969096      |
| llama.cpp-rocwmma |             |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 49.112958      |
| llama.cpp-rocwmma | 1           |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 205.103784     |
| llama.cpp-rocwmma |             | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 37.462679      |
| llama.cpp-rocwmma | 1           | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 97.617684      |
| llama.cpp-vulkan  |             |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 109.310075     |
| llama.cpp-vulkan  |             | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 100.512613     |
### TG sweep
| backend           | hipblaslt   | -fa   | -b   |   1 |   2 |   4 |   8 |   16 |   32 |   64 |   128 |   256 |   512 |   1024 |   2048 |   4096 | 8192          |
|-------------------|-------------|-------|------|-----|-----|-----|-----|------|------|------|-------|-------|-------|--------|--------|--------|---------------|
| llama.cpp-hip     |             |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 10.034434     |
| llama.cpp-hip     | 1           |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 10.035445     |
| llama.cpp-hip     |             | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 10.722624     |
| llama.cpp-hip     | 1           | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 10.734519     |
| llama.cpp-rocwmma |             |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 10.032416     |
| llama.cpp-rocwmma | 1           |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 10.033583     |
| llama.cpp-rocwmma |             | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 10.714311     |
| llama.cpp-rocwmma | 1           | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 10.705802     |
| llama.cpp-vulkan  |             |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 11.912339     |
| llama.cpp-vulkan  |             | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | **13.036002** |