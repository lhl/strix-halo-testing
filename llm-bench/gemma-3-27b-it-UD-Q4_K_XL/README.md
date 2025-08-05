# Benchmark Results
| backend           | hipblaslt   | -fa   | -b   |   pp512 |   tg128 | max_mem   |
|-------------------|-------------|-------|------|---------|---------|-----------|
| llama.cpp-hip     |             |       |      |     nan |     nan | 18008     |
| llama.cpp-hip     | 1           |       |      |     nan |     nan | 18039     |
| llama.cpp-hip     |             | -fa 1 |      |     nan |     nan | 18037     |
| llama.cpp-hip     | 1           | -fa 1 |      |     nan |     nan | 18069     |
| llama.cpp-rocwmma |             |       |      |     nan |     nan | **18006** |
| llama.cpp-rocwmma | 1           |       |      |     nan |     nan | 18038     |
| llama.cpp-rocwmma |             | -fa 1 |      |     nan |     nan | 18036     |
| llama.cpp-rocwmma | 1           | -fa 1 |      |     nan |     nan | 18068     |
| llama.cpp-vulkan  |             |       |      |     nan |     nan | 18726     |
| llama.cpp-vulkan  |             | -fa 1 |      |     nan |     nan | 18659     |
## Performance Charts

### Tokens/s Performance
![PP Tokens/s](pp_tokens_per_sec.png)
![TG Tokens/s](tg_tokens_per_sec.png)

### Memory Usage
![PP VRAM](pp_vram_peak_mib.png)
![TG VRAM](tg_vram_peak_mib.png)

## Detailed Sweeps

### PP sweep
| backend           | hipblaslt   | -fa   | -b   |   1 |   2 |   4 |   8 |   16 |   32 |   64 |   128 |   256 |   512 |   1024 |   2048 |   4096 |
|-------------------|-------------|-------|------|-----|-----|-----|-----|------|------|------|-------|-------|-------|--------|--------|--------|
| llama.cpp-hip     |             |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan |
| llama.cpp-hip     | 1           |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan |
| llama.cpp-hip     |             | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan |
| llama.cpp-hip     | 1           | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan |
| llama.cpp-rocwmma |             |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan |
| llama.cpp-rocwmma | 1           |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan |
| llama.cpp-rocwmma |             | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan |
| llama.cpp-rocwmma | 1           | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan |
| llama.cpp-vulkan  |             |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan |
| llama.cpp-vulkan  |             | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan |
### TG sweep
| backend           | hipblaslt   | -fa   | -b   |   1 |   2 |   4 |   8 |   16 |   32 |   64 |   128 |   256 |   512 |   1024 |   2048 |   4096 |
|-------------------|-------------|-------|------|-----|-----|-----|-----|------|------|------|-------|-------|-------|--------|--------|--------|
| llama.cpp-hip     |             |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan |
| llama.cpp-hip     | 1           |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan |
| llama.cpp-hip     |             | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan |
| llama.cpp-hip     | 1           | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan |
| llama.cpp-rocwmma |             |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan |
| llama.cpp-rocwmma | 1           |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan |
| llama.cpp-rocwmma |             | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan |
| llama.cpp-rocwmma | 1           | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan |
| llama.cpp-vulkan  |             |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan |
| llama.cpp-vulkan  |             | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan |