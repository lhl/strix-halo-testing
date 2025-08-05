# Benchmark Results
| backend           | hipblaslt   | -fa   | -b   |   pp512 |   tg128 | max_mem   |
|-------------------|-------------|-------|------|---------|---------|-----------|
| llama.cpp-hip     |             |       |      |     nan |     nan | 8143      |
| llama.cpp-hip     | 1           |       |      |     nan |     nan | 8175      |
| llama.cpp-hip     |             | -fa 1 |      |     nan |     nan | **6067**  |
| llama.cpp-hip     | 1           | -fa 1 |      |     nan |     nan | 6099      |
| llama.cpp-rocwmma |             |       |      |     nan |     nan | 8143      |
| llama.cpp-rocwmma | 1           |       |      |     nan |     nan | 8175      |
| llama.cpp-rocwmma |             | -fa 1 |      |     nan |     nan | **6067**  |
| llama.cpp-rocwmma | 1           | -fa 1 |      |     nan |     nan | 6099      |
| llama.cpp-vulkan  |             |       |      |     nan |     nan | 6492      |
| llama.cpp-vulkan  |             | -fa 1 |      |     nan |     nan | 6135      |
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
| llama.cpp-hip     |             |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 196.777822     |
| llama.cpp-hip     | 1           |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | **407.242257** |
| llama.cpp-hip     |             | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 105.354111     |
| llama.cpp-hip     | 1           | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 146.329906     |
| llama.cpp-rocwmma |             |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 200.319857     |
| llama.cpp-rocwmma | 1           |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 402.586271     |
| llama.cpp-rocwmma |             | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 104.26495      |
| llama.cpp-rocwmma | 1           | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 147.196762     |
| llama.cpp-vulkan  |             |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 386.581865     |
| llama.cpp-vulkan  |             | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 359.61209      |
### TG sweep
| backend           | hipblaslt   | -fa   | -b   |   1 |   2 |   4 |   8 |   16 |   32 |   64 |   128 |   256 |   512 |   1024 |   2048 |   4096 | 8192          |
|-------------------|-------------|-------|------|-----|-----|-----|-----|------|------|------|-------|-------|-------|--------|--------|--------|---------------|
| llama.cpp-hip     |             |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 21.174271     |
| llama.cpp-hip     | 1           |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 21.183963     |
| llama.cpp-hip     |             | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 23.68605      |
| llama.cpp-hip     | 1           | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 23.724341     |
| llama.cpp-rocwmma |             |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 21.206953     |
| llama.cpp-rocwmma | 1           |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 21.164052     |
| llama.cpp-rocwmma |             | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 23.735844     |
| llama.cpp-rocwmma | 1           | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 23.688242     |
| llama.cpp-vulkan  |             |       |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | 28.120159     |
| llama.cpp-vulkan  |             | -fa 1 |      | nan | nan | nan | nan |  nan |  nan |  nan |   nan |   nan |   nan |    nan |    nan |    nan | **33.615882** |