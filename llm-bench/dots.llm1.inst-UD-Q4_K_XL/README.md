# Benchmark Results
| backend          | hipblaslt   | -fa   | -b     | pp512         | tg128         | max_mem   |
|------------------|-------------|-------|--------|---------------|---------------|-----------|
| llama.cpp-vulkan |             |       |        | 45.352152     | 20.560089     | 87617     |
| llama.cpp-vulkan |             |       | -b 256 | 62.990238     | 20.503142     | 87470     |
| llama.cpp-vulkan |             | -fa 1 |        | 45.200454     | **20.642976** | 87557     |
| llama.cpp-vulkan |             | -fa 1 | -b 256 | **63.075788** | 20.578648     | **87391** |
## Performance Charts

### Tokens/s Performance
![PP Tokens/s](pp_tokens_per_sec.png)
![TG Tokens/s](tg_tokens_per_sec.png)

### Memory Usage
![PP VRAM](pp_vram_peak_mib.png)
![TG VRAM](tg_vram_peak_mib.png)

## Detailed Sweeps

### PP sweep
| backend          | hipblaslt   | -fa   | -b     | 1            | 2                  | 4            | 8             | 16            | 32            | 64            | 128           | 256           | 512           | 1024          | 2048          | 4096          |
|------------------|-------------|-------|--------|--------------|--------------------|--------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| llama.cpp-vulkan |             |       |        | **20.92023** | 7.562782           | 9.502482     | 12.081847     | 21.853825     | 33.11372      | **52.872539** | **71.015158** | **62.885915** | 45.352152     | 44.864281     | 44.321479     | 43.52484      |
| llama.cpp-vulkan |             |       | -b 256 | 20.672711    | **7.748233**       | **9.584894** | **12.111708** | **21.989551** | **33.209611** | 52.625463     | 70.660423     | 62.59552      | 62.990238     | 62.706707     | 61.67444      | **60.041643** |
| llama.cpp-vulkan |             | -fa 1 |        | 20.505938    | 7.5692070000000005 | 9.388567     | 12.07563      | 21.822866     | 32.664671     | 52.237397     | 69.216098     | 62.554135     | 45.200454     | 45.179621     | 44.475813     | 43.079747     |
| llama.cpp-vulkan |             | -fa 1 | -b 256 | 20.460135    | 7.5446159999999995 | 9.387418     | 12.051016     | 21.648124     | 32.682686     | 52.404597     | 69.423165     | 62.611495     | **63.075788** | **62.784051** | **61.701349** | 58.692821     |
### TG sweep
| backend          | hipblaslt   | -fa   | -b     | 1             | 2             | 4             | 8             | 16            | 32            | 64            | 128           | 256           | 512           | 1024          | 2048          | 4096          |
|------------------|-------------|-------|--------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| llama.cpp-vulkan |             |       |        | 20.637586     | 20.517832     | 20.702214     | **20.777383** | **20.976172** | **20.931864** | 20.775792     | 20.560089     | 20.207881     | 19.025833     | 16.782549     | 12.623831     | 8.573129      |
| llama.cpp-vulkan |             |       | -b 256 | **20.676842** | **20.553022** | **20.771794** | 20.754185     | 20.785683     | 20.78971      | **20.783599** | 20.503142     | 20.030778     | 19.021865     | 16.67647      | 12.546431     | 8.574004      |
| llama.cpp-vulkan |             | -fa 1 |        | 20.391352     | 20.538923     | 20.161356     | 20.391502     | 20.5489       | 20.799878     | 20.706999     | **20.642976** | **20.599771** | 20.185245     | 19.599759     | 18.518682     | 16.747888     |
| llama.cpp-vulkan |             | -fa 1 | -b 256 | 20.404704     | 20.34737      | 20.311917     | 20.454183     | 20.462601     | 20.560075     | 20.545954     | 20.578648     | 20.580562     | **20.377828** | **19.800902** | **18.560858** | **16.818945** |