# Benchmark Results
| backend          | hipblaslt   | -fa   | -b     | pp512          | tg128         | max_mem   |
|------------------|-------------|-------|--------|----------------|---------------|-----------|
| llama.cpp-hip    |             |       |        | 110.062773     | 17.167543     | 60728     |
| llama.cpp-hip    | 1           |       |        | **264.144777** | 17.164478     | 60765     |
| llama.cpp-hip    |             | -fa 1 |        | 93.995419      | 16.286141     | 60209     |
| llama.cpp-hip    | 1           | -fa 1 |        | 213.55044      | 16.27352      | 60237     |
| llama.cpp-vulkan |             |       |        | 173.232358     | 19.606697     | 60286     |
| llama.cpp-vulkan |             |       | -b 256 | 146.728579     | **19.668124** | 60067     |
| llama.cpp-vulkan |             | -fa 1 |        | 173.859797     | 19.351249     | 60268     |
| llama.cpp-vulkan |             | -fa 1 | -b 256 | 146.119299     | 19.345135     | **60037** |
## Performance Charts

### Tokens/s Performance
![PP Tokens/s](pp_tokens_per_sec.png)
![TG Tokens/s](tg_tokens_per_sec.png)

### Memory Usage
![PP VRAM](pp_vram_peak_mib.png)
![TG VRAM](tg_vram_peak_mib.png)

## Detailed Sweeps

### PP sweep
| backend          | hipblaslt   | -fa   | -b     | 1             | 2             | 4                      | 8             | 16            | 32            | 64           | 128            | 256                | 512            | 1024           | 2048           | 4096           |
|------------------|-------------|-------|--------|---------------|---------------|------------------------|---------------|---------------|---------------|--------------|----------------|--------------------|----------------|----------------|----------------|----------------|
| llama.cpp-hip    |             |       |        | 17.257808     | 22.098716     | 31.680183              | **41.963382** | **67.286749** | **71.332406** | 66.010713    | 85.551541      | 106.122404         | 110.062773     | 107.548881     | 105.57523      | 100.290184     |
| llama.cpp-hip    | 1           |       |        | 17.234388     | **22.157246** | **31.806767999999998** | 34.485094     | 57.443787     | 63.998599     | 91.316962    | **141.380404** | **196.366691**     | **264.144777** | **255.163895** | **236.443135** | **210.798748** |
| llama.cpp-hip    |             | -fa 1 |        | 16.306066     | 20.855548     | 30.238365              | 40.973672     | 62.087885     | 65.701995     | 60.104762    | 73.068078      | 92.759895          | 93.995419      | 93.197817      | 81.618568      | 69.220143      |
| llama.cpp-hip    | 1           | -fa 1 |        | 16.301389     | 20.916661     | 30.601308              | 40.624837     | 64.744191     | 68.553694     | **91.35537** | 137.073491     | 185.416123         | 213.55044      | 189.981017     | 153.621217     | 109.709723     |
| llama.cpp-vulkan |             |       |        | 19.878657     | 7.686462      | 9.451524               | 12.425651     | 25.833629     | 41.618101     | 63.372745    | 102.937517     | 148.762566         | 173.232358     | 168.144368     | 162.757085     | 159.392358     |
| llama.cpp-vulkan |             |       | -b 256 | **19.896438** | 7.531335      | 9.429977               | 12.291575     | 26.149455     | 42.040268     | 63.409361    | 103.545233     | 148.18868          | 146.728579     | 137.971288     | 141.634431     | 130.511215     |
| llama.cpp-vulkan |             | -fa 1 |        | 19.040761     | 7.537129      | 9.252738               | 12.371741     | 26.0906       | 41.868124     | 63.102697    | 102.266319     | 146.61933199999999 | 173.859797     | 171.67967      | 165.008335     | 152.063766     |
| llama.cpp-vulkan |             | -fa 1 | -b 256 | 19.389263     | 7.556843      | 9.242198               | 12.350794     | 25.739454     | 41.575574     | 62.665022    | 102.230406     | 148.272772         | 146.119299     | 137.067282     | 140.308531     | 124.492035     |
### TG sweep
| backend          | hipblaslt   | -fa   | -b     | 1             | 2             | 4             | 8             | 16                 | 32            | 64            | 128           | 256           | 512           | 1024          | 2048          | 4096         |
|------------------|-------------|-------|--------|---------------|---------------|---------------|---------------|--------------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|--------------|
| llama.cpp-hip    |             |       |        | 17.19597      | 17.28146      | 17.258775     | 17.306296     | 17.28916           | 17.290922     | 17.252076     | 17.167543     | 16.931914     | 16.555267     | 15.986413     | 14.786972     | 13.114607    |
| llama.cpp-hip    | 1           |       |        | 17.172821     | 17.288524     | 17.290641     | 17.274267     | 17.248851          | 17.288417     | 17.258547     | 17.164478     | 16.953116     | 16.54074      | 15.97659      | 14.811392     | 13.113559    |
| llama.cpp-hip    |             | -fa 1 |        | 16.267948     | 16.286472     | 16.264229     | 16.293126     | 16.286175          | 16.282448     | 16.281882     | 16.286141     | 16.261251     | 16.256349     | 15.860658     | 15.090314     | 13.680993    |
| llama.cpp-hip    | 1           | -fa 1 |        | 16.264933     | 16.289693     | 16.289437     | 16.302133     | 16.276338          | 16.292744     | 16.284262     | 16.27352      | 16.26275      | 16.26861      | 15.853814     | 15.076421     | 13.687798    |
| llama.cpp-vulkan |             |       |        | 19.89512      | 19.953574     | 19.846087     | **19.944716** | **19.905186**      | 19.869469     | 19.87386      | 19.606697     | **19.439303** | **19.297433** | 18.953789     | 18.105816     | 16.723876    |
| llama.cpp-vulkan |             |       | -b 256 | **19.974723** | **19.972674** | **19.969319** | 19.936374     | 19.809029          | **19.912953** | **19.875589** | **19.668124** | 19.412263     | 19.250586     | 18.924006     | 18.081305     | 16.745992    |
| llama.cpp-vulkan |             | -fa 1 |        | 19.42035      | 19.436526     | 19.2698       | 19.39698      | 19.372672          | 19.376482     | 19.346376     | 19.351249     | 19.302318     | 19.206259     | 19.040451     | 18.653384     | 17.926649    |
| llama.cpp-vulkan |             | -fa 1 | -b 256 | 19.434549     | 19.416633     | 19.465964     | 19.33647      | 19.408360000000002 | 19.336844     | 19.333993     | 19.345135     | 19.34846      | 19.21995      | **19.041296** | **18.711098** | **17.92843** |