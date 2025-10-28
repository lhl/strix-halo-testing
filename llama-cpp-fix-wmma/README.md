# Improving llama.cpp HIP Backent Performance

Recently some people on the Strix Halo HomeLab Discord noticed that the HIP rocWMMA backend actually performed much worse than the HIP backend (especially for decode/token generation) as context got longer.

After having done an analysis of how ggml-cuda worked and discovering the rocWMMA path was actually quite underoptimized, I decided to give a quick poke to see if there was an easy fix. 

The approach was to see if some minimal changes to improve performance much. The results showed that actually, there were huge optimizations available and that it'd be relatively clean to patch in improvements.

The big things discovered:
- The WMMA path is older/deprecated on the Nvidia side (designed for Volta) but there were a few parameters that could dramatically increase occupancy for rocWMMA
- The reason for the poor long-context decode is because it was only using the old VEC not the new TILE kernels when appropriate. There was a second issue of some weird TILE constraints (and no guards or fallbacks)

Once these were fixed, you could get a huge improvement for prefill over both the prior HIP and WMMA implementation at all contexts (but especially at long context), and then match the HIP performance for tg.


Building
```
# HIP
cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 && cmake --build build --config Release -j32

# rocWMMA
 cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DGGML_HIP_ROCWMMA_FATTN=ON && cmake --build build --config Release -j32
```
