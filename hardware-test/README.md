# Strix Halo Hardware Testing

## Summary

Full `hw-probe`: https://linux-hardware.org/?probe=6cace57e50

Passmark submitted run:
- https://www.passmark.com/baselines/V11/display.php?id=510063956264

GeekBench:
- CPU: https://browser.geekbench.com/v6/cpu/13245806
- RADV: https://browser.geekbench.com/v6/compute/4589439
- AMDVLK: https://browser.geekbench.com/v6/compute/4589426

### GPU MBW
This is what we mostly care about for inference. The theoretical maximum limit is 256GiB/s (LPDDR5x-8000 x 256-bit bus).

The maximum MBW I've gotten so far with GPU testing is 221 GB/s with Mesa RADV, 86% of the theoretical bus limit, which is pretty good!

Using `memtest_vulkan` we get decent performance with AMDVLK out of the box:
```
     11 iteration. Passed  5.4158 seconds  written:  551.0GB 214.6GB/sec        checked:  580.0GB 203.7GB/sec
```

We can get a decent bump with `amd_iommu=off` set as a kernel option (other settings like `iommu=pt` don't give you the benefit, so if you need IOMMU for VFIO/passthrough, etc you probably can't use this:
```
     11 iteration. Passed  5.2780 seconds  written:  551.0GB 216.7GB/sec        checked:  580.0GB 212.0GB/sec
```

Mesa RADV performs about the same:
```
     11 iteration. Passed  5.3863 seconds  written:  580.0GB 227.8GB/sec        checked:  609.0GB 214.4GB/sec
```

You *can* however, get another bump with `tuned` changed from the default `balanced` to `accelerator-performance`:
```
     11 iteration. Passed  5.2309 seconds  written:  580.0GB 234.4GB/sec        checked:  609.0GB 221.0GB/sec
```

### CPU MBW
CPU Memory Bandwidth is note so good, about half of the max MBW bus...

Transfers between CPU/GPU on ROCm are about 84-85 GB/s:
```
Device: 0,  AMD RYZEN AI MAX+ 395 w/ Radeon 8060S
Device: 1,  AMD Radeon Graphics,  GPU-XX,  c3:0.0

Unidirectional copy peak bandwidth GB/s
D/D       0           1           
0         N/A         83.715      
1         85.476      212.588     

Bidirectional copy peak bandwidth GB/s
D/D       0           1           
0         N/A         84.706      
1         84.706      N/A         
```

Primary testing with `likwid`:
```
Test: copy
MByte/s:                101478.93

Test: stream
MByte/s:                99972.51

Test: triad
MByte/s:                100355.00

Test: load
MByte/s:                108615.61

Test: store
MByte/s:                86856.92

Test: update
MByte/s:                177599.03

[2025-08-09 23:43:50] === Multi-threaded STREAM Test ===
Test: stream
MByte/s:                99949.73
```
Passmark:
```
  ME_ALLOC_S: 19656.591529967704
  ME_READ_S: 38838.813020833331
  ME_READ_L: 34486.138020833336
  ME_WRITE: 29566.629557291668
  ME_LARGE: 125117.86111111112
  ME_LATENCY: 92.466689494547722
  ME_THREADED: 125598.13818359375
```

Intel MLC:
```
❯ echo 32000 > /proc/sys/vm/nr_hugepages
❯ mlc --idle_latency
Using buffer size of 1800.000MiB
Each iteration took 453.5 base frequency clocks (       151.4   ns)

❯ mlc --bandwidth_matrix
Using buffer size of 100.000MiB/thread for reads and an additional 100.000MiB/thread for writes
Measuring Memory Bandwidths between nodes within system
Bandwidths are in MB/sec (1 MB/sec = 1,000,000 Bytes/sec)
Using all the threads from each core if Hyper-threading is enabled
Using Read-only traffic type
                Numa node
Numa node            0
       0        117876.8
```


This explanation seems plausible... https://chatgpt.com/share/68979a63-09c8-8012-8b07-513780f6d1db

> On Zen 5, each L3 “cluster” (8 cores) has a 32 bytes/cycle fabric link to the rest of the SoC. On mobile parts that link is 32 B/cycle in both directions and typically runs around ~2 GHz. That’s ~64 GB/s read per cluster. A 16‑core Strix Halo CCD has two clusters ⇒ ~128 GB/s aggregate peak for CPU reads. Your ~119 GB/s is ~93% of that theoretical cap, so you’re basically hitting the CCD→SoC link ceiling, not DRAM.
>
> Strix Halo’s “sea‑of‑wires” connection replaces the old GMI SERDES, but the per‑cluster width still matters. AMD’s own deep‑dive explains the 32 B/cycle links and the low‑latency fan‑out; the key point is that the GPU and memory controllers sit on the big SoC die while the CPU cores sit on the CCD, so CPU streaming ultimately funnels through those 32 B/cycle cluster links.
Chips and Cheese
> 
> That’s why your GPU test (Vulkan membench) gets ~230–235 GB/s — the iGPU talks to LPDDR5X directly (and can leverage MALL/last‑level cache structures on the SoC die), whereas the CPU can’t allocate into that GPU cache and is gated by the cluster links.

- https://chipsandcheese.com/p/amds-ryzen-9950x-zen-5-on-desktop
- https://chipsandcheese.com/p/amds-strix-halo-under-the-hood
