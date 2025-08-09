CPU Memory Bandwidth is about half of the max MBW (GMI link limitations?)

Likwid:
```
213:MByte/s:            101478.93
285:MByte/s:            99972.51
358:MByte/s:            100355.00
428:MByte/s:            108615.61
497:MByte/s:            86856.92
566:MByte/s:            177599.03
641:MByte/s:            99949.73
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

This seems plausible... https://chatgpt.com/share/68979a63-09c8-8012-8b07-513780f6d1db
```
On Zen 5, each L3 “cluster” (8 cores) has a 32 bytes/cycle fabric link to the rest of the SoC. On mobile parts that link is 32 B/cycle in both directions and typically runs around ~2 GHz. That’s ~64 GB/s read per cluster. A 16‑core Strix Halo CCD has two clusters ⇒ ~128 GB/s aggregate peak for CPU reads. Your ~119 GB/s is ~93% of that theoretical cap, so you’re basically hitting the CCD→SoC link ceiling, not DRAM.

Strix Halo’s “sea‑of‑wires” connection replaces the old GMI SERDES, but the per‑cluster width still matters. AMD’s own deep‑dive explains the 32 B/cycle links and the low‑latency fan‑out; the key point is that the GPU and memory controllers sit on the big SoC die while the CPU cores sit on the CCD, so CPU streaming ultimately funnels through those 32 B/cycle cluster links.
Chips and Cheese

That’s why your GPU test (Vulkan membench) gets ~230–235 GB/s — the iGPU talks to LPDDR5X directly (and can leverage MALL/last‑level cache structures on the SoC die), whereas the CPU can’t allocate into that GPU cache and is gated by the cluster links.
```
- https://chipsandcheese.com/p/amds-ryzen-9950x-zen-5-on-desktop
- https://chipsandcheese.com/p/amds-strix-halo-under-the-hood
