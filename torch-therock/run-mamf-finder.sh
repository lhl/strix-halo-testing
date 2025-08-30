# Quick run - course scan, 1 minute
python mamf-finder.py --dtype bfloat16 \
  --m_range 0 20480 256 \
  --n 4096 \
  --k 4096

## coarse scan - then 256 step +/- 1024
# python mamf-finder.py --dtype bfloat16 \
#  --m_range 0 20480 2048 --n_range 0 20480 2048 --k_range 0 20480 2048

## Full Sweep
#python ./mamf-finder.py --dtype bfloat16 \
#                 --m_range 0 16384 1024 \
#                 --n_range 0 16384 1024 \
#                 --k_range 0 16384 1024

### 4 Minutes
# Tried  79 shapes => the best outcomes were:
# mean:   21.6 TFLOPS @ 19200x4096x4096 (MxNxK)
# median: 21.7 TFLOPS @ 19200x4096x4096 (MxNxK)
# max:    22.6 TFLOPS @ 18176x4096x4096 (MxNxK)

# geomean: 19.0 TFLOPS for 79 shapes in range: m=[0, 20480, 256] | n=[4096] | k=[4096]

# Legend: TFLOPS = 10**12 FLOPS
# Elapsed time: 0:04:19
