cd /opt/ep0/port10
export TMO=600
for v in b_mm_only a2_mm_only; do
./t1.sh prefetch/pr4_$v "TT_METAL_DEVICE_PROFILER=1" tests.bench_tower_prefetch --variants a2,b --ibw 8 --prof $v --out /opt/ep0/port10/prefetch/pr4_$v.json
done
echo SWEEP_DONE
