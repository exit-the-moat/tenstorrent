cd /opt/ep0/port10
export TMO=600
./t1.sh prefetch/t_cal "" tests.bench_tower_prefetch --variants c,a,a2,b --ibw 8 --out /opt/ep0/port10/prefetch/t_cal.json
./t1.sh prefetch/pr_long "TT_METAL_DEVICE_PROFILER=1" tests.bench_tower_prefetch --variants b --ibw 8 --gap_rep 8 --prof b_with_small_ops --out /opt/ep0/port10/prefetch/pr_long.json
echo SWEEP_DONE
