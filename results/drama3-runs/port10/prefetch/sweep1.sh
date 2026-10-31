cd /opt/ep0/port10
export TMO=600
./t1.sh prefetch/s_lofi "" tests.bench_tower_prefetch --variants a2,b --fid LoFi --out /opt/ep0/port10/prefetch/s_lofi.json
./t1.sh prefetch/s_ibw8 "" tests.bench_tower_prefetch --variants b --ibw 8 --out /opt/ep0/port10/prefetch/s_ibw8.json
./t1.sh prefetch/s_ibw2 "" tests.bench_tower_prefetch --variants b --ibw 2 --out /opt/ep0/port10/prefetch/s_ibw2.json
./t1.sh prefetch/s_g1024 "" tests.bench_tower_prefetch --variants b --gcb_kb 1024 --out /opt/ep0/port10/prefetch/s_g1024.json
./t1.sh prefetch/s_g160 "" tests.bench_tower_prefetch --variants b --gcb_kb 160 --out /opt/ep0/port10/prefetch/s_g160.json
echo SWEEP_DONE
