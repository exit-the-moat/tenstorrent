cd /opt/ep0/port10
export TMO=600
B="tests.bench_tower_prefetch --layers 13 --n 20 --rpb 8"
./t1.sh prefetch/L13_cap "" $B --variants c,a,a2,b --out /opt/ep0/port10/prefetch/L13_cap.json
./t1.sh prefetch/L13_ahead_p128 "" $B --variants b --req ahead --page_kb 128 --out /opt/ep0/port10/prefetch/L13_ahead_p128.json
./t1.sh prefetch/L13_ahead_p128_g1024 "" $B --variants b --req ahead --page_kb 128 --gcb_kb 1024 --out /opt/ep0/port10/prefetch/L13_ahead_p128_g1024.json
echo SWEEP_DONE
