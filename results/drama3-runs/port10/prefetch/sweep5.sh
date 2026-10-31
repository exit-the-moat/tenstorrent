cd /opt/ep0/port10
export TMO=600
B="tests.bench_tower_prefetch"
./t1.sh prefetch/r8_cap "" $B --variants c,a,a2,b --rpb 8 --out /opt/ep0/port10/prefetch/r8_cap.json
./t1.sh prefetch/r8_ahead "" $B --variants b --rpb 8 --req ahead --out /opt/ep0/port10/prefetch/r8_ahead.json
./t1.sh prefetch/r4_ahead "" $B --variants b --rpb 4 --req ahead --out /opt/ep0/port10/prefetch/r4_ahead.json
./t1.sh prefetch/r8_ahead_p128 "" $B --variants b --rpb 8 --req ahead --page_kb 128 --out /opt/ep0/port10/prefetch/r8_ahead_p128.json
./t1.sh prefetch/r8_ahead_hifi2 "" $B --variants a,a2,b --rpb 8 --req ahead --fid HiFi2 --out /opt/ep0/port10/prefetch/r8_ahead_hifi2.json
echo SWEEP_DONE
