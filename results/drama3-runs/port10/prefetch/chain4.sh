#!/bin/bash
# shared prefetcher (tt/weight_prefetch.py) with the tower group: card 1
source /opt/ep0/port10/env10.sh
cd /opt/ep0/port10/prefetch
rm -rf s_tw b_both b_all b_off
./gen.sh s_tw 'DRAMA3_TOWER_PREFETCH=1 DRAMA3_TOWER_PREFETCH_SHARED=1' > s_tw.out 2>&1; echo "s_tw $(head -1 s_tw.out) $(grep -c '"frames_identical": true' s_tw.out)/8 vs i_bf8"
/opt/ep0/run.sh 'python /opt/ep0/port5/cmp_ttfa.py /opt/ep0/port10/prefetch/s_tw /opt/ep0/port10/prefetch/g_on3' 2>&1 | grep -c '"frames_identical": true, "wav_identical": true' | sed 's/^/s_tw vs g_on3 (old tower path): identical lines /'
./gen.sh b_both 'DRAMA3_BB_PREFETCH=1 DRAMA3_TOWER_PREFETCH=1' > b_both.out 2>&1; echo "b_both $(head -1 b_both.out)"
./gen.sh b_all 'DRAMA3_BB_PREFETCH=1 DRAMA3_TOWER_PREFETCH=1 DRAMA3_SAMPLER_FUSED=1 DRAMA3_SAMPLER_FUSED_CORES=4' > b_all.out 2>&1; echo "b_all $(head -1 b_all.out)"
./gen.sh b_off '' > b_off.out 2>&1; echo "b_off $(head -1 b_off.out) $(grep -c '"frames_identical": true' b_off.out)/8 vs i_bf8"
python3 /opt/ep0/port7/rtf.py b_off b_all b_both s_tw
echo CHAIN4_DONE
