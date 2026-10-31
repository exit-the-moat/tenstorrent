#!/bin/bash
# Episode 0 v20 (Sebastian, 2026-10-31: "use interactive and render v20"): all card lines of the v20 script
# (/opt/ep0/voice/v20f_lines.json = narration/card_lines.json) with ref_v2 on --interactive, seeds 0-2, card 0 alone.
# --interactive warms one suffix prefill bucket, so one pass per bucket (DRAMA3_ONLY_BUCKET); the first pass logs
# the bucket histogram. Card power alongside, Whisper proxies (v19/ana.py), sync + drop caches, SHA256SUMS.
source /opt/ep0/port10/env10.sh
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
out=/opt/ep0/v20/out_f
mkdir -p $out
echo "=== v20 start $(date +%T) $(iochk)"
lockc 0
pw_start $out 0
gen() {  # gen <bucket>
  EXTRA_ENV="TT_VISIBLE_DEVICES=0 DRAMA3_ONLY_BUCKET=$1" timeout 3600 $D/run_gen.sh $out/gen_$1.log --interactive --lines all \
    --lines-file /opt/ep0/voice/v20f_lines.json --ref /opt/ep0/voice/ref_v2.wav --seeds 0,1,2 --out $out
  podman ps --format '{{.Command}}' | grep -q generate_fast && echo "WARN container still running"
  echo "bucket $1: $(tail -1 $out/gen_$1.log) wavs=$(ls $out/*.wav 2>/dev/null | wc -l) $(date +%T)"
}
gen 128
for b in $(grep -o "suffix prefill buckets {.*}" $out/gen_128.log | grep -o "[0-9]*:" | tr -d :); do
  [ "$b" = 128 ] || gen $b
done
pw_stop $out
unlockc 0
/opt/ep0/v19/ana.sh /opt/ep0/voice/v20f_lines.json $out > $out/ana.log 2>&1
/opt/ep0/verify_new.sh $out "*.wav"  # evicts only the new wavs; drop_caches evicted the model weights too (2026-10-31)
echo "=== v20 end $(date +%T) $(iochk)"
echo V20F_DONE
