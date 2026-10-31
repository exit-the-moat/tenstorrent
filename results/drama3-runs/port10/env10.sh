# port10: batched throughput. Two cards: card N lock = /opt/ep0/cardN.lock (N = tt-metal device id, TT_VISIBLE_DEVICES=N).
# /opt/ep0/card.lock (old single lock) still means "all cards": take it only when both are free; lockN waits on it too.
source /opt/ep0/port8/env8.sh
P=/opt/ep0/port10
R="--ref /opt/ep0/voice/ref_v2.wav"
B7="$INTER QWEN36_GROUPED_MM_GLUE=1 DRAMA3_GDN_FUSED_BATCH=1 DRAMA3_MOE_FUSED_BATCH=1 DRAMA3_ATTN_FUSED_BATCH=1 DRAMA3_QUANT_DEVICE=1"
lockc() { while [ -e /opt/ep0/card.wanted ] || [ -e /opt/ep0/card.lock ]; do sleep 20; done; until mkdir /opt/ep0/card$1.lock 2>/dev/null; do sleep 20; done; echo "ep0-port10 $$" > /opt/ep0/card$1.lock/owner; }
unlockc() { rm -rf /opt/ep0/card$1.lock; }
pw_start() { rm -f $1/power.stop; PW_DEV=${2:-0} python3 $P/power_log.py log $1/power.jsonl $1/power.stop & }
pw_stop() { touch $1/power.stop; sleep 3; python3 /opt/ep0/port6/power_sum.py $1/power.jsonl $1/timing.json > $1/power.txt 2>&1; }
iochk() { echo "io=$(dmesg --level=err | grep -c "I/O error") disk=$(df --output=pcent /opt | tail -1 | tr -d " ")"; }
