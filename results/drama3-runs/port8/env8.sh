source /opt/ep0/port6/env.sh
I7="$INTER QWEN36_GROUPED_MM_GLUE=1 DRAMA3_QUANT_DEVICE=1"
# port7 best same-precision flags (the port8 start)
P7F="DRAMA3_FRAME_PIPE=1 DRAMA3_SAMPLER_LEAN=1 DRAMA3_TOWER_ETABLE=1 QWEN36_GROUPED_COMPACT=1"
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
P=/opt/ep0/port8
lock() { while [ -e /opt/ep0/card.wanted ]; do sleep 30; done; until mkdir /opt/ep0/card.lock 2>/dev/null; do sleep 20; done; echo ep0-port8 > /opt/ep0/card.lock/owner; }
unlock() { rm -rf /opt/ep0/card.lock; }
