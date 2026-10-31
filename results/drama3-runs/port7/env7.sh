source /opt/ep0/port6/env.sh
# today's --interactive (run_gen.sh) = INTER + the two port6 levers
I7="$INTER QWEN36_GROUPED_MM_GLUE=1 DRAMA3_QUANT_DEVICE=1"
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3
P=/opt/ep0/port7
lock() { while [ -e /opt/ep0/card.wanted ]; do sleep 30; done; until mkdir /opt/ep0/card.lock 2>/dev/null; do sleep 20; done; echo ep0-port7 > /opt/ep0/card.lock/owner; }
unlock() { rm -rf /opt/ep0/card.lock; }
