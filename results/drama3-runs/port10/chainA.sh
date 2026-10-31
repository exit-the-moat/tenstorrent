#!/bin/bash
cd /opt/ep0/port10
./run10.sh s16 16 short ""
./run10.sh s16_ct 16 short "DRAMA3_CODEC_DEV_TRACE=1"
./run10.sh s12 12 short ""
echo CHAIN_A_DONE
