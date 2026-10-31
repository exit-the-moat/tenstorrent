#!/bin/bash
cd /opt/ep0/v19
./ana.sh /opt/ep0/v19/base/card_lines.json /opt/ep0/v19/base > base.log 2>&1
./ana.sh /opt/ep0/voice/emo2_lines.json /opt/ep0/emo2/out > emo2.log 2>&1
./ana.sh /opt/ep0/voice/emo3_lines.json /opt/ep0/emo3/out > emo3.log 2>&1
./ana.sh /opt/ep0/emo/lines.json /opt/ep0/emo/out > emo.log 2>&1
echo ALL_DONE >> emo.log
