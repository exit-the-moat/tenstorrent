#!/bin/bash
# ep4 (from ep3 img4): passive 5 s log (no tt-smi): per card power W, temp C, vcore mV, current A, aiclk MHz (sysfs tt_aiclk); PSU total W.
OUT=$1
B=(); for h in /sys/class/hwmon/hwmon*; do n=$(cat $h/name); [ "$n" = blackhole ] && B+=($h); [ "$n" = corsairpsu ] && P=$h; done
echo "# hwmon ${B[*]}; aiclk /sys/class/tenstorrent/tenstorrent!0,1/tt_aiclk; psu $P" > $OUT
echo "epoch,c0_W,c0_C,c0_mV,c0_A,c0_aiclk,c1_W,c1_C,c1_mV,c1_A,c1_aiclk,psu_W" >> $OUT
while true; do
  l="$(date +%s.%N | cut -c1-14)"; i=0
  for h in "${B[@]}"; do
    l="$l,$(awk '{printf "%.1f",$1/1e6}' $h/power1_input),$(awk '{printf "%.1f",$1/1e3}' $h/temp1_input),$(cat $h/in0_input),$(awk '{printf "%.1f",$1/1e3}' $h/curr1_input),$(cat /sys/class/tenstorrent/tenstorrent\!$i/tt_aiclk)"; i=$((i+1))
  done
  l="$l,$(awk '{printf "%.1f",$1/1e6}' $P/power1_input)"
  echo "$l" >> $OUT; /usr/bin/sleep 5
done
