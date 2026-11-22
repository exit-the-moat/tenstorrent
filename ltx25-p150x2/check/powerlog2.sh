#!/bin/bash
# ep4 powerlog v2 (open frame, 18 Nov): passive 5 s log, no tt-smi. Per card: W, C, mV, A, AICLK; PSU output W.
# Plus BOARD SENSORS (not a room thermometer): nct6798 SYSTIN / CPUTIN, PSU case + VRM temp, CPU Tctl.
OUT=$1
B=(); for h in /sys/class/hwmon/hwmon*; do n=$(cat $h/name); [ "$n" = blackhole ] && B+=($h); [ "$n" = corsairpsu ] && P=$h; [ "$n" = nct6798 ] && N=$h; [ "$n" = k10temp ] && K=$h; done
echo "# hwmon ${B[*]} psu $P board $N cpu $K; board/psu temps are sensors on the board and in the PSU, not a room thermometer" > $OUT
echo "epoch,c0_W,c0_C,c0_mV,c0_A,c0_aiclk,c1_W,c1_C,c1_mV,c1_A,c1_aiclk,psu_W,systin_C,cputin_C,psu_case_C,psu_vrm_C,cpu_tctl_C" >> $OUT
t() { awk "{printf \"%.1f\",\$1/1e3}" $1; }
while true; do
  l="$(date +%s.%N | cut -c1-14)"; i=0
  for h in "${B[@]}"; do
    l="$l,$(awk "{printf \"%.1f\",\$1/1e6}" $h/power1_input),$(t $h/temp1_input),$(cat $h/in0_input),$(t $h/curr1_input),$(cat /sys/class/tenstorrent/tenstorrent\!$i/tt_aiclk)"; i=$((i+1))
  done
  l="$l,$(awk "{printf \"%.1f\",\$1/1e6}" $P/power1_input),$(t $N/temp1_input),$(t $N/temp2_input),$(t $P/temp2_input),$(t $P/temp1_input),$(t $K/temp1_input)"
  echo "$l" >> $OUT; /usr/bin/sleep 5
done
