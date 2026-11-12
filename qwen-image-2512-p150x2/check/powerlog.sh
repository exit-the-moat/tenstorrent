#!/bin/bash
# Passive power sampling at ~1 Hz from hwmon (no tt-smi): every blackhole card and, if present, a Corsair PSU (corsairpsu hwmon).
# Usage: powerlog.sh <out.csv> ; stop with kill. Values: W (power*_input is in microwatts).
OUT=$1
B=(); for h in /sys/class/hwmon/hwmon*; do n=$(cat $h/name); [ "$n" = blackhole ] && B+=($h); [ "$n" = corsairpsu ] && P=$h; done
hdr="epoch"; for i in "${!B[@]}"; do hdr="$hdr,card${i}_W,card${i}_temp_C"; done
for k in 1 2 3 4; do [ -e $P/power${k}_input ] && hdr="$hdr,psu_$(cat $P/power${k}_label | tr ' ' '_')_W"; done
echo "# blackhole hwmon order: ${B[*]}; psu: $P" > $OUT
echo "$hdr" >> $OUT
while true; do
  line="$(date +%s.%N | cut -c1-14)"
  for h in "${B[@]}"; do line="$line,$(awk '{printf "%.2f",$1/1e6}' $h/power1_input),$(awk '{printf "%.1f",$1/1e3}' $h/temp1_input)"; done
  for k in 1 2 3 4; do [ -e $P/power${k}_input ] && line="$line,$(awk '{printf "%.2f",$1/1e6}' $P/power${k}_input)"; done
  echo "$line" >> $OUT
  /usr/bin/sleep 1
done
