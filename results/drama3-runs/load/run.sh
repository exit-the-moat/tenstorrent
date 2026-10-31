#!/bin/bash
# Episode 0 scene 6: model load times on PCIe Gen3 x16. Log markers: STEP, DONE.
D=/opt/ep0/tt-metal/models/demos/blackhole/drama3; L=/opt/ep0/load; C=/opt/ep0/cache/drama3/P150/tensor_cache_bfp8
echo "io $(dmesg --level=err | grep -c "I/O error") $(date +%T)"
lspci -vv -d 1e52: 2>/dev/null | grep -E "LnkSta:" | head -1
echo "cache $(du -sb $C | cut -f1) bytes, $(ls $C | wc -l) files"
sync; echo 3 > /proc/sys/vm/drop_caches
echo "STEP disk_read $(date +%T)"; t0=$(date +%s.%N); cat $C/* > /dev/null; t1=$(date +%s.%N); echo "DISK_READ_S $(echo "$t1 - $t0" | bc)"
for name in warm1 warm2; do
  echo "STEP $name $(date +%T)"; t0=$(date +%s.%N)
  $D/run_gen.sh $L/$name.log --fast --lines close --out $L/$name
  t1=$(date +%s.%N); echo "$name WALL_S $(echo "$t1 - $t0" | bc) $(grep -h "resident" $L/$name.log) $(tail -1 $L/$name.log)"
done
sync; echo 3 > /proc/sys/vm/drop_caches
echo "STEP cold $(date +%T)"; t0=$(date +%s.%N)
$D/run_gen.sh $L/cold.log --fast --lines close --out $L/cold
t1=$(date +%s.%N); echo "cold WALL_S $(echo "$t1 - $t0" | bc) $(grep -h "resident" $L/cold.log) $(tail -1 $L/cold.log)"
echo "STEP pcie $(date +%T)"
/opt/ep0/run.sh "python /opt/ep0/load/pcie_bw.py" 2>&1 | grep -E "^PCIE|Error|error" | tail -3
echo "io $(dmesg --level=err | grep -c "I/O error") $(date +%T)"
echo DONE
