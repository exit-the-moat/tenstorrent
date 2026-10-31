source /opt/ep0/port10/bb12/env.sh
BB="DRAMA3_BB_PREFETCH=1 DRAMA3_PREFETCH_GCB_KB=128"
# per-thread CPU of the generator process while a run is in its frame loop
sampler() { sleep 75; p=$(pgrep -f "host.generate_fast" | head -1); [ -n "$p" ] && top -H -b -n 1 -d 1 -p $p | head -16 > $Q/$1.top; }
for n in j_off j_bb j_off2 j_bb2 j_off3 j_bb3; do
  case $n in j_bb*) ex="$BB";; *) ex="";; esac
  sampler $n & gen $n "$ex"; wait
done
echo A12_DONE
