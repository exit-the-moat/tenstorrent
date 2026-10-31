source /opt/ep0/port10/bb12/env.sh
for n in bb128 off3 bb128b; do
  case $n in bb*) ex="DRAMA3_BB_PREFETCH=1 DRAMA3_PREFETCH_GCB_KB=128";; *) ex="";; esac
  chk ft10_$n "$ex" tests.profile_frame7 --mode time --tower-mlp bfp8 --out $Q/ft10_$n.json
  python3 -c "import json;d=json.load(open(\"$Q/ft10_$n.json\"));print(\"$n\",{k:d[k] for k in (\"backbone_trace_ms\",\"tower_trace_ms\",\"bb_plus_tower_trace_ms\",\"gen_ms_per_frame\")})" 2>&1 | tail -1
done
echo A10_DONE
