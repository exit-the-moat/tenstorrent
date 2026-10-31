source /opt/ep0/port10/bb12/env.sh
for n in off on off2 on2; do
  case $n in on*) ex="DRAMA3_BB_PREFETCH=1 DRAMA3_PREFETCH_GCB_KB=160";; *) ex="";; esac
  chk ft_$n "$ex" tests.profile_frame7 --mode time --tower-mlp bfp8 --out $Q/ft_$n.json
  python3 -c "import json;d=json.load(open(\"$Q/ft_$n.json\"));print(\"$n\",{k:d[k] for k in (\"backbone_trace_ms\",\"tower_trace_ms\",\"bb_plus_tower_trace_ms\",\"gen_ms_per_frame\")})"
done
echo A8_DONE
