source /opt/ep0/port10/bb12/env.sh
BB="DRAMA3_BB_PREFETCH=1 DRAMA3_PREFETCH_GCB_KB=160"
SM="DRAMA3_SAMPLER_FUSED=1"
for n in off bb smp off2 bb2 smp2; do
  case $n in bb*) ex="$BB";; smp*) ex="$SM";; *) ex="";; esac
  chk ft9_$n "$ex" tests.profile_frame7 --mode time --tower-mlp bfp8 --out $Q/ft9_$n.json
  python3 -c "import json;d=json.load(open(\"$Q/ft9_$n.json\"));print(\"$n\",{k:d[k] for k in (\"backbone_trace_ms\",\"tower_trace_ms\",\"bb_plus_tower_trace_ms\",\"gen_ms_per_frame\")})" 2>&1 | tail -1
done
echo A9_DONE
