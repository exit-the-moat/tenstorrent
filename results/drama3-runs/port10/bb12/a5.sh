source /opt/ep0/port10/bb12/env.sh
for k in attn_qkv attn_o gdn_in gdn_o; do
chk g_$k "DRAMA3_BB_PREFETCH=1 DRAMA3_PREFETCH_GCB_KB=160 DRAMA3_BB_PREFETCH_ONLY=$k" tests.check_paged_backbone --out $Q/g_$k.json
done
echo A5_DONE
