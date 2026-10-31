#!/bin/bash
# Day 5: BFP8 gate/up expert cache files for the given layers, with the disk guard (dmesg I/O errors before/after,
# sync + drop caches, sha256 read back from disk).   convert_gu.sh 3,7,12
set -u
L=$1
CACHE=/opt/ep0/cache/drama3/P150/tensor_cache_bfp8
SUMS=/opt/ep0/day5/cache_bfp8.sha256
ioerr() { dmesg --level=err | grep -c "I/O error"; }
before=$(ioerr)
echo "=== convert $L start $(date +%T) io_errors=$before disk=$(df --output=pcent / | tail -1)"
/opt/ep0/run.sh "TT_METAL_CACHE=/opt/ep0/jit_cache QWEN36_EXPERT_GU_BFP8_LAYERS=$L python -m models.demos.blackhole.drama3.tests.convert_gu_bfp8 --layers $L"
rc=$?
sync; echo 3 > /proc/sys/vm/drop_caches
for i in ${L//,/ }; do
  ( cd $CACHE && sha256sum layers.$i/moe.experts.gate_proj_dtype_BFLOAT8_B_layout_TILE.tensorbin layers.$i/moe.experts.up_proj_dtype_BFLOAT8_B_layout_TILE.tensorbin ) >> $SUMS
done
if diff <(sort -k2 /opt/ep0/day5/cache_bfp8.written.sha256) <(sort -k2 $SUMS) > /dev/null; then echo "sha256 disk == written"; else echo "SHA256_MISMATCH"; echo FAILED; exit 1; fi
after=$(ioerr)
echo "=== convert $L end $(date +%T) rc=$rc io_errors=$after disk=$(df --output=pcent / | tail -1)"
if [ "$after" != "$before" ]; then echo "IO_ERRORS_APPEARED"; echo FAILED; exit 1; fi
[ $rc = 0 ] && echo CONVERT_DONE || echo FAILED
