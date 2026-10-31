#!/bin/bash
# Episode 0 Day 2: fill the Drama 3 tensor cache one batch of layers at a time (host side), with the
# disk guard from the plan: dmesg I/O-error count before/after each batch, sync + drop_caches, then a
# sha256 of every cache file the batch wrote (read back from disk, not the page cache). Stops at the
# first I/O error. Each batch also runs the teacher-forced PCC check of its layers on the card.
#
#   convert_batches.sh "3,2" "1,4,5,6" ...      (each argument is one batch)
set -u
CACHE=/opt/ep0/cache/drama3/P150/tensor_cache_bfp8
LOG=/opt/ep0/day2/convert.log
SUMS=/opt/ep0/day2/cache.sha256
ioerr() { dmesg --level=err | grep -c "I/O error"; }

for batch in "$@"; do
  before=$(ioerr)
  echo "=== batch $batch start $(date +%T) io_errors=$before disk=$(df --output=pcent / | tail -1)" | tee -a $LOG
  t0=$(date +%s)
  /opt/ep0/run.sh "HF_MODEL=/opt/ep0/drama3_shim TT_CACHE_PATH=/opt/ep0/cache/drama3 python -m models.demos.blackhole.drama3.tt.device_check --layers $batch" >> $LOG 2>&1
  rc=$?
  sync; echo 3 > /proc/sys/vm/drop_caches
  for L in ${batch//,/ }; do
    ( cd $CACHE && find "layers.$L" "layers.$L".*.tensorbin -type f -print0 2>/dev/null | sort -z | xargs -0 sha256sum ) >> $SUMS
  done
  after=$(ioerr)
  echo "=== batch $batch end $(date +%T) rc=$rc secs=$(( $(date +%s) - t0 )) io_errors=$after cache=$(du -sh $CACHE | cut -f1)" | tee -a $LOG
  if [ "$after" != "$before" ]; then echo "IO_ERRORS_APPEARED stop" | tee -a $LOG; echo FAILED >> $LOG; exit 1; fi
  if [ $rc != 0 ]; then echo "batch failed rc=$rc" | tee -a $LOG; echo FAILED >> $LOG; exit 1; fi
done
echo "ALL_BATCHES_DONE" | tee -a $LOG
