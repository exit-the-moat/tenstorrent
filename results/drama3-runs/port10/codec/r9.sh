cd /opt/ep0/port10
R=/opt/ep0/port10/codec/ref_default.pt
B=OUT_CH0:1,CAUSAL_PAD:1,SNAKE_KERNEL:1
TMO=420 codec/t1c.sh codec/r9_sa '' tests.bench_codec10 --ref $R --reps 11 --trace --cfgs snk=$B/sa=$B,SNAKE_ADD:1
echo R9_DONE
