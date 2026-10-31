cd /opt/ep0/port10
R=/opt/ep0/port10/codec/ref_default.pt
B=OUT_CH0:1,CAUSAL_PAD:1,SNAKE_KERNEL:1
TMO=420 codec/t1c.sh codec/r10_split '' tests.bench_codec10 --ref $R --reps 9 --trace --cfgs snk=$B/split=$B,CONV_OPTS:split
TMO=420 codec/t1c.sh codec/r10_full '' tests.bench_codec10 --ref $R --reps 9 --trace --cfgs snk=$B/full=$B,CONV_OPTS:full
TMO=420 codec/t1c.sh codec/r10_hifi2 'DRAMA3_CODEC_FIDELITY=HiFi2' tests.bench_codec10 --ref $R --reps 9 --trace --cfgs hifi2=$B
echo R10_DONE
