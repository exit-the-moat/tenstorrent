cd /opt/ep0/port10
R=/opt/ep0/port10/codec/ref_default.pt
B=OUT_CH0:1,CAUSAL_PAD:1,SNAKE_KERNEL:1
TMO=420 codec/t1c.sh codec/r8_cd '' tests.bench_codec10 --ref $R --reps 9 --trace --cfgs snk=$B/snkcd=$B,CONV_OPTS:cdram
TMO=420 codec/t1c.sh codec/r8_w5 'DRAMA3_CODEC_WINDOWS=8,16,32,64,96 DRAMA3_CODEC_OUT_CH0=1 DRAMA3_CODEC_CAUSAL_PAD=1 DRAMA3_CODEC_SNAKE_KERNEL=1 DRAMA3_CODEC_CONV_OPTS=cdram' tests.bench_codec10 --ref $R --reps 9 --trace --sizes 8,16,32,64,96 --cfgs all=
echo R8_DONE
