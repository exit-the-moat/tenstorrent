cd /opt/ep0/port10
R=/opt/ep0/port10/codec/ref_default.pt
TMO=420 codec/t1c.sh codec/r12_w5 'DRAMA3_CODEC_WINDOWS=8,16,32,64,96 DRAMA3_CODEC_OUT_CH0=1 DRAMA3_CODEC_CAUSAL_PAD=1 DRAMA3_CODEC_SNAKE_KERNEL=1' tests.bench_codec10 --ref $R --reps 9 --trace --sizes 8,16,32,64,96 --cfgs all= --cross 4,8,10,16
echo R12_DONE
