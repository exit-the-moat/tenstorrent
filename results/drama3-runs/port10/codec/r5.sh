cd /opt/ep0/port10
R=/opt/ep0/port10/codec/ref_default.pt
B=OUT_CH0:1,CAUSAL_PAD:1,SNAKE_KERNEL:1
TMO=600 ./t1.sh codec/r5_prof '' tests.bench_codec10 --ref $R --reps 3 --profile --sizes 32,96 --cfgs snk=$B
TMO=600 ./t1.sh codec/r5_win 'DRAMA3_CODEC_WINDOWS=8,16,32,64,96' tests.bench_codec10 --reps 11 --trace --sizes 8,16,32 --cfgs snk=$B/snkcd=$B,CONV_OPTS:cdram
echo R5_DONE
