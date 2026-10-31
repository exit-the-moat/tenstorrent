cd /opt/ep0/port10
W='DRAMA3_CODEC_WINDOWS=8,16,32,64,96'
TMO=300 codec/t1c.sh codec/r7_b "$W" tests.bench_codec10 --reps 3 --sizes 8,16 --cfgs b=OUT_CH0:1,CAUSAL_PAD:1
TMO=300 codec/t1c.sh codec/r7_s "$W" tests.bench_codec10 --reps 3 --sizes 8,16 --cfgs s=SNAKE_KERNEL:1
echo R7_DONE
