cd /opt/ep0/port10
R=/opt/ep0/port10/codec/ref_default.pt
B=OUT_CH0:1,CAUSAL_PAD:1
TMO=600 ./t1.sh codec/r3_snk '' tests.bench_codec10 --ref $R --reps 11 --cfgs both=$B/snk=$B,SNAKE_KERNEL:1/snk_only=SNAKE_KERNEL:1
TMO=600 ./t1.sh codec/r3_opts '' tests.bench_codec10 --ref $R --reps 9 --cfgs both=$B/adb=$B,CONV_OPTS:adb/wdb=$B,CONV_OPTS:wdb/split=$B,CONV_OPTS:split/full=$B,CONV_OPTS:full/abh64=$B,CONV_OPTS:abh=64/abh128=$B,CONV_OPTS:abh=128
echo R3_DONE
