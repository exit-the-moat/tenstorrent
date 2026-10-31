cd /opt/ep0/port10
R=/opt/ep0/port10/codec/ref_default.pt
B=OUT_CH0:1,CAUSAL_PAD:1
TMO=600 ./t1.sh codec/r2_ab '' tests.bench_codec10 --ref $R --reps 15 --cfgs def=/ch0=OUT_CH0:1/cpad=CAUSAL_PAD:1/both=$B
TMO=600 ./t1.sh codec/r2_opts '' tests.bench_codec10 --ref $R --reps 11 --cfgs both=$B/adb=$B,CONV_OPTS:adb/wdb=$B,CONV_OPTS:wdb/adbwdb=$B,CONV_OPTS:adb+wdb/split=$B,CONV_OPTS:split
echo R2_DONE
