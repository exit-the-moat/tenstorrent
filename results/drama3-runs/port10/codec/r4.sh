cd /opt/ep0/port10
R=/opt/ep0/port10/codec/ref_default.pt
B=OUT_CH0:1,CAUSAL_PAD:1
TMO=600 ./t1.sh codec/r4_tr '' tests.bench_codec10 --ref $R --reps 11 --trace --cfgs def=/both=$B/snk=$B,SNAKE_KERNEL:1
echo R4_DONE
