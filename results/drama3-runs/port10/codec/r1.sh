cd /opt/ep0/port10
R=/opt/ep0/port10/codec/ref_default.pt
TMO=600 ./t1.sh codec/r1_def '' tests.bench_codec10 --ref $R
TMO=600 ./t1.sh codec/r1_ch0 'DRAMA3_CODEC_OUT_CH0=1' tests.bench_codec10 --ref $R
TMO=600 ./t1.sh codec/r1_cpad 'DRAMA3_CODEC_CAUSAL_PAD=1' tests.bench_codec10 --ref $R
TMO=600 ./t1.sh codec/r1_both 'DRAMA3_CODEC_OUT_CH0=1 DRAMA3_CODEC_CAUSAL_PAD=1' tests.bench_codec10 --ref $R --profile
echo R1_DONE
