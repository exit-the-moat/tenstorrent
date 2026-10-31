cd /opt/ep0/port10
W='DRAMA3_CODEC_WINDOWS=8,16,32,64,96'
TMO=420 codec/t1c.sh codec/r6_def "$W" tests.bench_codec10 --reps 3 --sizes 8,16 --cfgs def= --cross 4,8,10,16
echo R6_DONE
