# port10 prefetch: IENV = the --interactive env of run_gen.sh (read from the script, so it stays current)
IENV=$(grep -o 'inter="DRAMA3[^"]*"' /opt/ep0/tt-metal/models/demos/blackhole/drama3/run_gen.sh | head -1 | sed 's/^inter="//;s/"$//')
