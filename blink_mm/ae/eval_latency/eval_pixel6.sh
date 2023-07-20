NUM_THREADS="${NUM_THREADS:=1}"

python -u -m blink_mm.tvm.export.bins.batch_profiling_main \
    --report ./ae-output/layerwise-latency-report \
    --num-threads $NUM_THREADS \
    --tuning-records ~/elasticedge1/xiaohu/tvm-tuning-records \
    --bin-path ./ae-output/bins \
    --target arm \
    --dev-name pixel6 \
    --host "127.0.0.1" \
    --port 9190 \
    --key pixel6 2> /dev/null | tee ./ae-output/pixel6-$NUM_THREADS-threads.txt
