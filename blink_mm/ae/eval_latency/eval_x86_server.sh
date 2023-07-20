for num_threads in 1 2 4; do
python -u -m blink_mm.tvm.export.bins.batch_profiling_main \
    --report ./ae-output/layerwise-latency-report \
    --num-threads $num_threads \
    --tuning-records ~/elasticedge1/xiaohu/tvm-tuning-records \
    --bin-path ./ae-output/bins \
    --target x86 \
    --dev-name x86_server 2> /dev/null | tee ./ae-output/x86_server-$num_threads-threads.txt
done
