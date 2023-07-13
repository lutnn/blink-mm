python -m blink_mm.tvm.export.bins.batch_mem_profiling_main \
    --target arm \
    --model-bin-path ./ae-output/bins \
    --dev-name pixel4 \
    --serial "98281FFAZ009SV" \
    --csv-path ./ae-output/memory.csv > ./ae-output/memory.log