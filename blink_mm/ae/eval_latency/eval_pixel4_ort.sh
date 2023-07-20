python -u -m blink_mm.expers.eval_ort_latency \
    --ort-path ~/onnxruntime \
    --serial "98281FFAZ009SV" \
    --model-folder ./ae-output/onnx-models \
    --taskset 80 \
    --profile-folder ./ae-output/layerwise-latency-report \
    --dev-name pixel4 \
    --max-num-threads 1 | tee ./ae-output/pixel4-ort.txt