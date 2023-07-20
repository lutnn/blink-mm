python -u -m blink_mm.expers.eval_ort_latency \
    --ort-path ~/onnxruntime \
    --serial "1B261FDF6009KS" \
    --model-folder ./ae-output/onnx-models \
    --taskset C0 \
    --profile-folder ./ae-output/layerwise-latency-report \
    --dev-name pixel6 \
    --max-num-threads 2 | tee ./ae-output/pixel6-ort.txt