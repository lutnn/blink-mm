python -m blink_mm.expers.eval_ort_latency \
    --ort-path ~/onnxruntime \
    --model-folder ./ae-output/onnx-models \
    --profile-folder ./ae-output/layerwise-latency-report \
    --dev-name x86_server \
    --max-num-threads 4 > ./ae-output/x86_server-ort.txt