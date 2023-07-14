# LUT-NN Artifact Evaluation for MobiCom 2023

This document contains instructions for reproducing the experiment results
for our MobiCom submission:
LUT-NN: Empower Efficient Neural Network Inference with Centroid Learning and Table Lookup.

## Latency Evaluation

First of all, run `mkdir ./ae-output` to prepare for AE outputs.

### Setup RPC for TVM

LUT-NN is implemented upon TVM.
To run LUT-NN on the local x86 machine, we only need to install TVM on it and launch the evaluation.
However, to run LUT-NN on Android phones, it is required to additionally
set up TVM RPC tracker on the local machine and RPC server on the Android phones.
Fortunately, the scripts to set up RPC have already been prepared in this repository.
Next, we will show you how to set up the RPC via several bash commands.

```bash
# Open one tmux session and run:
bash blink_mm/ae/setup_rpc_for_tvm/setup_rpc_tracker.sh
# Open another tmux session and run:
bash blink_mm/ae/setup_rpc_for_tvm/setup_pixel4.sh
# Open one last tmux session and run
# (note that you can configure the number of model inference threads via the environment variable):
NUM_THREADS=1 bash blink_mm/ae/setup_rpc_for_tvm/setup_pixel6.sh

# After running previous commands, run the following to check whether RPC is correctly settled down or not.
# You should see both Pixel4 and Pixel6 in the list.
python -m tvm.exec.query_rpc_tracker
```

Note that TVM RPC server does not support changing the number of model inference threads once set up.
So **if a different number of model inference threads is required**, we have to interrupt the rpc server and then re-setup. For example:

```bash
# Previous command:
#   NUM_THREADS=1 bash blink_mm/ae/setup_rpc_for_tvm/setup_pixel6.sh
# We use Ctrl+C to interrupt the command, and then execute:
NUM_THREADS=2 bash blink_mm/ae/setup_rpc_for_tvm/setup_pixel6.sh
# Now the RPC server on Pixel6 uses two threads.
```

### Evaluate Latency for the x86 Server

```bash
# Run the following command to compare LUT-NN model and the original model on x86 server with TVM.
# This command will automatically evaluate using different number of threads.
bash blink_mm/ae/eval_latency/eval_x86_server.sh

# Run the following command to evalute the original model's latency on x86 server with ONNX Runtime.
# This command will automatically evaluate using different number of threads.
bash blink_mm/ae/eval_latency/eval_x86_server_ort.sh
```

### Evaluate Latency for the Pixel4 Phone

After setting up RPC for Pixel4, we can start evaluating models' latency on it:

```bash
# Run the following command to compare LUT-NN model and the original model on Pixel4 with TVM.
# This command will evaluate with one model inference thread.
bash blink_mm/ae/eval_latency/eval_pixel4.sh

# Run the following command to evalute the original model's latency on Pixel4 with ONNX Runtime.
# This command will evaluate with one model inference thread.
bash blink_mm/ae/eval_latency/eval_pixel4_ort.sh
```

We only evaluate models on Pixel4 with one model inference thread
because Pixel4 phone only contains one large core (core with the largest frequency).

### Evaluate Latency for the Pixel6 Phone

After setting up RPC for Pixel6, we can start evaluating models' latency on it.
Pixel6 contains two large cores.
So we evaluate models' latency with 1 or 2 threads.

First, we evaluate the original models' latency on Pixel6 with ONNX Runtime:
```bash
# This command will automatically evaluate using different number of threads.
bash blink_mm/ae/eval_latency/eval_pixel6_ort.sh
```

Second, we evaluate LUT-NN models and the original models on Pixel6 with TVM:

```bash
# Command in the tmux session:
#   NUM_THREADS=1 bash blink_mm/ae/setup_rpc_for_tvm/setup_pixel6.sh
NUM_THREADS=1 bash blink_mm/ae/eval_latency/eval_pixel6.sh

# Switch the number of threads according to "Setup RPC for TVM" section.

# Command in the tmux session:
#   NUM_THREADS=2 bash blink_mm/ae/setup_rpc_for_tvm/setup_pixel6.sh
NUM_THREADS=2 bash blink_mm/ae/eval_latency/eval_pixel6.sh
```

### Output Files

The raw layerwise profiling result is contained in `ae-output/layerwise-latency-report/`.
You can run the following command to generate diagrams to visualize the layerwise latency speedup
(`ae-output/layerwise_latency.pdf`) of LUT-NN:

```bash
bash blink_mm/ae/organize/organize_layerwise_latency.sh
```

The raw latency data points are in `ae-output/${hardware}-${num_threads}-threads.txt`
and `ae-output/${hardware}-ort.txt`,
which contains results for LUT-NN/TVM and ONNX Runtime, respectively.
Note that the model name prefixed with `amm` stands for the LUT-NN accelerated model.
For example, `amm_bert_last_6_layers` stands for LUT-NN BERT model with 6 layers been replaced.

## GOPs and Disk Size Evaluation

**The evaluation of the disk size of the models must happen after the evaluation of latency**
because the model binaries must be firstly generated in the latency evaluation process.

To evaluate the GOPs of LUT-NN models and the original models, run:

```bash
bash blink_mm/ae/eval_ops.sh
```

To evaluate the disk size of LUT-NN models and the original models, run:

```bash
bash blink_mm/ae/eval_disk_size.sh
```

The GOPs result is contained in `ae-output/count_cnn.csv` and `ae-output/count_bert.csv`.
In `ae-output/count_cnn.csv`, OPs corresponds to `muladds` column plus `adds` column.
The disk size result is contained in `ae-output/disk_size.csv`.

## Accuracy Evaluation

To evaluate the accuracy of LUT-NN models, versus MADDNESS and the original models, run:

```bash
bash blink_mm/ae/eval_bert_accuracy.sh
bash blink_mm/ae/eval_cnn_accuracy.sh
```

Note that these two scripts only read in the pre-evaluated accuracy in the PyTorch checkpoint file
since this AE server does not have a GPU.
After evaluation, the accuracy data will be put in `ae-output/bert_accuracy.csv` and `ae-output/cnn_accuracy.csv`.
The `amm` prefixed model names stand for LUT-NN models.
And the `maddness` prefixed model names stand for MADDNESS models.

## Power and Memory Evaluation

**The evaluation of the power and memory of the models must happen after the evaluation of latency**
because the model binaries must be firstly generated in the latency evaluation process.

To evaluate the memory consumption of LUT-NN/TVM models, run:

```bash
bash blink_mm/ae/eval_memory.sh
```

The memory consumption of ONNX Runtime models is contained in `ae-output/${hardware}-ort.txt`
(Peak working set size).

To evaluate the power consumption of LUT-NN/TVM models, run:

```bash
bash blink_mm/ae/eval_power.sh
```

Finally, to compute the average power consumption of LUT-NN/TVM models, run:

```bash
bash blink_mm/ae/organize/organize_power.sh
```

The average power is in `ae-output/power-organized.csv`.
