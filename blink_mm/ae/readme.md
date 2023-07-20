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

The raw **layerwise** profiling result is contained in `ae-output/layerwise-latency-report/`.
You can run the following command to generate diagrams to visualize the layerwise latency speedup
(`ae-output/layerwise_latency.pdf`) of LUT-NN:

```bash
bash blink_mm/ae/organize/organize_layerwise_latency.sh
```

The raw **end-to-end** latency data points are in `ae-output/${hardware}-${num_threads}-threads.txt`
and `ae-output/${hardware}-ort.txt`,
which contains results for LUT-NN/TVM and ONNX Runtime, respectively.
In the LUT-NN/TVM and ONNX Runtime raw latency data file 
(`ae-output/${hardware}-${num_threads}-threads.txt` and `ae-output/${hardware}-ort.txt`),
we use mean latency to measure the performance of these three frameworks.

For instance, the following shows an example output for LUT-NN/TVM raw latency file.
The latency of ResNet18 (CIFAR10 version) using TVM on Pixel6 with 1 thread is 22.23ms.
And the latency of ResNet18 (CIFAR10 version) using LUT-NN is 7.41ms (`amm` stands for LUT-NN).
Additionally, `amm_bert_last_6_layers` represents LUT-NN BERT model with 6 layers been replaced.

```
Evaluating resnet18_cifar with 1 threads
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
  22.2302      22.2737      22.4591      21.8357       0.2242   

Evaluating amm_resnet18_cifar with 1 threads
Execution time summary:
 mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
   7.4144       6.9913       9.0854       6.9372       0.8382 
```

The example output in ONNX Runtime raw latency file is as follows.
The latency of ResNet18 (CIFAR10 version) using ONNX Runtime with 1 thread is 19.918ms on Pixel6.
The memory usage of ResNet18 (CIFAR10 version) using ONNX Runtime with 1 thread is 104.46MB (109531136 bytes).

```
Evaluating resnet18_cifar on ONNX Runtime with 1 threads
Setting intra_op_num_threads to 1
Session creation time cost: 0.0508382 s
Total inference time cost: 1.9918 s
Total inference requests: 100
Average inference time cost: 19.918 ms
Total inference run time: 1.99202 s
Number of inferences per second: 50.2002 
Avg CPU usage: 12 %
Peak working set size: 109531136 bytes
Avg CPU usage:12
Peak working set size:109531136
Runs:100
Min Latency: 0.0197812 s
Max Latency: 0.0202704 s
P50 Latency: 0.0199047 s
P90 Latency: 0.0200251 s
P95 Latency: 0.0201534 s
P99 Latency: 0.0202704 s
P999 Latency: 0.0202704 s
```


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
If you would like to train the models by yourself, please refer to the [training recipes](training_recipes.md).
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

## Ablation Studies

To reproduce the ablation studies
(learnable temperature/
impact of the number of centroids and vector length/
the impact of the number of layers to replace for LUT-NN BERT)
results, please refer to the ablation study section of
the [training recipes](training_recipes.md).