# LUT-NN

## Installation

### Install Only for Training

If you would like to train LUT-NN, then simply run:

```bash
pip install -r requirements.txt
```

### Install Also for Deployment

If you want to deploy LUT-NN to real hardware, please install according to the following steps besides installing Python requirements:

1. First, install [TVM for LUT-NN](https://github.com/lutnn/tvm-dpq) according to its readme.

2. Then, build `energy_profiler` and `profile_tvm_model` using these instructions:

```bash
mkdir build
cd build
cmake -DUSE_TVM=${LOCAL_PATH_TO_TVM} \
    -DCMAKE_TOOLCHAIN_FILE=${LOCAL_PATH_TO_NDK_TOOLCHAIN_FILE} \
    -DANDROID_ABI=arm64-v8a \
    ..
make energy_profiler -j4
make profile_tvm_model -j4
```

### Download Datasets

To download datasets, please run:

```bash
python -m blink_mm.data.download_datasets
```

This instruction will download CIFAR10, GTSRB, SVHN and Speech Commands datasets into the `./datasets` folder.
Otherwise, if you would like to train LUT-NN on UTKFace or ImageNet,
please manually download these two datasets to `./datasets/UTKFace` and `./datasets/imagenet-raw-data` respectively.

## Artifact Evaluation

Please refer to the [AE README](blink_mm/ae/readme.md).

## Training Recipes

Please refer to the [Training Recipes](blink_mm/ae/training_recipes.md).
