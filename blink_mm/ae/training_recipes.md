# LUT-NN Training Recipes

To train LUT-NN models, firstly you must train a corresponding vanilla baseline model,
then you can convert the baseline model's checkpoint to a LUT-NN model by finetuning.

## The Baseline Models

```bash
# CIFAR10 models

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type cifar10 \
    --lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/resnet18_cifar-cifar10 \
    --device-ids 0 \
    --num-procs 1 \
    --model-type resnet18_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'SGD','momentum':0.9,'weight_decay':5e-4}" \
    --log-interval 256

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type cifar10 \
    --lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/senet18_cifar-cifar10 \
    --device-ids 0 \
    --num-procs 1 \
    --model-type senet18_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'SGD','momentum':0.9,'weight_decay':5e-4}" \
    --log-interval 256

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type cifar10 \
    --lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/vgg11_cifar-cifar10 \
    --device-ids 0 \
    --num-procs 1 \
    --model-type vgg11_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'SGD','momentum':0.9,'weight_decay':5e-4}" \
    --log-interval 256

# GTSRB models
python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type gtsrb \
    --lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/resnet18_cifar-gtsrb \
    --device-ids 0 \
    --num-procs 1 \
    --model-type resnet18_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'SGD','momentum':0.9,'weight_decay':5e-4}" \
    --log-interval 256

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type gtsrb \
    --lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/senet18_cifar-gtsrb \
    --device-ids 0 \
    --num-procs 1 \
    --model-type senet18_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'SGD','momentum':0.9,'weight_decay':5e-4}" \
    --log-interval 256

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type gtsrb \
    --lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/vgg11_cifar-gtsrb \
    --device-ids 0 \
    --num-procs 1 \
    --model-type vgg11_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'SGD','momentum':0.9,'weight_decay':5e-4}" \
    --log-interval 256

# Speech Commands models
python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type speech_commands \
    --lr 1e-2 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/resnet18_cifar-speech_commands \
    --device-ids 0 \
    --num-procs 1 \
    --model-type resnet18_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'SGD','momentum':0.9,'weight_decay':5e-4}" \
    --log-interval 256

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type speech_commands \
    --lr 1e-2 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/senet18_cifar-speech_commands \
    --device-ids 0 \
    --num-procs 1 \
    --model-type senet18_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'SGD','momentum':0.9,'weight_decay':5e-4}" \
    --log-interval 256

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type speech_commands \
    --lr 1e-2 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/vgg11_cifar-speech_commands \
    --device-ids 0 \
    --num-procs 1 \
    --model-type vgg11_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'SGD','momentum':0.9,'weight_decay':5e-4}" \
    --log-interval 256

# SVHN models
python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type svhn \
    --lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/resnet18_cifar-svhn \
    --device-ids 0 \
    --num-procs 1 \
    --model-type resnet18_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'SGD','momentum':0.9,'weight_decay':5e-4}" \
    --log-interval 256

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type svhn \
    --lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/senet18_cifar-svhn \
    --device-ids 0 \
    --num-procs 1 \
    --model-type senet18_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'SGD','momentum':0.9,'weight_decay':5e-4}" \
    --log-interval 256

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type svhn \
    --lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/vgg11_cifar-svhn \
    --device-ids 0 \
    --num-procs 1 \
    --model-type vgg11_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'SGD','momentum':0.9,'weight_decay':5e-4}" \
    --log-interval 256

# UTKFace models
python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type utk_face \
    --lr 1e-3 \
    --num-epochs 90 \
    --work-dir ${CKPT_FOLDER}/resnet18-utk_face \
    --device-ids 0 \
    --num-procs 1 \
    --model-type resnet18 \
    --lr-scheduler "{'name':'StepLR','step_size_epochs':30,'gamma':0.1}" \
    --optimizer "{'name':'SGD','momentum':0.9,'weight_decay':1e-4}" \
    --log-interval 256 \
    --checkpoint-hook "{'save_best':'mae','compare_op':'less'}"

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type utk_face \
    --lr 1e-3 \
    --num-epochs 90 \
    --work-dir ${CKPT_FOLDER}/senet18-utk_face \
    --device-ids 0 \
    --num-procs 1 \
    --model-type senet18 \
    --lr-scheduler "{'name':'StepLR','step_size_epochs':30,'gamma':0.1}" \
    --optimizer "{'name':'SGD','momentum':0.9,'weight_decay':1e-4}" \
    --log-interval 256 \
    --checkpoint-hook "{'save_best':'mae','compare_op':'less'}"

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type utk_face \
    --lr 1e-3 \
    --num-epochs 90 \
    --work-dir ${CKPT_FOLDER}/vgg11_bn-utk_face \
    --device-ids 0 \
    --num-procs 1 \
    --model-type vgg11_bn \
    --lr-scheduler "{'name':'StepLR','step_size_epochs':30,'gamma':0.1}" \
    --optimizer "{'name':'SGD','momentum':0.9,'weight_decay':1e-4}" \
    --log-interval 256 \
    --checkpoint-hook "{'save_best':'mae','compare_op':'less'}"

# ImageNet models

# For ResNet18 models, please use ResNet18 from torchvision

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 32 \
    --root ./datasets/imagenet-raw-data \
    --dataset-type imagenet \
    --lr 1e-1 \
    --num-epochs 90 \
    --work-dir ${CKPT_FOLDER}/senet18-imagenet \
    --device-ids 0,1,2,3,4,5,6,7 \
    --num-procs 8 \
    --model-type senet18 \
    --lr-scheduler "{'name':'StepLR','step_size_epochs':30,'gamma':0.1}" \
    --optimizer "{'name':'SGD','momentum':0.9,'weight_decay':1e-4}" \
    --log-interval 256

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 32 \
    --root ./datasets/imagenet-raw-data \
    --dataset-type imagenet \
    --lr 1e-1 \
    --num-epochs 90 \
    --work-dir ${CKPT_FOLDER}/vgg11_bn-imagenet \
    --device-ids 0,1,2,3,4,5,6,7 \
    --num-procs 8 \
    --model-type vgg11_bn \
    --lr-scheduler "{'name':'StepLR','step_size_epochs':30,'gamma':0.1}" \
    --optimizer "{'name':'SGD','momentum':0.9,'weight_decay':1e-4}" \
    --log-interval 256

# BERT for GLUE
python -m blink_mm.expers.train_glue \
    --batch-size-per-gpu 32 \
    --work-dir ${CKPT_FOLDER}/bert \
    --device-ids 0 \
    --num-procs 1 \
    --model-type bert
```

## LUT-NN Models

The hardware requirements for training LUT-NN models are listed below:

|Dataset|LUT-NN Model|GPU Requirement|Estimated Time|
|-|-|-|-|
|CIFAR10, GTSRB, Speech Commands, SVHN|ResNet18, SENet18, VGG11|1x A100|24 hours|
|UTKFace|ResNet18, SENet18|4x A6000|9.5 hours|
|UTKFace|VGG11|4x A100|12 hours|
|ImageNet|ResNet18, SENet18|8x V100|1 week|
|ImageNet|VGG11|8x A100|3.5 days|
|GLUE|BERT|4x V100|several hours|

```bash
# CIFAR10 models

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type cifar10 \
    --lr 1e-3 \
    --temp-lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/amm_resnet18_cifar-cifar10 \
    --device-ids 0 \
    --num-procs 1 \
    --ckpt-path ${CKPT_FOLDER}/resnet18_cifar-cifar10/epoch_200.pth \
    --model-type amm_resnet18_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}" \
    --log-interval 256

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type cifar10 \
    --lr 1e-3 \
    --temp-lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/amm_senet18_cifar-cifar10 \
    --device-ids 0 \
    --num-procs 1 \
    --ckpt-path ${CKPT_FOLDER}/senet18_cifar-cifar10/epoch_200.pth \
    --model-type amm_senet18_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}" \
    --log-interval 256

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type cifar10 \
    --lr 1e-3 \
    --temp-lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/amm_vgg11_cifar-cifar10 \
    --device-ids 0 \
    --num-procs 1 \
    --ckpt-path ${CKPT_FOLDER}/vgg11_cifar-cifar10/epoch_200.pth \
    --model-type amm_vgg11_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}" \
    --subvec-len 9 \
    --log-interval 256

# GTSRB models

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type gtsrb \
    --lr 1e-3 \
    --temp-lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/amm_resnet18_cifar-gtsrb \
    --device-ids 0 \
    --num-procs 1 \
    --ckpt-path ${CKPT_FOLDER}/resnet18_cifar-gtsrb/epoch_200.pth \
    --model-type amm_resnet18_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}" \
    --log-interval 256

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type gtsrb \
    --lr 1e-3 \
    --temp-lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/amm_senet18_cifar-gtsrb \
    --device-ids 0 \
    --num-procs 1 \
    --ckpt-path ${CKPT_FOLDER}/senet18_cifar-gtsrb/epoch_200.pth \
    --model-type amm_senet18_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}" \
    --log-interval 256

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type gtsrb \
    --lr 1e-3 \
    --temp-lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/amm_vgg11_cifar-gtsrb \
    --device-ids 0 \
    --num-procs 1 \
    --ckpt-path ${CKPT_FOLDER}/vgg11_cifar-gtsrb/epoch_200.pth \
    --model-type amm_vgg11_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}" \
    --subvec-len 9 \
    --log-interval 256

# Speech Commands models

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type speech_commands \
    --lr 1e-4 \
    --temp-lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/amm_resnet18_cifar-speech_commands \
    --device-ids 0 \
    --num-procs 1 \
    --ckpt-path ${CKPT_FOLDER}/resnet18_cifar-speech_commands/epoch_200.pth \
    --model-type amm_resnet18_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}" \
    --log-interval 256

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type speech_commands \
    --lr 1e-4 \
    --temp-lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/amm_senet18_cifar-speech_commands \
    --device-ids 0 \
    --num-procs 1 \
    --ckpt-path ${CKPT_FOLDER}/senet18_cifar-speech_commands/epoch_200.pth \
    --model-type amm_senet18_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}" \
    --log-interval 256

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type speech_commands \
    --lr 1e-4 \
    --temp-lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/amm_vgg11_cifar-speech_commands \
    --device-ids 0 \
    --num-procs 1 \
    --ckpt-path ${CKPT_FOLDER}/vgg11_cifar-speech_commands/epoch_200.pth \
    --model-type amm_vgg11_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}" \
    --subvec-len 9 \
    --log-interval 256

# SVHN

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type svhn \
    --lr 1e-3 \
    --temp-lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/amm_resnet18_cifar-svhn \
    --device-ids 0 \
    --num-procs 1 \
    --ckpt-path ${CKPT_FOLDER}/resnet18_cifar-svhn/epoch_200.pth \
    --model-type amm_resnet18_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}" \
    --log-interval 256

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type svhn \
    --lr 1e-3 \
    --temp-lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/amm_senet18_cifar-svhn \
    --device-ids 0 \
    --num-procs 1 \
    --ckpt-path ${CKPT_FOLDER}/senet18_cifar-svhn/epoch_200.pth \
    --model-type amm_senet18_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}" \
    --log-interval 256

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type svhn \
    --lr 1e-3 \
    --temp-lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/amm_vgg11_cifar-svhn \
    --device-ids 0 \
    --num-procs 1 \
    --ckpt-path ${CKPT_FOLDER}/vgg11_cifar-svhn/epoch_200.pth \
    --model-type amm_vgg11_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}" \
    --subvec-len 9 \
    --log-interval 256

# UTKFace
python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type utk_face \
    --lr 1e-4 \
    --temp-lr 1e-1 \
    --num-epochs 150 \
    --work-dir ${CKPT_FOLDER}/amm_resnet18-utk_face \
    --device-ids 0 \
    --num-procs 1 \
    --ckpt-path ${CKPT_FOLDER}/resnet18-utk_face/epoch_90.pth \
    --model-type amm_resnet18 \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}" \
    --log-interval 256 \
    --checkpoint-hook "{'save_best':'mae','compare_op':'less'}"

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type utk_face \
    --lr 1e-4 \
    --temp-lr 1e-1 \
    --num-epochs 150 \
    --work-dir ${CKPT_FOLDER}/amm_senet18-utk_face \
    --device-ids 0 \
    --num-procs 1 \
    --ckpt-path ${CKPT_FOLDER}/senet18-utk_face/epoch_90.pth \
    --model-type amm_senet18 \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}" \
    --log-interval 256 \
    --checkpoint-hook "{'save_best':'mae','compare_op':'less'}"

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 32 \
    --root ./datasets \
    --dataset-type utk_face \
    --lr 1e-4 \
    --temp-lr 1e-1 \
    --num-epochs 150 \
    --work-dir ${CKPT_FOLDER}/amm_vgg11_bn-utk_face \
    --device-ids 0,1,2,3,4,5,6,7 \
    --num-procs 8 \
    --ckpt-path ${CKPT_FOLDER}/vgg11_bn-utk_face/epoch_90.pth \
    --model-type amm_vgg11_bn \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}" \
    --subvec-len 9 \
    --log-interval 256 \
    --checkpoint-hook "{'save_best':'mae','compare_op':'less'}"

# ImageNet
python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 32 \
    --root ./datasets/imagenet-raw-data \
    --dataset-type imagenet \
    --lr 1e-3 \
    --temp-lr 1e-1 \
    --num-epochs 150 \
    --work-dir ${CKPT_FOLDER}/amm_resnet18-imagenet \
    --device-ids 0,1,2,3,4,5,6,7 \
    --num-procs 8 \
    --ckpt-path IMAGENET1K_V1 \
    --model-type amm_resnet18 \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}" \
    --log-interval 256

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 32 \
    --root ./datasets/imagenet-raw-data \
    --dataset-type imagenet \
    --lr 1e-3 \
    --temp-lr 1e-1 \
    --num-epochs 150 \
    --work-dir ${CKPT_FOLDER}/amm_senet18-imagenet \
    --device-ids 0,1,2,3,4,5,6,7 \
    --num-procs 8 \
    --ckpt-path ${CKPT_FOLDER}/senet18-imagenet/epoch_90.pth \
    --model-type amm_senet18 \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}" \
    --log-interval 256

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 32 \
    --root ./datasets/imagenet-raw-data \
    --dataset-type imagenet \
    --lr 1e-3 \
    --temp-lr 1e-1 \
    --num-epochs 150 \
    --work-dir ${CKPT_FOLDER}/amm_vgg11_bn-imagenet \
    --device-ids 0,1,2,3,4,5,6,7 \
    --num-procs 8 \
    --ckpt-path ${CKPT_FOLDER}/vgg11_bn-imagenet/epoch_90.pth \
    --model-type amm_vgg11_bn \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}" \
    --subvec-len 9 \
    --log-interval 256

# GLUE models

python -m blink_mm.expers.train_glue \
    --batch-size-per-gpu 32 \
    --temp-lr 1e-1 \
    --work-dir ${CKPT_FOLDER}/amm_bert \
    --device-ids 0 \
    --num-procs 1 \
    --ckpt-path ${CKPT_FOLDER}/bert \
    --model-type amm_bert
```

## Ablation Study

```bash
# Learnable Temperature

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type cifar10 \
    --lr 1e-3 \
    --temp-lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/temperature-ablation-study/amm_resnet18_cifar-cifar10/annealing-temp \
    --device-ids 0 \
    --num-procs 1 \
    --ckpt-path ${CKPT_FOLDER}/resnet18_cifar-cifar10/epoch_200.pth \
    --model-type amm_resnet18_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}" \
    --log-interval 256 \
    --temperature-config manual \
    --temperature 1 1e-1

python -m blink_mm.expers.train_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type cifar10 \
    --lr 1e-3 \
    --temp-lr 1e-1 \
    --num-epochs 200 \
    --work-dir ${CKPT_FOLDER}/temperature-ablation-study/amm_resnet18_cifar-cifar10/const-temp \
    --device-ids 0 \
    --num-procs 1 \
    --ckpt-path ${CKPT_FOLDER}/resnet18_cifar-cifar10/epoch_200.pth \
    --model-type amm_resnet18_cifar \
    --lr-scheduler "{'name':'CosineAnnealingLR'}" \
    --optimizer "{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}" \
    --log-interval 256 \
    --temperature-config manual \
    --temperature 1 1

# Impact of the number of centroids and vector length

python -m blink_mm.expers.search.grid_search_cnn \
    --imgs-per-gpu 256 \
    --root ./datasets \
    --dataset-type cifar10 \
    --lr 1e-3 \
    --temp-lr 1e-1 \
    --num-epochs 50 \
    --work-dir ${CKPT_FOLDER}/kv-grid_search/amm_resnet18_cifar-cifar10 \
    --device-ids 0 \
    --num-procs 1 \
    --ckpt-path ${CKPT_FOLDER}/resnet18_cifar-cifar10/epoch_200.pth \
    --model-type amm_resnet18_cifar \
    --temperature-config inverse \
    --temperature 0.065

# The impact of the number of layers to replace for LUT-NN BERT

python -m blink_mm.expers.search.grid_search_glue \
    --batch-size-per-gpu 32 \
    --temp-lr 1e-1 \
    --work-dir ${CKPT_FOLDER}/num_layers-grid_search/amm_bert \
    --device-ids 0 \
    --num-procs 1 \
    --ckpt-dir ${CKPT_FOLDER}/bert \
    --dataset-type stsb \
    --grid-search-type layers
```
