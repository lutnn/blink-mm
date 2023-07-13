# LUT-NN Training Recipes

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
    --root ./datasets \
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
    --root ./datasets \
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
