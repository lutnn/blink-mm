import torch.nn as nn
import torch
import torchvision

from qat.networks.cnn_wrapper import CNNWrapper

import blink_mm.networks.resnet_large_cifar.resnet as resnet_large_cifar_resnet
import blink_mm.networks.resnet_large_cifar.amm_resnet as resnet_large_cifar_amm_resnet
import blink_mm.networks.vgg_cifar.vgg as vgg_cifar_vgg
import blink_mm.networks.vgg_cifar.amm_vgg as vgg_cifar_amm_vgg
import blink_mm.networks.senet_cifar.senet as senet_cifar_senet
import blink_mm.networks.senet_cifar.amm_senet as senet_cifar_amm_senet

import blink_mm.networks.resnet_large_imagenet.amm_resnet as resnet_large_imagenet_amm_resnet
import blink_mm.networks.vgg_imagenet.vgg as vgg_imagenet_vgg
import blink_mm.networks.vgg_imagenet.amm_vgg as vgg_imagenet_amm_vgg
import blink_mm.networks.senet_imagenet.senet as senet_imagenet_senet
import blink_mm.networks.senet_imagenet.amm_senet as senet_imagenet_amm_senet

from blink_mm.networks.seq_models.bert import BERT
from blink_mm.networks.seq_models.amm_bert import AMMBERT


# CIFAR10 models

def resnet18_cifar(ckpt_path=None):
    model = resnet_large_cifar_resnet.resnet18_cifar()
    model = CNNWrapper(model, "cpu")
    model.eval()
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt)
    return model


def amm_resnet18_cifar(ckpt_path=None):
    model = resnet_large_cifar_amm_resnet.amm_resnet18_cifar()
    model = CNNWrapper(model, "cpu")
    model.eval()
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt)
    return model


def vgg11_cifar(ckpt_path=None):
    model = vgg_cifar_vgg.vgg11_cifar()
    model = CNNWrapper(model, "cpu")
    model.eval()
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt)
    return model


def amm_vgg11_cifar(ckpt_path=None):
    model = vgg_cifar_amm_vgg.amm_vgg11_cifar()
    model = CNNWrapper(model, "cpu")
    model.eval()
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt)
    return model


def senet18_cifar(ckpt_path=None):
    model = senet_cifar_senet.senet18_cifar()
    model = CNNWrapper(model, "cpu")
    model.eval()
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt)
    return model


def amm_senet18_cifar(ckpt_path=None):
    model = senet_cifar_amm_senet.amm_senet18_cifar()
    model = CNNWrapper(model, "cpu")
    model.eval()
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt)
    return model


# ImageNet models

def resnet18():
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.to("cpu")
    model.eval()
    return model


def amm_resnet18(ckpt_path=None):
    model = resnet_large_imagenet_amm_resnet.amm_resnet18()
    model = CNNWrapper(model, "cpu")
    model.eval()
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt)
    return model


def vgg11_bn(ckpt_path=None):
    model = vgg_imagenet_vgg.vgg11_bn()
    model = CNNWrapper(model, "cpu")
    model.eval()
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt)
    return model


def amm_vgg11_bn(ckpt_path=None):
    model = vgg_imagenet_amm_vgg.amm_vgg11_bn()
    model = CNNWrapper(model, "cpu")
    model.eval()
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt)
    return model


def senet18(ckpt_path=None):
    model = senet_imagenet_senet.senet18()
    model = CNNWrapper(model, "cpu")
    model.eval()
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt)
    return model


def amm_senet18(ckpt_path=None):
    model = senet_imagenet_amm_senet.amm_senet18()
    model = CNNWrapper(model, "cpu")
    model.eval()
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt)
    return model

# bert


def bert_last_6_layers():
    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            model = BERT(2, num_hidden_layers=6, torchscript=True)
            self.encoder = model.transformer.bert.encoder

        def forward(self, x):
            return self.encoder(x).last_hidden_state

    return Wrapper()


def amm_bert_last_6_layers():
    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            model = AMMBERT(2, num_hidden_layers=6, torchscript=True)
            self.encoder = model.transformer.bert.encoder

        def forward(self, x):
            return self.encoder(x).last_hidden_state

    return Wrapper()


def amm_bert_for_layerwise_benchmark():
    from blink_mm.ops.amm_linear import AMMLinear

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            subvec_len = 32
            self.linear0 = AMMLinear(768 // subvec_len, 768, 768, True)
            self.linear1 = AMMLinear(768 // subvec_len, 768, 3072, True)
            self.linear2 = AMMLinear(3072 // subvec_len, 3072, 768, True)

        def forward(self, x):
            x = self.linear0(x)
            x = self.linear1(x)
            x = self.linear2(x)
            return x

    return Wrapper()


MODEL_ARCHIVE = {
    # CIFAR10 models
    "resnet18_cifar": {
        "model": resnet18_cifar,
        "input": (torch.randn(1, 3, 32, 32), ),
    },
    "amm_resnet18_cifar": {
        "model": amm_resnet18_cifar,
        "input": (torch.randn(1, 3, 32, 32), ),
    },
    "vgg11_cifar": {
        "model": vgg11_cifar,
        "input": (torch.randn(1, 3, 32, 32), ),
    },
    "amm_vgg11_cifar": {
        "model": amm_vgg11_cifar,
        "input": (torch.randn(1, 3, 32, 32), ),
    },
    "senet18_cifar": {
        "model": senet18_cifar,
        "input": (torch.randn(1, 3, 32, 32), ),
    },
    "amm_senet18_cifar": {
        "model": amm_senet18_cifar,
        "input": (torch.randn(1, 3, 32, 32), ),
    },

    # ImageNet models
    "resnet18": {
        "model": resnet18,
        "input": (torch.randn(1, 3, 224, 224),),
    },
    "amm_resnet18": {
        "model": amm_resnet18,
        "input": (torch.randn(1, 3, 224, 224),),
    },
    "vgg11_bn": {
        "model": vgg11_bn,
        "input": (torch.randn(1, 3, 224, 224),),
    },
    "amm_vgg11_bn": {
        "model": amm_vgg11_bn,
        "input": (torch.randn(1, 3, 224, 224),),
    },
    "senet18": {
        "model": senet18,
        "input": (torch.randn(1, 3, 224, 224),),
    },
    "amm_senet18": {
        "model": amm_senet18,
        "input": (torch.randn(1, 3, 224, 224),),
    },

    # BERT
    "bert_last_6_layers": {
        "model": bert_last_6_layers,
        "input": (torch.randn(1, 64, 768),)
    },
    "amm_bert_last_6_layers": {
        "model": amm_bert_last_6_layers,
        "input": (torch.randn(1, 64, 768),)
    },
    "amm_bert_for_layerwise_benchmark": {
        "model": amm_bert_for_layerwise_benchmark,
        "input": (torch.randn(1, 64, 768),)
    }
}
