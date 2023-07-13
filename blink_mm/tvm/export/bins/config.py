def batch_config(target):
    # model, tuner, tuning_records, report, tune for different thread number

    assert target in ["arm", "x86", "x86_avx512"]

    ls = []

    # CIFAR10 models
    ls.extend([
        ["resnet18_cifar", "autotvm",
            "resnet18_cifar.json", "resnet18_cifar.csv", False],
        ["amm_resnet18_cifar", "autotvm",
            "amm_resnet18_cifar.json", "amm_resnet18_cifar.csv", True],
        ["vgg11_cifar", "autotvm",
            "vgg11_cifar.json", "vgg11_cifar.csv", False],
        ["amm_vgg11_cifar", "autotvm",
            "amm_vgg11_cifar.json", "amm_vgg11_cifar.csv", True],
        ["senet18_cifar", "autotvm",
            "senet18_cifar.json", "senet18_cifar.csv", False],
        ["amm_senet18_cifar", "autotvm",
            "amm_senet18_cifar.json", "amm_senet18_cifar.csv", True],
    ])

    # ImageNet models
    ls.extend([
        ["resnet18", "autotvm",
            "resnet18.json", "resnet18.csv", False],
        ["amm_resnet18", "autotvm",
            "amm_resnet18.json", "amm_resnet18.csv", True],
        ["vgg11_bn", "autotvm",
            "vgg11_bn.json", "vgg11_bn.csv", False],
        ["amm_vgg11_bn", "autotvm",
            "amm_vgg11_bn.json", "amm_vgg11_bn.csv", True],
        ["senet18", "autotvm",
            "senet18.json", "senet18.csv", False],
        ["amm_senet18", "autotvm",
            "amm_senet18.json", "amm_senet18.csv", True],
    ])

    # BERT
    if target == "arm":
        ls.extend([
            ["bert_last_6_layers", "auto_scheduler",
             "bert_last_6_layers.json", "bert_last_6_layers.csv", True],
            ["amm_bert_last_6_layers", "autotvm",
             "amm_bert_last_6_layers.json", "amm_bert_last_6_layers.csv", True],
            ["amm_bert_for_layerwise_benchmark", "autotvm",
                "amm_bert_for_layerwise_benchmark.json", "amm_bert_for_layerwise_benchmark.csv", True],
        ])
    elif target in ["x86", "x86_avx512"]:
        ls.extend([
            ["bert_last_6_layers", "autotvm",
             "bert_last_6_layers.json", "bert_last_6_layers.csv", False],
            ["amm_bert_last_6_layers", "autotvm",
             "amm_bert_last_6_layers.json", "amm_bert_last_6_layers.csv", True],
            ["amm_bert_for_layerwise_benchmark", "autotvm",
             "amm_bert_for_layerwise_benchmark.json", "amm_bert_for_layerwise_benchmark.csv", True],
        ])

    return ls
