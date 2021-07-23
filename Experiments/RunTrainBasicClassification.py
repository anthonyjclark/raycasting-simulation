from subprocess import run

compared_models = [
    "resnet18",
    "xresnet18",
    "xresnet18_deep",
    "xresnet18_deeper",
    "xse_resnet18",
    "xresnext18",
    "xse_resnext18",
    "xse_resnext18_deep",
    "xse_resnext18_deeper",
    "resnet50",
    "xresnet50",
    "xresnet50_deep",
    "xresnet50_deeper",
    "xse_resnet50",
    "xresnext50",
    "xse_resnext50",
    "xse_resnext50_deep",
    "xse_resnext50_deeper",
    "squeezenet1_1",
    "densenet121",
    "densenet201",
    "vgg11_bn",
    "vgg19_bn",
    "alexnet",
]

for dataset in ["uniform-full", "wander-full"]:

    for model in compared_models:

        run(
            [
                "python",
                "TrainBasicClassification.py",
                model,
                dataset,
                "--pretrained",
            ]
        )
