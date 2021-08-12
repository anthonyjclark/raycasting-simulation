# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from subprocess import run

compared_models = [
    "resnet18",
]

for model in compared_models:

    run(
        [
            "python",
            "TrainCmdClassification.py",
            model,
            "corrected-wander-full",
            "--pretrained",
        ]
    )
