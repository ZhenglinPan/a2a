"""Anomalib Torch Inferencer Script.

This script performs torch inference by reading model config files and weights
from command line, and show the visualization results.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
print(sys.path)
import torch
from torchvision.datasets import CIFAR10, CIFAR100

from anomalib.data.utils import (
    generate_output_image_filename,
    get_image_filenames,
    read_image,
)
from anomalib.deploy import TorchInferencer
from anomalib.post_processing import Visualizer
from util.utils import parse_args, load_config
import numpy
def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to a config file")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--input", type=Path, required=True, help="Path to an image to infer.")
    parser.add_argument("--output", type=Path, required=False, help="Path to save the output image.")
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="auto",
        help="Device to use for inference. Defaults to auto.",
        choices=["auto", "cpu", "gpu", "cuda"],  # cuda and gpu are the same but provided for convenience
    )
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        help="Task type.",
        default="classification",
        choices=["classification", "detection", "segmentation"],
    )
    parser.add_argument(
        "--visualization_mode",
        type=str,
        required=False,
        default="simple",
        help="Visualization mode.",
        choices=["full", "simple"],
    )
    parser.add_argument(
        "--show",
        action="store_true",
        required=False,
        help="Show the visualized predictions on the screen.",
    )

    args = parser.parse_args()

    return args


def infer(config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """Infer predictions.

    Show/save the output if path is to an image. If the path is a directory, go over each image in the directory.
    """
    # Get the command line arguments, and config from the config.yaml file.
    # This config file is also used for training and contains all the relevant
    # information regarding the data, model, train and inference details.
    if config['dataset'] == 'cifar10':
        test_ds = CIFAR10(root='./data', train=False, download=True)
        LEN_TEST = len(test_ds)
        classes = 10
    elif config['dataset'] == 'cifar100':
        test_ds = CIFAR100(root='./data', train=False, download=True)
        LEN_TEST = len(test_ds)
        classes = 100
    
    torch.set_grad_enabled(False)

    # Create the inferencer and visualizer.
    inferencer = TorchInferencer(config=config['model_config'], model_source=config['model_weights'], device='cuda')

    acc = 0
    for image in test_ds.data:
        predictions = inferencer.predict(image=image)
        if predictions.pred_label == 'Normal':
            acc += 1
    print(f"Test_acc: {acc/LEN_TEST}")

    # transformed_data = test_ds.data
    # mean = numpy.mean(transformed_data, axis=(0, 1, 2))
    # std = numpy.std(transformed_data, axis=(0, 1, 2))
    # attacked_data_np = attacked_data_np * std + mean

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    infer(config)
