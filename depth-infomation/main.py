import cv2 
import numpy as np
import os
from techniques.methods import calculation_matching


def run():
    import argparse
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    args.left_image_path = "./data/Tsukuba/left.png"
    args.right_image_path = "./data/Tsukuba/right.png"
    args.disparity_range = 16
    args.metrics = "l2"
    args.scale = 10
    args.save_result = True
    args.results_path = "./data/results"
    args.method = "window-base" # pixel-wise
    args.kernel_size = 3
    args.use_padding = True
    calculation_matching(args)

if __name__ == "__main__":
    run() 