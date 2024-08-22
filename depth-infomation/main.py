import cv2 
import numpy as np
import os
from techniques.methods import calculation_matching
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_image_path", type=str, default="./data/Tsukuba/left.png")
    parser.add_argument("--right_image_path", type=str, default="./data/Tsukuba/right.png")
    parser.add_argument("--method", type=str, default="pixel-wise")
    parser.add_argument("--disparity_range", type=int, default=16)
    parser.add_argument("--distance", type=str, default="l2")
    parser.add_argument("--scale", type=int, default=10)
    parser.add_argument("--save_result", type=bool, default=True)
    parser.add_argument("--results_path", type=str, default="./data/results")
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--use_padding", action="store_true", default=False)
    
    args = parser.parse_args()
    
    return args


def run():
    # import argparse
    # parser = argparse.ArgumentParser()

    # args = parser.parse_args()

    # args.left_image_path = "./data/Tsukuba/left.png"
    # args.right_image_path = "./data/Tsukuba/right.png"
    # args.disparity_range = 16
    # args.metrics = "l2"
    # args.scale = 10
    # args.save_result = True
    # args.results_path = "./data/results"
    # args.method = "window-base" # pixel-wise
    # args.kernel_size = 3
    # args.use_padding = True
    # calculation_matching(args)
    args = parse_args()
    calculation_matching(args)


if __name__ == "__main__":
    run() 