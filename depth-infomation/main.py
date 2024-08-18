import cv2 
import numpy as np
import os
from distance.tools import l1_distance, l2_distance
from techniques.pixelwise import pixel_wise_calculation


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
    pixel_wise_calculation(args)


if __name__ == "__main__":
    run() 