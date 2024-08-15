import cv2 
import numpy as np
import os
from techniques.distance import l1_distance, l2_distance

def dispasity_calculation(args):
    
    """
    disparity: int 
    distance: "l1", "l2"
    left_image_path, right_image_path, disparity_range, distance, scale, save_result = True
    """
    
    left_image = cv2.imread(args.left_image_path, 0)
    right_image = cv2.imread(args.right_image_path, 0)

    left_img = left_image.astype(np.float32)
    right_img = right_image.astype(np.float32)

    height, width = left_img.shape

    costs = np.full((height, width, args.disparity_range), fill_value = 255, dtype = np.float32)

    for j in range(args.disparity_range):
        left_d = left_img[:, j: width]
        right_d = right_img[:, 0: width - j]
        
        if args.distance == "l1":
            costs[:, j: width, j] = l1_distance(left_d, right_d)
        
        if args.distance == "l2":
            costs[:, j: width, j] = l2_distance(left_d, right_d)

    min_cost_indices = np.argmin(costs, axis = 2)
    depth = min_cost_indices * args.scale
    depth = depth.astype(np.uint8)

    if args.save_result == True:
        print("Saving result...")
        cv2.imwrite(f"{args.results_path}/vectorization_{args.distance}.png", depth)
        cv2.imwrite(f"{args.results_path}/vectorization_{args.distance}_color.png", cv2.applyColorMap(depth, cv2.COLORMAP_JET))
    print("Done!")

    return depth


def run():
    import argparse
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    args.left_image_path = "./data/Tsukuba/left.png"
    args.right_image_path = "./data/Tsukuba/right.png"
    args.disparity_range = 16
    args.distance = "l2"
    args.scale = 10
    args.save_result = True
    args.results_path = "./data/results"
    dispasity_calculation(args)


if __name__ == "__main__":
    run()