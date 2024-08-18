import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import cv2
from distance.tools import l1_distance, l2_distance, cosine_similarity



def window_base_calculation(args):
    """
    img_left
    img_right
    kernel_size:
    scale
    disparity_range
    metrics
    use_padding
	 
    """
	
    left_image = cv2.imread(args.left_image_path, 0)
    right_image = cv2.imread(args.right_image_path, 0)

    left_img = left_image.astype(np.float32)
    right_img = right_image.astype(np.float32)

    height, width = left_img.shape
    half_kernel = (args.kernel_size[0] - 1) / 2
    
    costs = np.full((height - half_kernel, width - half_kernel, args.disparity_range), fill_value = 255, dtype = np.float32)
	
    if args.use_padding == True:
        center = half_kernel // 2

        padding_left_img = np.full((height + half_kernel, width + half_kernel, args.disparity_range), fill_value = 255, dtype = np.float32)
        padding_left_img[center: center + height, 
                         center: center + width] = left_img
        
        padding_right_img = np.full((height + half_kernel, width + half_kernel, args.disparity_range), fill_value = 255, dtype = np.float32)
        padding_right_img[center: center + height, 
                         center: center + width] = right_img
        
        costs = np.full((height + half_kernel, width + half_kernel, args.disparity_range), fill_value = 255, dtype = np.float32)

    
    for j in range(args.disparity_range):
        if args.kernel_size == False:
            
            left_d = left_img[half_kernel: height - half_kernel, half_kernel + j: width - half_kernel]
            right_d = right_img[half_kernel: height - half_kernel, half_kernel: width - j - half_kernel]

            v = sliding_window_view(right_d)
            

