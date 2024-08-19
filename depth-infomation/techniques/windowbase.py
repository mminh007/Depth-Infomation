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

    half_kernel = int((args.kernel_size - 1) / 2)
    window_side = (args.kernel_size, args.kernel_size)
    
    costs = np.full((height, width, args.disparity_range), fill_value = 255, dtype = np.float32)
	
        #if args.use_padding == False:
            
    for j in range(args.disparity_range):
        left_d = left_img[:, j: width]
        right_d = right_img[:, 0: width - j]

        if args.use_padding == True:
            center = args.kernel_size // 2

            h,w = left_d.shape

            padding_left_img = np.full((h + half_kernel*2, w + half_kernel*2), fill_value = 0, dtype = np.float32) 
            padding_right_img = np.full((h + half_kernel*2, w + half_kernel*2), fill_value = 0, dtype = np.float32)

            padding_left_img[center: center + h, 
                            center: center + w] = left_d
                
            padding_right_img[center: center + h, 
                              center: center + w] = right_d
                
            s_left = sliding_window_view(padding_left_img, window_side)
            s_right = sliding_window_view(padding_right_img, window_side)
                
            left_cumsum = np.array([np.cumsum(i)[-1] for l in s_left for i in l])
            right_cumsum = np.array([np.cumsum(i)[-1] for l in s_right for i in l])

            left_rs = left_cumsum.reshape(h, w)
            right_rs = right_cumsum.reshape(h, w)

            if args.metrics == "l1":
                costs[:, j: width, j] = l1_distance(left_rs, right_rs)
                    
            if args.metrics == "l2":
                costs[:, j: width, j] = l2_distance(left_rs, right_rs)

        else:
            h,w = left_d.shape[0] - half_kernel*2, left_d.shape[1] - half_kernel*2 
                    
            # sliding window base by numpy
            s_left = sliding_window_view(left_d, window_side)
            s_right = sliding_window_view(right_d, window_side)
                    
            # accumulate sum
            left_cumsum = np.array([np.cumsum(i)[-1] for l in s_left for i in l])
            right_cumsum = np.array([np.cumsum(i)[-1] for l in s_right for i in l])

            left_rs = left_cumsum.reshape(h, w)
            right_rs = right_cumsum.reshape(h, w)

            if args.metrics == "l1":
                costs[half_kernel: height - half_kernel, half_kernel + j: width - half_kernel, j] = l1_distance(left_rs, right_rs)
                        
            if args.metrics == "l2":
                costs[half_kernel: height - half_kernel, half_kernel + j: width - half_kernel, j] = l2_distance(left_rs, right_rs)
            

        min_cost_indices = np.argmin(costs, axis = 2)
        depth = min_cost_indices * args.scale
        depth = depth.astype(np.uint8)

    if args.save_result == True:
        print("Saving result...")
        cv2.imwrite(f"{args.results_path}/{args.method}/vectorization_{args.metrics}.png", depth)
        cv2.imwrite(f"{args.results_path}/{args.method}/vectorization_{args.metrics}_color.png", cv2.applyColorMap(depth, cv2.COLORMAP_JET))
        print("Done!")
    
    return depth