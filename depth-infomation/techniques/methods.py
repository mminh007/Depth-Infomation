
from techniques.pixelwise import pixel_wise_calculation
from techniques.windowbase import window_base_calculation



def calculation_matching(args):
    
    if args.method == "pixel-wise":
        return pixel_wise_calculation(args)

    if args.method == "window-base":
        return window_base_calculation(args)

