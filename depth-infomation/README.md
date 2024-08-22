# Stereo Matching Image Depth Estimation

The `main.py` scripts that performs depth image computation

## Installing dependencies

> git clone https://github.com/mminh007/Depth-Infomation.git
> cd depth-infomation
> pip install -r requirements.txt

**script**

```
python ./main.py \
	--left_image_path ./data/Tsukuba/left.png \
	--right_image_path ./data/Tsukuba/right.png \
	--method pixel-wise \
	--disparity_range 16 \
	--metrics l2 \
	--scale 10 \
	--save_result True \
	--results_path ./data results

```

## hardware

**method**
pixel-wise matching
window-base matching **_*updating*_**
