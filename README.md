# Image-Stitching
## Introduction

This is a small “panorama maker” for **two overlapping photos**. You give it a left image and a right image, and it tries to line them up into one wider view. Under the hood it finds matching points, figures out how one photo should be stretched/shifted to sit on top of the other, and then blends the overlap so it doesn’t look like a harsh cut.

It’s set up to produce three stitched outputs:
- **Global** (one homography for the whole image)
- **APAP** (a local warp that adapts across the canvas)
- **KR** (a “rew” local pipeline in this repo; based on the same inlier matches)

---

## Data used and its preprocessing

### Data used
The project is image-based. Sample inputs live in:
- `TestImage/` (example pairs: `bridge/1.jpg` + `bridge/2.jpg`, `park/1.JPG` + `park/2.JPG`, etc.)

In most folders:
- `1.jpg` / `2.jpg` (or `1.JPG` / `2.JPG`) are the two images to stitch  
Some folders also include:
- `w1.jpg` / `w2.jpg` (alternate versions)
- `apap_res.jpg`, `our_res.jpg` (reference outputs)

### Preprocessing 
When you pass two paths into the pipeline (`Load_data.load_data`):
- it reads both images using `cv2.imread(...)`
- it immediately converts them to **grayscale** (`cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)`)
- it keeps **both** versions around (color + gray), because matching works on gray but stitching writes color output

After that:
- feature points are detected with **SIFT** by default
- if SIFT can’t produce enough usable matches, the code falls back to **ORB**
- matches are filtered using a **ratio test** (default `0.7` for SIFT, `0.75` for ORB)
- a homography is estimated with **RANSAC** (`HM_ransac`) using:
  - `max_iteration = 500`
  - `ransac_threshold = 30.0` pixels

---

## Features and output

### Final Outcomes
For a single run, the intent is to save:
- `_global.jpg`
- `_apap.jpg`
- `_kr.jpg`

### Each Ouput
- **Global stitch (`mosaic_global.py`)**
  - uses one homography `H`
  - builds a canvas big enough to hold both images
  - warps the second image onto that canvas with interpolation (`scipy.ndimage.map_coordinates`)
  - blends the two layers

- **APAP (`apap.py`)**
  - computes location-dependent homographies (weighted by distance to matched points)
  - warps image 2 onto a wider canvas, changing the warp smoothly across the image
  - blends with image 1

- **KR (`comp_KR.py` + `mosaic_local_ori.py`)**
  - reuses the RANSAC inliers and a homography estimate
  - applies a per-cell warp idea in `mosaic_local_ori.py`
  - blends the result

### Note on blending 
The repository *tries* to use a seam-based blend (`seam_blend`), but the seam finder (`find_seam`) is not actually implemented, and `seam_blend` isn’t wired correctly across files.  
For local runs, the simplest working blend is the already-implemented **`uniform_blend`** (it averages the overlap and keeps non-overlap areas as-is).

---

## Prerequisites

- Python 3.8+
- `opencv-python`
- `numpy`
- `scipy`

Install locally:
```bash
pip install opencv-python numpy scipy
```
---
## How to run it 

### 1) Pick an image pair  
For example:
- `Image-Stitching-main/TestImage/bridge/1.jpg`
- `Image-Stitching-main/TestImage/bridge/2.jpg`


### 2)  “Blend fix” (one-time)  
Because the seam blending path is incomplete, switch the mosaics to use `uniform_blend`.

#### A) `mosaic_global.py`
- change: `from utils import seam_blend`
- to:  `from utils import uniform_blend`
- Last line: `return seam_blend(out1, out2)`
- to: `return uniform_blend(out1, out2)`

#### B) `mosaic_local_ori.py`
- remove the `seam_blend` import
- change:  `return seam_blend(im1, warp2)`
- to: `return uniform_blend(im1, warp2)`

*(Note)* If you want APAP to blend the same way, do the same swap inside `apap.py`.

---

### 3) Set the paths in `main.py`  
Open `main.py` and update:
```python
img1_path = "Image-Stitching-main/TestImage/bridge/1.jpg"
img2_path = "Image-Stitching-main/TestImage/bridge/2.jpg"
```
### 4) Run  
From the folder that contains `Image-Stitching-main/`:

```bash
python Image-Stitching-main/main.py
```
#### You should see stitched outputs saved in the same place you ran the command from, named like:
- new_global.jpg
- new_apap.jpg
- new_kr.jpg
