import os
import numpy as np
from PIL import Image, ImageChops
import cv2 as cv

wide = Image.open('02_fundus_crop_mask.png').convert('L')
wide = wide.resize((1536,1024))
wide = np.array(wide, dtype=np.uint8)

n_wide = np.zeros((wide.shape[0], wide.shape[1], 3), dtype=np.uint8)
n_wide[:,:,0] = wide
n_wide[:,:,1] = wide

fundus = Image.open('02_fundus_mask.png').convert('L')

crop_fundus = fundus.crop((100, 0, 1300, 1024))

# fundus.crop((1000, 1536, 0, 0))
crop_fundus = crop_fundus.resize((1536, 1024))
shift = (500, 0)
crop_fundus = crop_fundus.transform(crop_fundus.size, Image.AFFINE, (1, 0, 0, 0, 1, 100))
crop_fundus = crop_fundus.transform(crop_fundus.size, Image.AFFINE, (1, 0, 30, 0, 1, 0))


crop_fundus = np.array(crop_fundus, dtype=np.uint8)

# Image.fromarray(fundus).save('r_fundus.png')
n_fundus = np.zeros((crop_fundus.shape[0], crop_fundus.shape[1], 3), dtype=np.uint8)
n_fundus[:,:,1] = crop_fundus
n_fundus[:,:,2] = crop_fundus

img = cv.addWeighted(n_wide,0.5,n_fundus,0.5,0)

Image.fromarray(img).save('a.png')

Image.fromarray(crop_fundus).save('03_fundus.png')
Image.fromarray(wide).save('03_wide.png')