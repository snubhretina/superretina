import os
import numpy as np
from PIL import Image

f1 = '/mnt/data3/sjgo/super-retina/data/SNUBH-DB/00_OT_8898_R00099633_I00381657_1.2.410.200055.999.999.1969579916.5864.1536510508.75865.png'
f2 = '/mnt/data3/sjgo/super-retina/data/SNUBH-DB/01_OT_0001_R00099634_I00381660_1.2.410.200055.999.999.1969579916.5864.1536510508.75868.png'

img1 =Image.open(f1)
img1 = img1.resize((1536, 1024))
img1.save('a.png')
# img1 = np.array(img1)
# img2 = np.array(Image.open(f2))

# img_fov = np.ones((img1.shape[0], img1.shape[1]), dtype=np.uint8) * 255

# img_fov[np.where(np.bitwise_and(np.bitwise_and(img1[:,:,0] < 10, img1[:,:,1] < 10), img1[:,:,2] <10))] = 0

# Image.fromarray(img_fov).save('img_fov1.png')