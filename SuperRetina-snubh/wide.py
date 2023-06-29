import os
import numpy as np
from PIL import Image

img = Image.open('wide_resize_remove.png').convert('L')
img.resize((1536, 1024))

# img = np.array(img)

# img[0:500, :] = 0

img.save('wide_resize.png')