from PIL import Image

img = Image.open('02_fundus_mask.png').resize((768,768))
img.save('000_f.png')

img = Image.open('02_fundus_crop_mask.png').resize((768,768))
img.save('000_w.png')
