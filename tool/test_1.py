from PIL import Image
import numpy as np

im = Image.open('example.png')
a = np.asarray(im)
print(a.shape)