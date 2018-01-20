import numpy as np

from Images import Images

imgs = Images(restart=True)

for img in imgs:
    print(img.shape)
print("Max Shape: ", imgs.max_shape())

print(imgs[3].shape)
print(np.all(imgs[0] == imgs[-1]))
imgs[-1] = imgs[0]
print(np.all(imgs[0] == imgs[-1]))
print(len(imgs))

imgs = Images(restart=False)
