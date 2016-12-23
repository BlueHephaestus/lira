"""
For getting the size of all bacterial clumps in the images returned from
bmp_loader's load_greyscales function. 

From our greyscale matrix, we sum all the pixels over our threshold,
in this case 18, and then print the size for each image.

-Blake Edwards / Dark Element
"""
import numpy as np
import bmp_loader

#Get greyscales with our data dir
greyscales = bmp_loader.load_greyscales("../data")

#Get all the cells that are > 18 brightness, arbitrary threshold
for greyscale_i, greyscale in enumerate(greyscales):
  total_size = 0
  for row in greyscale:
    for pixel in row:
      if pixel > 18:
        total_size += 1
  print "Total size of Bacterial Clumps in Image #%i is %i" % (greyscale_i, total_size)

