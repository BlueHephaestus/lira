import numpy as np

image = 12

strides = [float(s) for s in xrange(1, 2+1)]
filters = [float(f) for f in xrange(2, 116)]#Filters in this range is 2x2, 116x116, etc
pooling = True

for s in strides:
  for f in filters:
    c = (image-f)/s + 1
    if pooling:
        if int(c) % 2 == 0:
            if c.is_integer() and c > 1:#Won't return ones that aren't divisible by two, if I want to use pooling. Also won't return negative filters for obvious reasons
                print "%i x %i, Stride: %i, Filter: %i x %i" % (int(c), int(c), int(s), int(f), int(f))
    else:
        if c.is_integer() and c > 1:#Won't return ones that aren't divisible by two, if I want to use pooling. Also won't return negative filters for obvious reasons
            print "%i x %i, Stride: %i, Filter: %i x %i" % (int(c), int(c), int(s), int(f), int(f))

