import os

dir = "type_III"

for i, old_fname in enumerate(os.listdir(dir)):
    old_fname = os.path.join(dir, old_fname)
    new_fname = "%s%i.jpg" % (os.path.join(dir, dir), i)
    os.rename(old_fname, new_fname)
