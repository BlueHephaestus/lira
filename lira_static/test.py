import img_handler
from img_handler import *


img_h = 512
img_w = 1024
sub_h = 16
sub_w = 32
img_divide_factor = 4
i = 0
output = []
for row_i in range(img_divide_factor):
    for col_i in range(img_divide_factor):
        print row_i, col_i
        for j in range(8*16):#number of subs in each big sub
            if j % 8 == 0: print "--"
            print i, get_global_prediction_i(i, 512, 1024, 16, 32, 4, row_i, col_i)
            i+=1


"""
sub_img_h = (img_h//img_divide_factor)//sub_h = 8
sub_img_w = (img_w//img_divide_factor)//sub_w = 8

total_elements_already_passed = sub_img_h * sub_img_w * (sub_img_row_i * img_divide_factor + sub_img_col_i) 
local_i = i - (total_elements_already_passed) 
local_row_i = local_i // sub_img_w 
local_col_i = local_i % sub_img_h
global_row_i = local_row_i + (sub_img_h * sub_img_row_i)
global_col_i = local_col_i + (sub_img_w * sub_img_col_i)
global_i = global_row_i * (img_w//sub_w) + global_col_i
return global_i
"""
