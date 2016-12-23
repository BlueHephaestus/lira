import sys
import os
import re
import shutil

#This program takes our unformatted data from the MK. 1 and moves the files into our one big file
old_dir = "../data/raw_training"
new_dir = "../data/samples"

sample_dict = {}

def rm_digits(s):
    return re.sub("\d+", "", s)#Remove digits if they exist(in the case of wikimedia)

def get_ext(s):
    return s[-3:]

def get_name(s):
    #Remove the unnecessary alphabetical numbering and extension
    return s[:-5]

def reformat_audio_layout(raw_dir, new_dir):
    sample_num = 0#For the global number we're at

    for f in os.listdir(os.path.abspath(raw_dir)):
      f = os.path.abspath(raw_dir + "/" + f)
      #F is our subfolder

      if os.path.isdir(f):
          for f_sub in os.listdir(os.path.abspath(f)):
              sample_name = get_name(f_sub)
              file_ext = get_ext(f_sub)

              f_sub = os.path.abspath(f + "/" + f_sub)


              if sample_name in sample_dict:
                  sample_dict[sample_name] += 1
              else:
                  sample_dict[sample_name] = 0
              local_sample_index = sample_dict[sample_name]

              input_fname = f_sub
              output_fname = "%s/%s%i.%s" % (new_dir, sample_name, local_sample_index, file_ext)

              #remove exceptions
              if "Gates" not in sample_name and "octave" not in sample_name and "Quick" not in sample_name and file_ext != "svs":
                  print "Copying #%i: %s%i.%s" % (sample_num, sample_name, local_sample_index, file_ext)
                  shutil.copyfile(input_fname, output_fname)
                  sample_num+=1


reformat_audio_layout(old_dir, new_dir)



"""
TODO FUTURE SELF

Make it add a number to the name of each one
Handle unnecessary xlsx stuff

gl hf
remember catherine is coming over at 4 or w/e
"""
