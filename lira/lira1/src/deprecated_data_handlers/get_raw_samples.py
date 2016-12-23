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
    if re.compile('\d').search(s):
        #If there are digits, remove them and return that
        s = rm_digits(s)
        return s[:-4]
    else:
        return s[:-5]

def reformat_audio_layout(raw_dir, new_dir):
    sample_num = 0#For the global number we're at

    #Go through and remove any samples that are already there
    for fname in os.listdir(new_dir):
        fpath = os.path.join(new_dir, fname)
        try: 
            if os.path.isfile(fpath):
                os.unlink(fpath)
            else:
                shutil.rmtree(fpath)
                sys.exit()
        except:
            pass

    for f in os.listdir(os.path.abspath(raw_dir)):
      f = os.path.abspath(raw_dir + "/" + f)
      #F is our subfolder

      if "raw_rim" in f:
          #If we have our raw rim files, which are in the form of individual subsections,
          #We copy the entire dir to the samples dir for get_archive.py to handle
          new_raw_rim_dir = "%s/raw_rim/" % new_dir
          shutil.copytree(f, new_raw_rim_dir)

      else:
          #Otherwise do our normal thing
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
                      #Overwrite last letters with whitespace to get a clean update
                      sys.stdout.write("\r\t\t\t\t\t\t\t")

                      #Print progress
                      sys.stdout.write("\rCopying #%i: %s%i.%s" % (sample_num, sample_name, local_sample_index, file_ext))
                      sys.stdout.flush()

                      shutil.copyfile(input_fname, output_fname)
                      sample_num+=1
    print ""


reformat_audio_layout(old_dir, new_dir)



"""
TODO FUTURE SELF

Make it add a number to the name of each one
Handle unnecessary xlsx stuff
"""
