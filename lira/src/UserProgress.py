import os, json

from base import *

class UserProgress(object):
    #For use with our json for keeping track of user progress
    def __init__(self, uid):
        self.uid = uid

        #Create json for this uid
        self.archive_dir = "../data/user_progress/" #where we will create and store the .json file for each user's progress

        #Default starting json/dict
        self.initial_progress = {
          "type_ones_started_editing": False,#Equivalent to finished_generating
          "type_ones_finished_editing": False,
          "type_ones_image": 0,
          "prediction_grids_started_editing": False,#Equivalent to finished_generating
          "prediction_grids_finished_editing": False,
          "prediction_grids_image": 0,
          "prediction_grids_transparency_factor": 0.33,#Default value
          "prediction_grids_resize_factor": 0.1,#Default value
        }

        #Get archive fpath
        self.archive_fpath = os.path.join(self.archive_dir, "{}.json".format(uid))

    def ensure_progress_json(self):
        #Create our json with our initial_progress attribute if it doesnt already exist
        if not file_exists(self.archive_fpath):
            with open(self.archive_fpath, 'w') as f:
                json.dump(self.initial_progress, f)

    def __getitem__(self, key):
        #Ensure our json exists, then load this as our progress and use this.
        self.ensure_progress_json()

        with open(self.archive_fpath, 'r') as f:
            return json.load(f)[key]

    def __setitem__(self, key, data):
        #Ensure our json exists, then load this as our progress and use this.
        self.ensure_progress_json()

        with open(self.archive_fpath, 'r') as f:
            progress = json.load(f)

        with open(self.archive_fpath, 'w') as f:
            progress[key] = data
            json.dump(progress, f)

    def editing_started(self):
        #Ensure our json exists, then load this as our progress and use this.
        self.ensure_progress_json()

        #If the current json is not the default starting json.
        with open(self.archive_fpath, 'r') as f:
            return json.load(f) != self.initial_progress

    def restart(self):
        #Set our json back to the default starting json.
        with open(self.archive_fpath, 'w') as f:
            json.dump(self.initial_progress, f)


