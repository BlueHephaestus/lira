#Collection of miscellaneous helpers for the LIRA project.
import os, shutil
import cv2
import numpy as np
import sys

def fnames(dir):
    """
    Arguments:
        dir: directory to recursively traverse. Should be a string.

    Returns:
        Yields through a list of filenames in the directory, e.g. 
            ["test.txt", "1.png"]
    """
    paths = []
    for (path, dirs, fnames) in os.walk(dir):
        for fname in fnames:
            yield fname

def file_exists(fname):
    return os.path.exists(fname)

def rects_connected(rect1, rect2):
    #Complicated conditional statements to check if coordinates of two 2d rects are bordering or overlapping (i.e. connected)
    rect1_x1, rect1_y1, rect1_x2, rect1_y2 = rect1
    rect2_x1, rect2_y1, rect2_x2, rect2_y2 = rect2
    
    return  ((rect1_x1 >= rect2_x1 and rect1_x1 <= rect2_x2) and (rect1_y1 >= rect2_y1 and rect1_y1 <= rect2_y2)) or \
            ((rect1_x2 >= rect2_x1 and rect1_x2 <= rect2_x2) and (rect1_y1 >= rect2_y1 and rect1_y1 <= rect2_y2)) or \
            ((rect1_x1 >= rect2_x1 and rect1_x1 <= rect2_x2) and (rect1_y2 >= rect2_y1 and rect1_y2 <= rect2_y2)) or \
            ((rect1_x2 >= rect2_x1 and rect1_x2 <= rect2_x2) and (rect1_y2 >= rect2_y1 and rect1_y2 <= rect2_y2))

def get_rect_clusters(rects):
    sys.setrecursionlimit(16000)
    #Get all rect clusters as a list of clusters, where each cluster is a list of rects in the cluster.
    connected = set()#For reference and avoiding infinite recursion
    clusters = []
    for rect in rects:
        if str(rect) not in connected:#if rect not part of a cluster yet
            clusters.append(get_rect_cluster(rect, rects, connected))#add this rect's cluster to list of clusters
    return clusters
    
def get_rect_cluster(rect, rects, connected):
    #Recursive function used by get_rect_clusters. Gets a cluster of rects from the given rect.

    #Resulting list of rects for our cluster
    cluster = [rect]

    #After this function our rect will be done.
    connected.add(str(rect))#str() in order to hash our list

    for candidate_rect in rects:
        if str(candidate_rect) not in connected:#Rects that are not part of a cluster yet 
            if rects_connected(rect, candidate_rect):#These two rects are connected
                cluster.append(candidate_rect)
                cluster.extend(get_rect_cluster(candidate_rect, rects, connected))#Extend this cluster to include all subclusters of this candidate rect
    return cluster

def windows(img, step_size, win_shape):
    #Yields windows of shape win_shape across our img, in steps step_size. Just uses img for the shape.
    for row_i in range(0, img.shape[0], step_size):
        for col_i in range(0, img.shape[1], step_size):
            if row_i + win_shape[0] <= img.shape[0] and col_i + win_shape[1] <= img.shape[1]:
                yield (row_i, col_i, img[row_i:row_i + win_shape[0], col_i:col_i + win_shape[1]])

def weighted_overlay(img, overlay, alpha):
    #Overlays our overlay onto our img with alpha transparency, and returns the resulting combined img
    return cv2.addWeighted(overlay.astype(np.uint8), alpha, img, 1-alpha, 0)

def is_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

def clear_dir(d, f=None):
    #Removes all files and subdirectories from directory d if they match the given condition f
    for fname in os.listdir(d):
        fpath = os.path.join(d, fname)
        try: 
            if f!=None:
                #ensure the condition is true otherwise skip
                if not f(fpath):
                    continue

            if os.path.isfile(fpath):
                os.unlink(fpath)
            elif os.path.isdir(fpath): 
                shutil.rmtree(fpath)
        except:
            pass

