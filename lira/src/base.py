#Collection of miscellaneous helpers for the LIRA project.
import os
import cv2
import numpy as np

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

class suppress_stderr(object):
    """
    A context manager for doing a "deep suppression" of stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    Credit goes to: https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
        They're a lifesaver.
    """
    def __init__(self):
        # Open a null file
        self.null_fd =  os.open(os.devnull,os.O_RDWR)
        # Save the actual stderr (2) file descriptor.
        self.save_fd = os.dup(2)

    def __enter__(self):
        # Assign the null pointer to stderr.
        os.dup2(self.null_fd,2)

    def __exit__(self, *_):
        # Re-assign the real stderr back to (2)
        os.dup2(self.save_fd,2)
        # Close file descriptor
        os.close(self.null_fd)
        os.close(self.save_fd)
