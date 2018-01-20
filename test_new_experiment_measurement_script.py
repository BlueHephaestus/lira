import pickle
import numpy as np
with open("dark_results.pkl", "r") as f:
    #pre_stats, post_stats = pickle.load(f)
    #print np.ceil(pre_stats[0,...]).astype(np.int64)
    post_stats = pickle.load(f)
    #print np.ceil(np.sum(pre_stats, axis=1)).astype(np.int64)
    print""
    #print np.ceil(np.sum(post_stats, axis=1)).astype(np.int64)
    print np.ceil(post_stats[0,...]).astype(np.int64)
