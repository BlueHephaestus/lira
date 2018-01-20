"""
Quick script to modify our .pkl file for our interactive sessions, if we need to.
"""
import pickle
interactive_session_metadata_dir="dark_interactive_session_metadata.pkl"
f = open(interactive_session_metadata_dir, "r+")
#print pickle.load(f)
#pickle.dump((138,2,0.18,1),f)
#pickle.dump((11,1,0.15,1),f)
#pickle.dump((0,0,0.15,1),f)
pickle.dump((0,0.15),f)

interactive_session_metadata_dir="blake_interactive_session_metadata.pkl"
f = open(interactive_session_metadata_dir, "r+")
pickle.dump((0,0.15),f)
