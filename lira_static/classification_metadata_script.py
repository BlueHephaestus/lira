import pickle

classifications = ["Healthy Tissue", "Type I - Caseum", "Type II", "Empty Slide", "Type III", "Type I - Rim", "Unknown"]

#BGR Colors to give to each classification index
#         Pink,          Red,         Green,       Light Grey,      Yellow,        Blue         Purple
colors = [(255, 0, 255), (0, 0, 255), (0, 255, 0), (200, 200, 200), (0, 255, 255), (255, 0, 0), (244,66,143)]

#Write our classification - color matchup metadata for future use
f = open("classification_metadata.pkl", "w")

pickle.dump(([classifications, colors]), f)

f = open("classification_metadata.pkl", 'r')
classification_metadata = pickle.load(f)
print classification_metadata
