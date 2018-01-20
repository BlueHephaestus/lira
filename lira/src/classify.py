from Dataset import Dataset

#Main function to put all images through our classification pipeline.
def classify():
    dataset = Dataset()
    dataset.detect_type_ones()
    dataset.predict_grids()

classify()
