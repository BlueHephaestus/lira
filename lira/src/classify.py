from Dataset import Dataset

#Main function to put all images through our classification pipeline. Returns the dataset used during the pipeline.
def classify():
    dataset = Dataset()
    dataset.detect_type_ones()
    dataset.predict_grids()
    dataset.get_stats()
    return dataset

classify()

