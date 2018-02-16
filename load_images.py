import os, glob
from scipy import ndimage


def load():

    dataset = []
    load_class(dataset, "Landscape")
    load_class(dataset, "Cityscape")

    return dataset


def load_class(dataset, class_image):
    script_dir = os.path.dirname(os.path.abspath('__file__'))
    rel_path = class_image
    abs_file_path = os.path.join(script_dir, rel_path)
    abs_file_path = abs_file_path+"/*.jpg"
    files=glob.glob(abs_file_path)

    for file in files:
        image = ndimage.imread(file)

        if class_image == "Landscape":
            dataset.append([image, 1])
        else:
            dataset.append([image, -1])
