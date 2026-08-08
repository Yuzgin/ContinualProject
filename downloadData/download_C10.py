# Download the CIFAR-10 dataset so it can be used for training, from huggingface because original slow

import os
from datasets import load_dataset

data_folder = "../data"
dataset_path = "../data/cifar10"

# Create the data folder if it does not exist
if not os.path.exists(data_folder):
    os.mkdir(data_folder)

# Check if the dataset has already been downloaded
if os.path.exists(dataset_path):
    print("Dataset downloaded.")
else:
    print("Downloading CIFAR-10")
    print("From https://huggingface.co/datasets/uoft-cs/cifar10")

    # Download the dataset from huggingface
    ds = load_dataset("uoft-cs/cifar10")

    os.mkdir(dataset_path)

    # Save the train images into a folder with a csv of labels
    train_folder = dataset_path + "/train"
    os.mkdir(train_folder)

    train_csv = open(dataset_path + "/train_labels.csv", "w")
    train_csv.write("image,label\n")

    for i in range(len(ds["train"])):
        item = ds["train"][i]
        image_name = str(i) + ".png"
        item["img"].save(train_folder + "/" + image_name)
        train_csv.write(image_name + "," + str(item["label"]) + "\n")

    train_csv.close()

    # Save the test images into a folder with a csv of labels
    
    test_folder = dataset_path + "/test"
    os.mkdir(test_folder)

    test_csv = open(dataset_path + "/test_labels.csv", "w")
    test_csv.write("image,label\n")

    for i in range(len(ds["test"])):
        item = ds["test"][i]
        image_name = str(i) + ".png"
        item["img"].save(test_folder + "/" + image_name)
        test_csv.write(image_name + "," + str(item["label"]) + "\n")

    test_csv.close()

    print("Download finished")
