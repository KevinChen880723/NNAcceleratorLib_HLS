import pickle
import torch
import numpy as np
import os

def main():
    with open("./deepLearning/MNIST/Model/BestModel.pickle", 'rb') as f:
        weights = pickle.load(f)
    weightNames = list(weights.keys())
    if not os.path.isdir("./weight"):
        os.mkdir("./weight")
    for weightName in weightNames:
        print("\n======== {} ========\n".format(weightName))
        print("Shape of the weight: {}".format(weights[weightName].shape))
        print(weights[weightName])
        weight = weights[weightName]
        weightLen = 1
        for w in weight.shape:
            weightLen *= w
        weight = weight.reshape(np.array(weightLen))
        with open("./weight/{}.h".format(weightName), "w") as file:
            for element in weight:
                file.write("{},\n".format(element))
        print("\n================\n")

if __name__ == "__main__":
    main()