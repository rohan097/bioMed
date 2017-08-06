import numpy as np
import pandas as pd
import scipy.io
import os
import matplotlib.pyplot as plt
import json
import argparse
import warnings

parser = argparse.ArgumentParser(
    description="Enter the number of dogs you wish to generate the spectrogram for."
)
parser.add_argument('num_of_dogs', type=int, default=0, nargs='?')
args = parser.parse_args()

if args.num_of_dogs == 0:
    warnings.warn("No argument has been provided. All folders will be processed.")


def loadConfig():

    with open('config.json', 'r') as f:
        config = json.load(f)

    return config

def loadData():

    config = loadConfig()
    folders = sorted(os.listdir(config['path']))
    if (args.num_of_dogs > len(folders)):
        raise Exception("There aren't that many folders.")
    matFiles = {}
    for i in range (0, args.num_of_dogs):
        matFiles[folders[i]] = sorted(os.listdir(config['path']+folders[i]))
    print (matFiles.keys())


def main():

    loadData()

if __name__ == '__main__':
    main()
