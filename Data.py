import numpy as np
import pandas as pd
import os


path = "/home/rohan/Course Projects/bioMed/Raw Data"

def load_folders():
    folders = os.listdir(path)
    return folders


def consolidate():
    folders = load_folders()
    for i in folders:
        files = os.listdir(path + "/" + i)



def main():
    consolidate()


if __name__ == '__main__':
    main()
