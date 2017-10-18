import numpy as np
import pandas as pd
import os

def load_folders():
    folders = os.listdir("/home/rohan/Course Projects/bioMed/Raw Data")
    print (folders)


def main():
    load_folders()


if __name__ == '__main__':
    main()
