import os
import shutil


path = "/home/rohan/Course Projects/bioMed/Raw Data"
output = "/home/rohan/Course Projects/bioMed/Data/"


def load_folders():
    folders = os.listdir(path)
    return folders


def consolidate():
    folders = load_folders()
    k = 0
    l = 0
    for i in folders:
        files = os.listdir(path + "/" + i)
        for j in files:
            if 'interictal' in j:
                shutil.copy(path + "/" + i + "/" + j, output + "Interictal/interictal_%d.png" % k)
                k += 1
            elif 'preictal' in j:
                shutil.copy(path + "/" + i + "/" + j, output + "Preictal/preictal_%d.png" % l)
                l += 1


def main():
    consolidate()


if __name__ == '__main__':
    main()
