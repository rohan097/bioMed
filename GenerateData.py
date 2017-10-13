import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
import json
import argparse
import multiprocessing
import warnings
from matplotlib.ticker import NullLocator
from matplotlib.pyplot import savefig

parser = argparse.ArgumentParser(
    description="Enter the number of dogs you wish to generate the spectrogram for."
)
parser.add_argument('num_of_dogs', type=int, default=0, nargs='?')
args = parser.parse_args()


def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)


config = load_config()
path = config['path']
fs = config['fs']
NFFT = config['NFFT']
noverlap = config['noverlap']
electrode = config['electrode']
output_path = config['output_path']


def load_data():
    folders = sorted(os.listdir(path))
    if args.num_of_dogs > len(folders):
        raise Exception("There aren't that many folders.")
    jobs = []
    k = 0
    l = 0

    print("Generating Spectrogram for folder: " + folders[args.num_of_dogs - 1])

    for i in range(0, len(os.listdir(path + folders[args.num_of_dogs - 1]))):

        data = scipy.io.loadmat(path + folders[args.num_of_dogs - 1] + '/' +
                                sorted(os.listdir(path + folders[args.num_of_dogs - 1]))[1:][i])
        segment = sorted(data.keys(), reverse=True)[0]
        data = data[segment][0][0][0]
        if 'interictal' in segment:
            p = multiprocessing.Process(target=generate_spectrograms,
                                        args=(data, folders[args.num_of_dogs - 1], 'interictal_', k))
            k += 1
        elif 'preictal' in segment:
            p = multiprocessing.Process(target=generate_spectrograms,
                                        args=(data, folders[args.num_of_dogs - 1], 'preictal_', l))
            l += 1
        jobs.append(p)
        p.start()


# generate spectrograms. label 1 => interictal,2 = > preictal
def generate_spectrograms(data, folder, segment, counter):
    data = np.fft.fft(data[electrode - 1])
    plt.specgram(data, NFFT=NFFT, Fs=fs, noverlap=noverlap, cmap=plt.cm.jet)
    gca = plt.gca
    fig = plt.gcf()
    fig.set_size_inches(3, 2)
    gca().set_axis_off()
    gca().xaxis.set_major_locator(NullLocator())
    gca().yaxis.set_major_locator(NullLocator())
    savefig(output_path + folder + '_Spectrograms/' + segment + str(counter) + '.png',
            bbox_inches='tight', pad_inches=0)


def main():
    if args.num_of_dogs == 0:
        warnings.warn("No argument has been provided. All folders will be processed.")
    load_data()


if __name__ == '__main__':
    main()
