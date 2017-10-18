import tensorflow as tf
import numpy as np
import glob
from random import shuffle
from PIL import Image
print ("Imported all necessary modules.")

preictal_path = "/home/rohan/Course Projects/bioMed/Data/Preictal/*.png"
interictal_path = "/home/rohan/Course Projects/bioMed/Data/Interictal/*.png"
tfrecords_filename_train = "train.tfrecords"
tfrecords_filename_test = "test.tfrecords"
tfrecords_filename_valid = "valid.tfrecords"
shuffle_ = 1


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Zero for Preictal, and One for Interictal
def load_data():
    addrs = glob.glob(preictal_path) + glob.glob(interictal_path)
    labels = [1 if 'interictal' in addr else 0 for addr in addrs]
    return addrs, labels


def shuffle_data(addrs, labels):
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)
    return addrs, labels


def generate():

    addrs, labels = load_data()
    if shuffle:
        addrs, labels = shuffle_data(addrs, labels)

    train_addrs = addrs[0: int(0.6 * len(addrs))]
    train_labels = labels[0: int(0.6 * len(labels))]
    val_addrs = addrs[int(0.6 * len(addrs)): int(0.8 * len(addrs))]
    val_labels = labels[int(0.6 * len(addrs)): int(0.8 * len(addrs))]
    test_addrs = addrs[int(0.8 * len(addrs)):]
    test_labels = labels[int(0.8 * len(labels)):]

    writer = tf.python_io.TFRecordWriter(tfrecords_filename_train)

    for i in range(len(train_addrs)):
        img = np.array(Image.open(train_addrs[i]))
        label = train_labels[i]
        height = img.shape[0]
        width = img.shape[1]
        depth = img.shape[2]
        img = img.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'depth': _int64_feature(depth),
            'image_raw': _bytes_feature(img),
            'annotation': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    writer.close()
    print ("Finished generating binary file for training data.")

    writer = tf.python_io.TFRecordWriter(tfrecords_filename_test)

    for i in range(len(test_addrs)):
        img = np.array(Image.open(test_addrs[i]))
        label = test_labels[i]
        height = img.shape[0]
        width = img.shape[1]
        depth = img.shape[2]
        img = img.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'depth': _int64_feature(depth),
            'image_raw': _bytes_feature(img),
            'annotation': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    writer.close()
    print("Finished generating binary file for testing data.")

    writer = tf.python_io.TFRecordWriter(tfrecords_filename_valid)

    for i in range(len(val_addrs)):
        img = np.array(Image.open(val_addrs[i]))
        label = val_labels[i]
        height = img.shape[0]
        width = img.shape[1]
        depth = img.shape[2]
        img = img.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'depth': _int64_feature(depth),
            'image_raw': _bytes_feature(img),
            'annotation': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    writer.close()
    print("Finished generating binary file for validation data.")


def main():
    generate()


if __name__ == '__main__':
    main()