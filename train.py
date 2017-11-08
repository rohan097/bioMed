import tensorflow as tf
import os

print("All necessary modules have been imported.")

train_reader = "/home/rohan/Course Projects/bioMed/train.tfrecords"
valid_reader = "/home/rohan/Course Projects/bioMed/valid.tfrecords"
test_reader = "/home/rohan/Course Projects/bioMed/test.tfrecords"
IMAGE_HEIGHT = 161
IMAGE_WIDTH = 239
IMAGE_DEPTH = 4

batch_size = 8

patch_size_1 = 5
patch_size_2 = 7
patch_size_3 = 9
kernel_depth = 4

num_hidden = 128
num_labels = 2
num_of_steps = 10001


def writer_path():
    runs = len(os.listdir("/home/rohan/Course Projects/bioMed/Tensorboard/"))
    log_dir = "/home/rohan/Course Projects/bioMed/Tensorboard/Run_%d" % runs
    return log_dir


def read_and_decode(filename_queue, training, batch):
    """ A function to read the TfRecord files, decode the data and convert the
        labels to a one-hot encoding."""
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    print("finished reading file")
    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'annotation': tf.FixedLenFeature([], tf.int64)
        }
    )
    print("Features parsed.")
    image = tf.decode_raw(features['image_raw'], tf.uint8)

    image = tf.reshape(image, [161, 239, 4])
    annotation = tf.cast(features['annotation'], tf.int32)

    resized_image = tf.image.resize_image_with_crop_or_pad(
        image=image,
        target_height=IMAGE_HEIGHT,
        target_width=IMAGE_WIDTH)

    if training:
        resized_image, annotation = tf.train.shuffle_batch(
            [resized_image, annotation],
            batch_size=batch,
            capacity=48,
            num_threads=2,
            min_after_dequeue=32)
    else:
        resized_image, annotation = tf.train.shuffle_batch(
            [resized_image, annotation],
            batch_size=batch,
            capacity=646,
            min_after_dequeue=645)
    annotations = tf.one_hot(annotation, 2, on_value=1, off_value=0, axis=1)
    return resized_image, annotations


def main():
    graph = tf.Graph()
    with graph.as_default():

        with tf.name_scope("FileReaderOp"):
            filename_queue_train = tf.train.string_input_producer(
                [train_reader]
            )
            image, annotation = read_and_decode(filename_queue_train, training=True, batch=batch_size)
            filename_queue_valid = tf.train.string_input_producer(
                [valid_reader]
            )
            valid_data, valid_labels = read_and_decode(filename_queue_valid, training=False, batch=645)
            valid_data = tf.cast(valid_data, tf.float32)
            filename_queue_test = tf.train.string_input_producer(
                [test_reader]
            )
            test_data, test_labels = read_and_decode(filename_queue_test, training=False, batch=645)
            test_data = tf.cast(test_data, tf.float32)

        with tf.name_scope("Input_Pipeline"):
            train_data = tf.placeholder(tf.float32,
                                        shape=[batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH],
                                        name="TrainingData")
            train_labels = tf.placeholder(tf.int32,
                                          shape=[batch_size, 2])

        with tf.name_scope("Weights_and_Biases"):
            w1 = tf.Variable(tf.truncated_normal(
                [patch_size_1, patch_size_1, IMAGE_DEPTH, kernel_depth],
                stddev=0.1), name="Weights1")
            b1 = tf.Variable(tf.zeros([kernel_depth]), name="Bias1")
            tf.summary.histogram("Weights1", w1)
            tf.summary.histogram("Bias1", b1)
            w2 = tf.Variable(tf.truncated_normal(
                [patch_size_2, patch_size_2, kernel_depth, kernel_depth],
                stddev=0.1), name="Weights2")
            b2 = tf.Variable(tf.zeros([kernel_depth]), name="Bias2")
            tf.summary.histogram("Weights2", w2)
            tf.summary.histogram("Bias2", b2)
            w3 = tf.Variable(tf.truncated_normal(
                [((IMAGE_HEIGHT // 4) + 1) * ((IMAGE_WIDTH + 1) // 4) * kernel_depth, num_hidden],
                stddev=0.1), name="Weights4")
            b3 = tf.Variable(tf.zeros([num_hidden]), name="Bias3")
            tf.summary.histogram("Weights3", w3)
            tf.summary.histogram("Bias3", b3)
            w4 = tf.Variable(tf.truncated_normal(
                [num_hidden, num_labels],
                stddev=0.1), name="Bias4")
            b4 = tf.Variable(tf.zeros([num_labels]), name="Bias4")
            tf.summary.histogram("Weights4", w4)
            tf.summary.histogram("Bias4", b4)

        with tf.name_scope("Architecture"):
            def model(data):
                conv = tf.nn.conv2d(data, w1, [1, 1, 1, 1], padding='SAME')
                conv = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
                hidden = tf.nn.relu(conv + b1)
                conv = tf.nn.conv2d(hidden, w2, [1, 1, 1, 1], padding='SAME')
                conv = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
                hidden = tf.nn.relu(conv + b2)
                shape = hidden.get_shape().as_list()
                reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
                hidden = tf.nn.relu(tf.matmul(reshape, w3) + b3)
                return tf.matmul(hidden, w4) + b4

            logits = model(train_data)
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=train_labels,
                                                        logits=logits))
            tf.summary.scalar("Loss", loss)
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(0.005,
                                                       global_step=global_step,
                                                       decay_rate=0.9,
                                                       decay_steps=10000
                                                       )
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        with tf.name_scope("Evaluations"):
            train_prediction = tf.nn.softmax(logits)
            valid_prediction = tf.nn.softmax(model(valid_data))
            test_prediction = tf.nn.softmax(model(test_data))

            train_acc = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(train_labels, 1))
            train_acc = tf.reduce_mean(tf.cast(train_acc, tf.float32))

            test_acc = tf.equal(tf.argmax(test_prediction, 1), tf.argmax(test_labels, 1))
            test_acc = tf.reduce_mean(tf.cast(test_acc, tf.float32))

            valid_acc = tf.equal(tf.argmax(valid_prediction, 1), tf.argmax(valid_labels, 1))
            valid_acc = tf.reduce_mean(tf.cast(valid_acc, tf.float32))

            tf.summary.scalar("Training_Accuracy", train_acc)
            tf.summary.scalar("Testing Accuracy", test_acc)
            tf.summary.scalar("Validation Accuracy", valid_acc)

            merged_summary_op = tf.summary.merge_all()

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer(), name="Initialisation_Op")

    with tf.Session(graph=graph) as session:

        writer = tf.summary.FileWriter(writer_path(), session.graph)
        session.run(init_op)
        print("Variables initialised.")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for step in range(num_of_steps):
            img, labels = session.run([image, annotation])
            feed_dict = {train_data: img, train_labels: labels}
            _, l_value = session.run([optimizer, loss], feed_dict=feed_dict)

            if step % 250 == 0:
                summary = session.run(merged_summary_op, feed_dict=feed_dict)
                writer.add_summary(summary, step)

                if step % 500 == 0:
                    print("Minibatch loss at step %d: %f" % (step, l_value))
                    print("Training accuracy at step %d: %f" % (step, train_acc.eval(feed_dict=feed_dict)))
                    print("Validation accuracy at step %d: %f" % (step, valid_acc.eval()))
                    print("Testing accuracy at step %d: %f" % (step, test_acc.eval()))

        writer.close()

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
