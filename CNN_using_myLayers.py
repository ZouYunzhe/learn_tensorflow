import os
import os.path
import tensorflow as tf
import myLayers
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# %% TO get better tensorboard figures!
IMG_W = 28
IMG_H = 28
N_CLASSES = 10
BATCH_SIZE = 100
learning_rate = 1e-4
MAX_STEP = 1000
IS_PRETRAIN = False


def myCNN(x, n_classes, keep_prob, is_pretrain=False):
    with tf.name_scope('myCNN'):
        x = myLayers.conv('conv1', x, 32, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        with tf.name_scope('pool1'):
            x = myLayers.pool('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], pad='VALID', is_max_pool=True)
        x = myLayers.conv('conv2', x, 64, kernel_size=[5, 5], stride=[1, 1, 1, 1], is_pretrain=is_pretrain)
        with tf.name_scope('pool2'):
            x = myLayers.pool('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], pad='VALID', is_max_pool=True)
        x = myLayers.FC_layer('fc1', x, out_nodes=1024, activation_function=tf.nn.relu)
        # with tf.name_scope('batch_norm2'):
        # x = myLayers.batch_norm(x)
        x = myLayers.dropout(x, keep_prob)
        x = myLayers.FC_layer('fc2', x, out_nodes=n_classes, activation_function=tf.nn.softmax)
        return x


# %%   Training
def train():
    # pre_trained_weights = './/vgg16_pretrain//vgg16.npy'
    # data_dir = './/data//cifar-10-batches-bin//'
    train_log_dir = './/logs//train//'
    val_log_dir = './/logs//val//'

    with tf.name_scope('input'):
        # tra_image_batch, tra_label_batch = mnist.train.next_batch(BATCH_SIZE)
        # val_image_batch = mnist.test.images[:1000]
        # val_label_batch = mnist.test.labels[:1000]
        # tra_image_batch, tra_label_batch = input_data.read_cifar10(data_dir=data_dir,
        #                                                            is_train=True,
        #                                                            batch_size=BATCH_SIZE,
        #                                                            shuffle=True)
        # val_image_batch, val_label_batch = input_data.read_cifar10(data_dir=data_dir,
        #                                                            is_train=False,
        #                                                            batch_size=BATCH_SIZE,
        #                                                            shuffle=False)

        x = tf.placeholder(tf.float32, shape=[None, IMG_W*IMG_H]) / 255
        x_image = tf.reshape(x, [-1, IMG_W, IMG_H, 1])
        y_ = tf.placeholder(tf.float32, shape=[None, N_CLASSES])
        keep_prob = tf.placeholder(tf.float32)

    logits = myCNN(x_image, N_CLASSES, keep_prob, IS_PRETRAIN)
    loss = myLayers.loss(logits, y_)
    accuracy = myLayers.accuracy(logits, y_)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = myLayers.optimize(loss, learning_rate, my_global_step)

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # load the parameter file, assign the parameters, skip the specific layers
    # tools.load_with_skip(pre_trained_weights, sess, ['fc6', 'fc7', 'fc8'])

    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

    try:
        for step in np.arange(MAX_STEP):

            # if coord.should_stop():
            #     break
            tra_images, tra_labels = mnist.train.next_batch(BATCH_SIZE)
            # tra_images, tra_labels = sess.run([tra_image_batch, tra_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x: tra_images, y_: tra_labels, keep_prob: 0.5})
            if step % 50 == 0 or (step + 1) == MAX_STEP:
                print('(training) Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op, feed_dict={x: tra_images, y_: tra_labels, keep_prob: 0.5})
                tra_summary_writer.add_summary(summary_str, step)

                # validation
                val_images = mnist.test.images[:1000]
                val_labels = mnist.test.labels[:1000]
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={x: val_images, y_: val_labels, keep_prob: 1})
                print('** (Validation) Step %d, val loss = %.4f, val accuracy = %.4f%%  **' % (step, val_loss, val_acc))
                summary_str = sess.run(summary_op, feed_dict={x: val_images, y_: val_labels, keep_prob: 1})
                val_summary_writer.add_summary(summary_str, step)
                # if step % 200 == 0 or (step + 1) == MAX_STEP:
                #     val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                #     val_loss, val_acc = sess.run([loss, accuracy],
                #                                  feed_dict={x: val_images, y_: val_labels})
                #     print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc))
                #
                #     summary_str = sess.run(summary_op, feed_dict={x: val_images, y_: val_labels})
                #     val_summary_writer.add_summary(summary_str, step)

                if step % 500 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # coord.request_stop()
        pass

    # coord.join(threads)
    sess.close()


if __name__ == '__main__':
    train()
