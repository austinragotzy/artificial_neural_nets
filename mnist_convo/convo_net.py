import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./", one_hot=True)


def main():
    # create model
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    cov1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=5, padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(cov1, 2, 2)
    cov2 = tf.layers.conv2d(pool1, 64, kernel_size=5, padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(cov2, 2, 2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(pool2_flat, units=1024, activation=tf.nn.relu)
    logits = tf.layers.dense(dense, units=10)
    loss = tf.losses.mean_squared_error(y_, logits)
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.5)
    update = opt.minimize(loss)

    # use model
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # train model
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(150)
        batch_xs = batch_xs.reshape(150, 28, 28, 1)
        l, _ = sess.run([loss, update], feed_dict={x: batch_xs, y_: batch_ys})
        if i % 20 == 0:
            print('loss: ' + str(l))

    # evaluate model
    correct_predict = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    x_test = mnist.test.images.reshape(10000, 28, 28, 1)
    x_test = x_test[0:1000]
    y_test = mnist.test.labels
    y_test = y_test[0:1000]
    print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))


if __name__ == '__main__':
    main()
