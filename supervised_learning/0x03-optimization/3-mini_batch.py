#!/usr/bin/env python3
"""
    module
"""
import tensorflow.compat.v1 as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ train """
    m = X_train.shape[0]
    saver = tf.train.import_meta_graph(load_path + '.meta')
    with tf.Session() as sess:
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]
        for ep in range(epochs):
            t_dict = {x: X_train, y: Y_train}
            v_dict = {x: X_valid, y: Y_valid}
            train_cost = sess.run(loss, feed_dict=t_dict)
            train_accuracy = sess.run(accuracy, feed_dict=t_dict)
            valid_cost = sess.run(loss, feed_dict=v_dict)
            valid_accuracy = sess.run(accuracy, feed_dict=v_dict)
            print(f"After {ep} epochs:")
            print(f"\tTraining Cost: {train_cost}")
            print(f"\tTraining Accuracy: {train_accuracy}")
            print(f"\tValidation Cost: {valid_cost}")
            print(f"\tValidation Accuracy: {valid_accuracy}")
            x_shuff, y_shuff = shuffle_data(X_train, Y_train)
            for it in range(m // batch_size):
                b_size, idx = batch_size, (it * batch_size)
                if (idx + 1) * batch_size > m:
                    b_size = m - idx
                train_dict = {
                    x: x_shuff[idx: idx + b_size],
                    y: y_shuff[idx: idx + b_size]
                }
                if it != 0 and (it % 100) == 0:
                    step_cost = sess.run(loss, feed_dict=train_dict)
                    step_accuracy = sess.run(accuracy, feed_dict=train_dict)
                    print(f"\tStep {it}:")
                    print(f"\t\tCost: {step_cost}")
                    print(f"\t\tAccuracy: {step_accuracy}")
                sess.run(train_op, feed_dict=train_dict)
        saver.save(sess, save_path)
