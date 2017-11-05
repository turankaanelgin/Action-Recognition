import pickle
import os
import tensorflow as tf

CATEGORIES = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
DATASET_DIR = "../data"

def split_data(data):
    features = []
    labels = []

    for ex in data:
        features.append(ex["frames"])
        labels.append(ex["category"])
    return (features, labels)

def labels2indices(labels):
    indices = []
    for label in labels:
        indices.append(CATEGORIES.index(label))
    return indices

def cnn_model_optical_flow(features, labels, mode):
    nrVideos = len(features)
    input_layer = tf.reshape(features, [-1, nrVideos, 8, 11, 2])

    conv1 = tf.layers.conv3d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3, 3],
        padding="same",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling3d(
        inputs=conv1,
        pool_size=[2, 2, 2],
        strides=16)

    conv2 = tf.layers.conv3d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3, 3],
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling3d(
        inputs=conv2,
        pool_size=[2, 2, 2],
        strides=8)
    pool2_flat = tf.reshape(pool2, [-1, nrVideos/4*2*2*64])

    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=150,
        activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.5,
        training=(mode == tf.estimator.ModeKeys.TRAIN))

    logits = tf.layers.dense(
        inputs=dropout,
        units=6)

    onehot_labels = tf.one_hot(indices=labels2indices(labels), depth=6)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        predictions = tf.argmax(input=logits, axis=1)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions)
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

if __name__ == "__main__":
    train = pickle.load(open(os.path.join(DATASET_DIR, "train_dense_flow.pickle"), "rb"))
    print("Training set was loaded...")
    dev = pickle.load(open(os.path.join(DATASET_DIR, "dev_dense_flow.pickle"), "rb"))
    print("Development set was loaded...")
    test = pickle.load(open(os.path.join(DATASET_DIR, "test_dense_flow.pickle"), "rb"))
    print("Test set was loaded...")

    train_feat, train_labels = split_data(train)
    dev_feat, dev_labels = split_data(dev)
    test_feat, test_labels = split_data(test)

    classifier = tf.estimator.Estimator(model_fn=cnn_model_optical_flow, model_dir=DATASET_DIR)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_feat, y=t
    )