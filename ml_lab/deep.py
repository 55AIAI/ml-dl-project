"""Small TensorFlow data helper; TensorFlow stays an optional dependency."""
def as_dataset(X, y, batch_size, training=False):
    import tensorflow as tf
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        dataset = dataset.shuffle(len(y), seed=42, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return dataset.batch(batch_size).with_options(options)
