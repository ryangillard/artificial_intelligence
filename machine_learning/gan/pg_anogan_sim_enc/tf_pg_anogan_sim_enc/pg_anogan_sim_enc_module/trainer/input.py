import tensorflow as tf

from . import image_utils
from .print_object import print_obj


def decode_example(protos, params):
    """Decodes TFRecord file into tensors.

    Given protobufs, decode into image and label tensors.

    Args:
        protos: protobufs from TFRecord file.
        params: dict, user passed parameters.

    Returns:
        Image and label tensors.
    """
    # Create feature schema map for protos.
    features = {
        "image_raw": tf.FixedLenFeature(shape=[], dtype=tf.string),
        "label": tf.FixedLenFeature(shape=[], dtype=tf.int64)
    }

    # Parse features from tf.Example.
    parsed_features = tf.parse_single_example(
        serialized=protos, features=features
    )
    print_obj("\ndecode_example", "features", features)

    # Convert from a scalar string tensor (whose single string has
    # length height * width * depth) to a uint8 tensor with shape
    # [height * width * depth].
    image = tf.decode_raw(
        input_bytes=parsed_features["image_raw"], out_type=tf.uint8
    )
    print_obj("decode_example", "image", image)

    # Reshape flattened image back into normal dimensions.
    image = tf.reshape(
        tensor=image,
        shape=[params["height"], params["width"], params["depth"]]
    )
    print_obj("decode_example", "image", image)

    # Preprocess image.
    image = image_utils.preprocess_image(image=image, params=params)
    print_obj("decode_example", "image", image)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(x=parsed_features["label"], dtype=tf.int32)
    print_obj("decode_example", "label", label)

    return {"image": image}, label


def set_static_shape(features, labels, batch_size):
    features["image"].set_shape(
        features["image"].get_shape().merge_with(
            tf.TensorShape([batch_size, None, None, None])
        )
    )
    labels.set_shape(
        labels.get_shape().merge_with(tf.TensorShape([batch_size]))
    )

    return features, labels


def read_dataset(filename, mode, batch_size, params):
    """Reads CSV time series data using tf.data, doing necessary preprocessing.

    Given filename, mode, batch size, and other parameters, read CSV dataset
    using Dataset API, apply necessary preprocessing, and return an input
    function to the Estimator API.

    Args:
        filename: str, file pattern that to read into our tf.data dataset.
        mode: The estimator ModeKeys. Can be TRAIN or EVAL.
        batch_size: int, number of examples per batch.
        params: dict, dictionary of user passed parameters.

    Returns:
        An input function.
    """
    def _input_fn(params):
        """Wrapper input function used by Estimator API to get data tensors.

        Returns:
            Batched dataset object of dictionary of feature tensors and label
                tensor.
        """
        batch_size = params["batch_size"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # read the dataset
        dataset = tf.data.Dataset.list_files(filename, shuffle=is_training)
        if is_training:
            dataset = dataset.repeat()
        def fetch_dataset(filename):
            buffer_size = 8 * 1024 * 1024 # 8 MiB per file
            dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
            return dataset
        dataset = dataset.apply(
            tf.contrib.data.parallel_interleave(
                fetch_dataset, cycle_length=64, sloppy=True))
        if is_training:
            dataset = dataset.shuffle(1024)

        # augment and batch
#         dataset = dataset.apply(
#             tf.contrib.data.map_and_batch(
#                 read_and_preprocess, batch_size=batch_size,
#                 num_parallel_batches=num_cores, drop_remainder=True
#             )
#         )
#         # Create list of files that match pattern.
#         file_list = tf.gfile.Glob(filename=filename)

#         # Create dataset from file list.
#         dataset = tf.data.TFRecordDataset(
#             filenames=file_list, num_parallel_reads=tf.contrib.data.AUTOTUNE
#         )

#         # Shuffle and repeat if training with fused op.
#         if mode == tf.estimator.ModeKeys.TRAIN:
#             dataset = dataset.apply(
#                 tf.contrib.data.shuffle_and_repeat(
#                     buffer_size=50 * params["batch_size"],
#                     count=None  # indefinitely
#                 )
#             )

        # Decode CSV file into a features dictionary of tensors, then batch.
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                map_func=lambda x: decode_example(
                    protos=x,
                    params=params
                ),
                batch_size=params["batch_size"],
                num_parallel_batches=8,
                drop_remainder=True,
#                 num_parallel_calls=tf.contrib.data.AUTOTUNE
            )
        )

        # Assign static shape.
        dataset = dataset.map(lambda x, y: set_static_shape(features=x, labels=y, batch_size=batch_size))

        # Prefetch data to improve latency.
        dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

        # Create a iterator, then get batch of features from example queue.
        batched_dataset = dataset.make_one_shot_iterator().get_next()

        return batched_dataset
    return _input_fn
