import tensorflow as tf

from . import image_utils
from .print_object import print_obj


def decode_example(protos, mode, params):
    """Decodes TFRecord file into tensors.

    Given protobufs, decode into image and label tensors.

    Args:
        protos: protobufs from TFRecord file.
        mode: tf.estimator.ModeKeys with values of either TRAIN or EVAL.
        params: dict, user passed parameters.

    Returns:
        Image tensors from domain a and 2.
    """
    func_name = "decode_example"

    # Create feature schema map for protos.
    feature_schema = {
        "domain_a_image_raw": tf.FixedLenFeature(shape=[], dtype=tf.string),
        "domain_b_image_raw": tf.FixedLenFeature(shape=[], dtype=tf.string)
    }

    # Parse features from tf.Example.
    parsed_features = tf.parse_single_example(
        serialized=protos, features=feature_schema
    )
    print_obj("\n" + func_name, "parsed_features", parsed_features)

    # Decode source image.
    domain_a_image = image_utils.handle_input_image(
        image_bytes=parsed_features["domain_a_image_raw"],
        mode=mode,
        params=params
    )
    print_obj(func_name, "domain_a_image", domain_a_image)

    # Decode target image.
    domain_b_image = image_utils.handle_input_image(
        image_bytes=parsed_features["domain_b_image_raw"],
        mode=mode,
        params=params
    )
    print_obj(func_name, "domain_b_image", domain_b_image)

    return {"domain_a_image": domain_a_image, "domain_b_image": domain_b_image}


def set_static_shape(features, batch_size, params):
    """Sets static shape of batched input tensors in dataset.
    Args:
        features: dict, keys are feature names and values are feature tensors.
        batch_size: int, number of examples per batch.
        params:
    Returns:
        Features tensor dictionary and labels tensor.
    """
    features["domain_a_image"].set_shape(
        features["domain_a_image"].get_shape().merge_with(
            tf.TensorShape(dims=[batch_size, None, None, None])
        )
    )

    features["domain_b_image"].set_shape(
        features["domain_b_image"].get_shape().merge_with(
            tf.TensorShape(dims=[batch_size, None, None, None])
        )
    )
    return features


def read_dataset(filename, mode, batch_size, params):
    """Read data using tf.data, doing necessary preprocessing.

    Given filename, mode, batch size, and other parameters, read dataset
    using Dataset API, apply necessary preprocessing, and return an input
    function to the Estimator API.

    Args:
        filename: str, file pattern to read into our tf.data dataset.
        mode: The estimator ModeKeys. Can be TRAIN or EVAL.
        batch_size: int, number of examples per batch.
        params: dict, dictionary of user passed parameters.

    Returns:
        An input function.
    """
    def _input_fn():
        """Wrapper input function used by Estimator API to get data tensors.

        Returns:
            Batched dataset object of dictionary of feature tensors and label
                tensor.
        """
        # Create list of files that match pattern.
        file_list = tf.gfile.Glob(filename=filename)

        # Create dataset from file list.
        if params["input_fn_autotune"]:
            dataset = tf.data.TFRecordDataset(
                filenames=file_list,
                num_parallel_reads=tf.contrib.data.AUTOTUNE
            )
        else:
            dataset = tf.data.TFRecordDataset(filenames=file_list)

        # Shuffle and repeat if training with fused op.
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(
                    buffer_size=50 * batch_size,
                    count=None  # indefinitely
                )
            )

        # Decode file into a features dictionary of tensors, then batch.
        if params["input_fn_autotune"]:
            dataset = dataset.apply(
                tf.contrib.data.map_and_batch(
                    map_func=lambda x: decode_example(
                        protos=x,
                        mode=mode,
                        params=params
                    ),
                    batch_size=batch_size,
                    num_parallel_calls=tf.contrib.data.AUTOTUNE
                )
            )
        else:
            dataset = dataset.apply(
                tf.contrib.data.map_and_batch(
                    map_func=lambda x: decode_example(
                        protos=x,
                        mode=mode,
                        params=params
                    ),
                    batch_size=batch_size,
                    drop_remainder=True
                )
            )

        # Assign static shape, namely make the batch size axis static.
        dataset = dataset.map(
            map_func=lambda x: set_static_shape(
                features=x, batch_size=batch_size, params=params
            )
        )
        # Prefetch data to improve latency.
        if params["input_fn_autotune"]:
            dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
        else:
            dataset = dataset.prefetch(buffer_size=1)

        # Create a iterator, then get batch of features from example queue.
        batched_dataset = dataset.make_one_shot_iterator().get_next()

        return batched_dataset
    return _input_fn
