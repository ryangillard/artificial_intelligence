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
    func_name = "decode_example"
    # Create feature schema map for protos.
    if params["dataset"] == "cifar10":
        features = {
            "image_raw": tf.FixedLenFeature(shape=[], dtype=tf.string),
            "label": tf.FixedLenFeature(shape=[], dtype=tf.int64)
        }
    elif params["dataset"] == "celeba_hq":
        features = {
            "image_raw": tf.FixedLenFeature(shape=[], dtype=tf.string)
        }

    # Parse features from tf.Example.
    parsed_features = tf.parse_single_example(
        serialized=protos, features=features
    )
    print_obj("\n" + func_name, "features", features)

    # Convert from a scalar string tensor (whose single string has
    # length height * width * depth) to a uint8 tensor with shape
    # [height * width * depth].
    if params["dataset"] == "cifar10":
        image = tf.decode_raw(
            input_bytes=parsed_features["image_raw"], out_type=tf.uint8
        )
    elif params["dataset"] == "celeba_hq":
        image = tf.image.decode_jpeg(
            contents=parsed_features["image_raw"], channels=3
        )
    print_obj(func_name, "image", image)

    # Reshape flattened image back into normal dimensions.
    image = tf.reshape(
        tensor=image,
        shape=[params["height"], params["width"], params["depth"]]
    )
    print_obj(func_name, "image", image)

    # Preprocess image.
    image = image_utils.preprocess_image(image=image, params=params)
    print_obj(func_name, "image", image)

    if params["dataset"] == "cifar10":
        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(x=parsed_features["label"], dtype=tf.int32)
    elif params["dataset"] == "celeba_hq":
        label = tf.zeros(shape=[], dtype=tf.int32)
    print_obj(func_name, "label", label)

    return {"image": image}, label


def set_static_shape(features, labels, batch_size, params):
    """Sets static shape of batched input tensors in dataset.

    Args:
        features: dict, keys are feature names and values are feature tensors.
        labels: tensor, label data.
        batch_size: int, number of examples per batch.

    Returns:
        Features tensor dictionary and labels tensor.
    """
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
    """Reads TF Record data using tf.data, doing necessary preprocessing.

    Given filename, mode, batch size, and other parameters, read TF Record
    dataset using Dataset API, apply necessary preprocessing, and return an
    input function to the Estimator API.

    Args:
        filename: str, file pattern that to read into our tf.data dataset.
        mode: The estimator ModeKeys. Can be TRAIN or EVAL.
        batch_size: int, number of examples per batch.
        params: dict, dictionary of user passed parameters.

    Returns:
        An input function.
    """
    def fetch_dataset(filename):
        """Fetches TFRecord Dataset from given filename.

        Args:
            filename: str, name of TFRecord file.
        Returns:
            Dataset containing TFRecord Examples.
        """
        buffer_size = 8 * 1024 * 1024  # 8 MiB per file
        dataset = tf.data.TFRecordDataset(
            filenames=filename, buffer_size=buffer_size
        )

        return dataset

    def _input_fn(params):
        """Wrapper input function used by Estimator API to get data tensors.

        Args:
            params: dict, created by TPU job that contains the per core batch
                size.
        Returns:
            Batched dataset object of dictionary of feature tensors and label
                tensor.
        """
        # Extract per core batch size from created dict for TPU.
        batch_size = params["batch_size"]

        # Determine if we are in train or eval mode.
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # Create dataset to contain list of files matching pattern.
        dataset = tf.data.Dataset.list_files(
            file_pattern=filename, shuffle=is_training
        )

        # Repeat dataset files indefinitely if in training.
        if is_training:
            dataset = dataset.repeat()

        # Parallel interleaves multiple files at once with map function.
        dataset = dataset.apply(
            tf.contrib.data.parallel_interleave(
                map_func=fetch_dataset, cycle_length=64, sloppy=True
            )
        )

        # Shuffle the Dataset TFRecord Examples if in training.
        if is_training:
            dataset = dataset.shuffle(buffer_size=1024)

        # Decode CSV file into a features dictionary of tensors, then batch.
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                map_func=lambda x: decode_example(
                    protos=x,
                    params=params
                ),
                batch_size=batch_size,
                num_parallel_batches=8,
                drop_remainder=True,
            )
        )

        # Assign static shape, namely make the batch size axis static.
        dataset = dataset.map(
            map_func=lambda x, y: set_static_shape(
                features=x, labels=y, batch_size=batch_size, params=params
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
