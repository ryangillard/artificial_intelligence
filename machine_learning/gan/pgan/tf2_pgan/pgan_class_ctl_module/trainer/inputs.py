import tensorflow as tf


def preprocess_image(image):
    """Preprocess image tensor.

    Args:
        image: tensor, input image with shape
            [batch_size, height, width, depth].

    Returns:
        Preprocessed image tensor with shape
            [batch_size, height, width, depth].
    """
    # Convert from [0, 255] -> [-1.0, 1.0] floats.
    image = tf.cast(x=image, dtype=tf.float32) * (2. / 255) - 1.0

    return image


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
        "image_raw": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "label": tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
    }

    # Parse features from tf.Example.
    parsed_features = tf.io.parse_single_example(
        serialized=protos, features=features
    )

    # Convert from a scalar string tensor (whose single string has
    # length height * width * depth) to a uint8 tensor with shape
    # [height * width * depth].
    image = tf.io.decode_raw(
        input_bytes=parsed_features["image_raw"], out_type=tf.uint8
    )

    # Reshape flattened image back into normal dimensions.
    image = tf.reshape(
        tensor=image,
        shape=[params["height"], params["width"], params["depth"]]
    )

    # Preprocess image.
    image = preprocess_image(image=image)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(x=parsed_features["label"], dtype=tf.int32)

    return {"image": image}, label


def set_static_shape(features, labels, batch_size):
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


def read_dataset(filename, batch_size, params, training):
    """Reads TF Record data using tf.data, doing necessary preprocessing.

    Given filename, mode, batch size, and other parameters, read TF Record
    dataset using Dataset API, apply necessary preprocessing, and return an
    input function to the Estimator API.

    Args:
        filename: str, file pattern that to read into our tf.data dataset.
        batch_size: int, number of examples per batch.
        params: dict, dictionary of user passed parameters.
        training: bool, if training or not.

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


    def _input_fn():
        """Wrapper input function used by Estimator API to get data tensors.

        Returns:
            Batched dataset object of dictionary of feature tensors and label
                tensor.
        """
        # Create dataset to contain list of files matching pattern.
        dataset = tf.data.Dataset.list_files(
            file_pattern=filename, shuffle=training
        )

        # Repeat dataset files indefinitely if in training.
        if training:
            dataset = dataset.repeat()

        # Parallel interleaves multiple files at once with map function.
        dataset = dataset.apply(
            tf.data.experimental.parallel_interleave(
                map_func=fetch_dataset, cycle_length=64, sloppy=True
            )
        )

        # Shuffle the Dataset TFRecord Examples if in training.
        if training:
            dataset = dataset.shuffle(buffer_size=1024)

        # Decode TF Record Example into a features dictionary of tensors.
        dataset = dataset.map(
            map_func=lambda x: decode_example(
                protos=x,
                params=params
            ),
            num_parallel_calls=(
                tf.contrib.data.AUTOTUNE
                if params["input_fn_autotune"]
                else None
            )
        )

        # Batch dataset and drop remainder so there are no partial batches.
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

        # Assign static shape, namely make the batch size axis static.
        dataset = dataset.map(
            map_func=lambda x, y: set_static_shape(
                features=x, labels=y, batch_size=batch_size
            )
        )

        # Prefetch data to improve latency.
        dataset = dataset.prefetch(
            buffer_size=(
                tf.data.experimental.AUTOTUNE
                if params["input_fn_autotune"]
                else 1
            )
        )

        return dataset

    return _input_fn
