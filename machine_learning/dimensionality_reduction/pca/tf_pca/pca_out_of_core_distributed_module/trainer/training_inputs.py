import tensorflow as tf


def decode_example(protos, params):
    """Decodes TFRecord file into tensors.

    Given protobufs, decode into image and label tensors.

    Args:
        protos: protobufs from TFRecord file.
        params: dict, user passed parameters.

    Returns:
        Image and label tensors.
    """
    dtype_map = {
        "str": tf.string,
        "int": tf.int64,
        "float": tf.float32
    }

    # Create feature schema map for protos.
    tf_example_features = {
        feat["name"]: (
            tf.io.FixedLenFeature(
                shape=feat["shape"], dtype=dtype_map[feat["dtype"]]
            )
            if feat["type"] == "FixedLen"
            else tf.io.FixedLenSequenceFeature(
                shape=feat["shape"], dtype=dtype_map[feat["dtype"]]
            )
        )
        for feat in params["tf_record_example_schema"]
    }

    # Parse features from tf.Example.
    parsed_features = tf.io.parse_single_example(
        serialized=protos, features=tf_example_features
    )

    # Convert from a scalar string tensor (whose single string has
    # length height * width * depth) to a uint8 tensor with shape
    # [height * width * depth].
    if params["image_encoding"] == "raw":
        image = tf.io.decode_raw(
            input_bytes=parsed_features[params["image_feature_name"]],
            out_type=tf.uint8
        )
    elif params["image_encoding"] == "png":
        image = tf.io.decode_png(
            contents=parsed_features[params["image_feature_name"]],
            channels=params["image_depth"]
        )
    elif params["image_encoding"] == "jpeg":
        image = tf.io.decode_jpeg(
            contents=parsed_features[params["image_feature_name"]],
            channels=params["image_depth"]
        )

    # Reshape flattened image back into normal dimensions.
    image = tf.reshape(
        tensor=image,
        shape=[
            params["image_height"],
            params["image_width"],
            params["image_depth"]
        ]
    )

    return image


def read_dataset(file_pattern, batch_size, params):
    """Reads TF Record data using tf.data, doing necessary preprocessing.

    Given filename, mode, batch size, and other parameters, read TF Record
    dataset using Dataset API, apply necessary preprocessing, and return an
    input function to the Estimator API.

    Args:
        file_pattern: str, file pattern that to read into our tf.data dataset.
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

    def _input_fn():
        """Wrapper input function used by Estimator API to get data tensors.

        Returns:
            Batched dataset object of dictionary of feature tensors and label
                tensor.
        """
        # Create dataset to contain list of files matching pattern.
        dataset = tf.data.Dataset.list_files(
            file_pattern=file_pattern, shuffle=False
        )

        # Parallel interleaves multiple files at once with map function.
        dataset = dataset.apply(
            tf.data.experimental.parallel_interleave(
                map_func=fetch_dataset, cycle_length=64, sloppy=True
            )
        )

        # Decode TF Record Example into a features dictionary of tensors.
        dataset = dataset.map(
            map_func=lambda x: decode_example(
                protos=x, params=params
            ),
            num_parallel_calls=(
                tf.contrib.data.AUTOTUNE
                if params["input_fn_autotune"]
                else None
            )
        )

        # Batch dataset and drop remainder so there are no partial batches.
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)

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
