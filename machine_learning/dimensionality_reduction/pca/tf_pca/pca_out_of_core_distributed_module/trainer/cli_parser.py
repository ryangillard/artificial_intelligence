
import argparse


def parse_file_arguments(parser):
    """Parses command line file arguments.

    Args:
        parser: instance of `argparse.ArgumentParser`.
    """
    parser.add_argument(
        "--train_file_pattern",
        help="GCS location to read training data.",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_dir",
        help="GCS location to write checkpoints and export models.",
        type=str,
        required=True
    )
    parser.add_argument(
        "--job-dir",
        help="This model ignores this field, but it is required by gcloud.",
        type=str,
        default="junk"
    )


def parse_data_arguments(parser):
    """Parses command line data arguments.

    Args:
        parser: instance of `argparse.ArgumentParser`.
    """
    parser.add_argument(
        "--tf_record_example_schema",
        help="Serialized TF Record Example schema.",
        type=str,
        required=True
    )
    parser.add_argument(
        "--image_feature_name",
        help="Name of image feature.",
        type=str,
        default="image"
    )
    parser.add_argument(
        "--image_encoding",
        help="Encoding of image: raw, png, or jpeg.",
        type=str,
        default="raw"
    )
    parser.add_argument(
        "--image_height",
        help="Height of image.",
        type=int,
        default=32
    )
    parser.add_argument(
        "--image_width",
        help="Width of image.",
        type=int,
        default=32
    )
    parser.add_argument(
        "--image_depth",
        help="Depth of image.",
        type=int,
        default=3
    )
    parser.add_argument(
        "--label_feature_name",
        help="Name of label feature.",
        type=str,
        default="label"
    )


def parse_training_arguments(parser):
    """Parses command line training arguments.

    Args:
        parser: instance of `argparse.ArgumentParser`.
    """
    parser.add_argument(
        "--tf_version",
        help="Version of TensorFlow",
        type=float,
        default=2.3
    )
    parser.add_argument(
        "--use_graph_mode",
        help="Whether to use graph mode or not (eager).",
        type=str,
        default="True"
    )
    parser.add_argument(
        "--distribution_strategy",
        help="Which distribution strategy to use, if any.",
        type=str,
        default=""
    )
    parser.add_argument(
        "--train_dataset_length",
        help="Number of examples in one epoch of training set.",
        type=int,
        default=100
    )
    parser.add_argument(
        "--train_batch_size",
        help="Number of examples in training batch.",
        type=int,
        default=32
    )
    parser.add_argument(
        "--input_fn_autotune",
        help="Whether to autotune input function performance.",
        type=str,
        default="True"
    )
    parser.add_argument(
        "--save_checkpoints_steps",
        help="How many steps to train before saving a checkpoint.",
        type=int,
        default=100
    )
    parser.add_argument(
        "--keep_checkpoint_max",
        help="Max number of checkpoints to keep.",
        type=int,
        default=100
    )


def parse_resnet_arguments(parser):
    """Parses command line ResNet arguments.

    Args:
        parser: instance of `argparse.ArgumentParser`.
    """
    parser.add_argument(
        "--resnet_weights",
        help="The type of weights to use in Resnet, i.e. imagenet.",
        type=str,
        default="imagenet"
    )
    parser.add_argument(
        "--resnet_layer_name",
        help="Number of top principal components to keep.",
        type=str,
        default="conv4_block1_0_conv"
    )
    parser.add_argument(
        "--preprocess_input",
        help="Whether to preprocess input for ResNet.",
        type=str,
        default="True"
    )


def parse_pca_arguments(parser):
    """Parses command line PCA arguments.

    Args:
        parser: instance of `argparse.ArgumentParser`.
    """
    parser.add_argument(
        "--num_cols",
        help="Number of dimensions for each data instance.",
        type=int,
        default=1
    )
    parser.add_argument(
        "--use_sample_covariance",
        help="Whether using sample or population covariance.",
        type=str,
        default="True"
    )
    parser.add_argument(
        "--top_k_pc",
        help="Number of top principal components to keep.",
        type=int,
        default=1
    )


def parse_command_line_arguments():
    """Parses command line arguments and returns dictionary.

    Returns:
        Dictionary containing command line arguments.
    """
    parser = argparse.ArgumentParser()

    # Add various arguments to parser.
    parse_file_arguments(parser)
    parse_data_arguments(parser)
    parse_training_arguments(parser)
    parse_resnet_arguments(parser)
    parse_pca_arguments(parser)

    # Parse all arguments.
    args = parser.parse_args()
    arguments = args.__dict__

    return arguments
