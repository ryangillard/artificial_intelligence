import argparse
import json
import os

from . import model


def convert_string_to_bool(string):
    """Converts string to bool.
    Args:
        string: str, string to convert.
    Returns:
        Boolean conversion of string.
    """
    return False if string.lower() == "false" else True


def convert_string_to_none_or_float(string):
    """Converts string to None or float.

    Args:
        string: str, string to convert.

    Returns:
        None or float conversion of string.
    """
    return None if string.lower() == "none" else float(string)


def convert_string_to_none_or_int(string):
    """Converts string to None or int.

    Args:
        string: str, string to convert.

    Returns:
        None or int conversion of string.
    """
    return None if string.lower() == "none" else int(string)


def convert_string_to_list_of_bools(string, sep):
    """Converts string to list of bools.

    Args:
        string: str, string to convert.
        sep: str, separator string.

    Returns:
        List of bools conversion of string.
    """
    return [convert_string_to_bool(x) for x in string.split(sep)]


def convert_string_to_list_of_ints(string, sep):
    """Converts string to list of ints.

    Args:
        string: str, string to convert.
        sep: str, separator string.

    Returns:
        List of ints conversion of string.
    """
    return [int(x) for x in string.split(sep)]


def convert_string_to_list_of_floats(string, sep):
    """Converts string to list of floats.

    Args:
        string: str, string to convert.
        sep: str, separator string.

    Returns:
        List of floats conversion of string.
    """
    return [float(x) for x in string.split(sep)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # File arguments.
    parser.add_argument(
        "--train_file_pattern",
        help="GCS location to read training data.",
        required=True
    )
    parser.add_argument(
        "--eval_file_pattern",
        help="GCS location to read evaluation data.",
        required=True
    )
    parser.add_argument(
        "--output_dir",
        help="GCS location to write checkpoints and export models.",
        required=True
    )
    parser.add_argument(
        "--job-dir",
        help="This model ignores this field, but it is required by gcloud.",
        default="junk"
    )

    # Training parameters.
    parser.add_argument(
        "--train_batch_size",
        help="Number of examples in training batch.",
        type=int,
        default=32
    )
    parser.add_argument(
        "--train_steps",
        help="Number of steps to train for.",
        type=int,
        default=100
    )
    parser.add_argument(
        "--save_summary_steps",
        help="How many steps to train before saving a summary.",
        type=int,
        default=100
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
    parser.add_argument(
        "--input_fn_autotune",
        help="Whether to autotune input function performance.",
        type=str,
        default="True"
    )

    # Eval parameters.
    parser.add_argument(
        "--eval_batch_size",
        help="Number of examples in evaluation batch.",
        type=int,
        default=32
    )
    parser.add_argument(
        "--eval_steps",
        help="Number of steps to evaluate for.",
        type=str,
        default="None"
    )
    parser.add_argument(
        "--start_delay_secs",
        help="Number of seconds to wait before first evaluation.",
        type=int,
        default=60
    )
    parser.add_argument(
        "--throttle_secs",
        help="Number of seconds to wait between evaluations.",
        type=int,
        default=120
    )

    # Image parameters.
    parser.add_argument(
        "--height",
        help="Height of image.",
        type=int,
        default=32
    )
    parser.add_argument(
        "--width",
        help="Width of image.",
        type=int,
        default=32
    )
    parser.add_argument(
        "--depth",
        help="Depth of image.",
        type=int,
        default=3
    )

    # Generator parameters.
    parser.add_argument(
        "--latent_size",
        help="The latent size of the noise vector.",
        type=int,
        default=3
    )
    parser.add_argument(
        "--generator_projection_dims",
        help="The 3D dimensions to project latent noise vector into.",
        type=str,
        default="8,8,256"
    )
    parser.add_argument(
        "--generator_num_filters",
        help="Number of filters for generator conv layers.",
        type=str,
        default="128, 64"
    )
    parser.add_argument(
        "--generator_kernel_sizes",
        help="Kernel sizes for generator conv layers.",
        type=str,
        default="5,5"
    )
    parser.add_argument(
        "--generator_strides",
        help="Strides for generator conv layers.",
        type=str,
        default="1,2"
    )
    parser.add_argument(
        "--generator_use_batch_norm",
        help="Whether generator layer uses batch norm.",
        type=str,
        default="True"
    )
    parser.add_argument(
        "--generator_batch_norm_before_act",
        help="Whether generator layers have batch norm before activation.",
        type=str,
        default="True"
    )
    parser.add_argument(
        "--generator_use_leaky_relu",
        help="Whether generator layers use leaky relu activations.",
        type=str,
        default="False"
    )
    parser.add_argument(
        "--generator_leaky_relu_alpha",
        help="The amount of leakyness of generator's leaky relus.",
        type=float,
        default=0.2
    )
    parser.add_argument(
        "--generator_final_activation",
        help="The final activation function of generator.",
        type=str,
        default="None"
    )
    parser.add_argument(
        "--generator_l1_regularization_scale",
        help="Scale factor for L1 regularization for generator.",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--generator_l2_regularization_scale",
        help="Scale factor for L2 regularization for generator.",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--generator_optimizer",
        help="Name of optimizer to use for generator.",
        type=str,
        default="Adam"
    )
    parser.add_argument(
        "--generator_learning_rate",
        help="How quickly we train our model by scaling the gradient for generator.",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--generator_adam_beta1",
        help="Adam optimizer's beta1 hyperparameter for first moment.",
        type=float,
        default=0.9
    )
    parser.add_argument(
        "--generator_adam_beta2",
        help="Adam optimizer's beta2 hyperparameter for second moment.",
        type=float,
        default=0.999
    )
    parser.add_argument(
        "--generator_adam_epsilon",
        help="Adam optimizer's epsilon hyperparameter for numerical stability.",
        type=float,
        default=1e-8
    )
    parser.add_argument(
        "--generator_clip_gradients",
        help="Global clipping to prevent gradient norm to exceed this value for generator.",
        type=str,
        default="None"
    )
    parser.add_argument(
        "--generator_train_steps",
        help="Number of steps to train generator for per cycle.",
        type=int,
        default=100
    )

    # Discriminator parameters.
    parser.add_argument(
        "--discriminator_num_filters",
        help="Number of filters for discriminator conv layers.",
        type=str,
        default="64, 128"
    )
    parser.add_argument(
        "--discriminator_kernel_sizes",
        help="Kernel sizes for discriminator conv layers.",
        type=str,
        default="5,5"
    )
    parser.add_argument(
        "--discriminator_strides",
        help="Strides for discriminator conv layers.",
        type=str,
        default="1,2"
    )
    parser.add_argument(
        "--discriminator_use_batch_norm",
        help="Whether discriminator layer uses batch norm.",
        type=str,
        default="True"
    )
    parser.add_argument(
        "--discriminator_batch_norm_before_act",
        help="Whether discriminator layers have batch norm before activation.",
        type=str,
        default="True"
    )
    parser.add_argument(
        "--discriminator_use_leaky_relu",
        help="Whether discriminator layers use leaky relu activations.",
        type=str,
        default="False"
    )
    parser.add_argument(
        "--discriminator_leaky_relu_alpha",
        help="The amount of leakyness of discriminator's leaky relus.",
        type=float,
        default=0.2
    )
    parser.add_argument(
        "--discriminator_l1_regularization_scale",
        help="Scale factor for L1 regularization for discriminator.",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--discriminator_l2_regularization_scale",
        help="Scale factor for L2 regularization for discriminator.",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--discriminator_optimizer",
        help="Name of optimizer to use for discriminator.",
        type=str,
        default="Adam"
    )
    parser.add_argument(
        "--discriminator_learning_rate",
        help="How quickly we train our model by scaling the gradient for discriminator.",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--discriminator_adam_beta1",
        help="Adam optimizer's beta1 hyperparameter for first moment.",
        type=float,
        default=0.9
    )
    parser.add_argument(
        "--discriminator_adam_beta2",
        help="Adam optimizer's beta2 hyperparameter for second moment.",
        type=float,
        default=0.999
    )
    parser.add_argument(
        "--discriminator_adam_epsilon",
        help="Adam optimizer's epsilon hyperparameter for numerical stability.",
        type=float,
        default=1e-8
    )
    parser.add_argument(
        "--discriminator_clip_gradients",
        help="Global clipping to prevent gradient norm to exceed this value for discriminator.",
        type=str,
        default="None"
    )
    parser.add_argument(
        "--discriminator_train_steps",
        help="Number of steps to train discriminator for per cycle.",
        type=int,
        default=100
    )

    # Loss parameters.
    parser.add_argument(
        "--loss_a",
        help="Loss constant a for discriminator's fake labels.",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--loss_b",
        help="Loss constant b for discriminator's real labels.",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--loss_c",
        help="Loss constant c for generator's fake labels.",
        type=float,
        default=1.0
    )

    # Parse all arguments.
    args = parser.parse_args()
    arguments = args.__dict__

    # Unused args provided by service.
    arguments.pop("job_dir", None)
    arguments.pop("job-dir", None)

    # Fix input_fn_autotune.
    arguments["input_fn_autotune"] = convert_string_to_bool(
        string=arguments["input_fn_autotune"]
    )

    # Fix eval steps.
    arguments["eval_steps"] = convert_string_to_none_or_int(
        string=arguments["eval_steps"])

    # Fix generator_projection_dims.
    arguments["generator_projection_dims"] = convert_string_to_list_of_ints(
        string=arguments["generator_projection_dims"], sep=","
    )

    # Fix num_filters.
    arguments["generator_num_filters"] = convert_string_to_list_of_ints(
        string=arguments["generator_num_filters"], sep=","
    )

    arguments["discriminator_num_filters"] = convert_string_to_list_of_ints(
        string=arguments["discriminator_num_filters"], sep=","
    )

    # Fix kernel_sizes.
    arguments["generator_kernel_sizes"] = convert_string_to_list_of_ints(
        string=arguments["generator_kernel_sizes"], sep=","
    )

    arguments["discriminator_kernel_sizes"] = convert_string_to_list_of_ints(
        string=arguments["discriminator_kernel_sizes"], sep=","
    )

    # Fix strides.
    arguments["generator_strides"] = convert_string_to_list_of_ints(
        string=arguments["generator_strides"], sep=","
    )

    arguments["discriminator_strides"] = convert_string_to_list_of_ints(
        string=arguments["discriminator_strides"], sep=","
    )

    # Fix use_batch_norm.
    arguments["generator_use_batch_norm"] = convert_string_to_list_of_bools(
        string=arguments["generator_use_batch_norm"], sep=","
    )

    arguments["discriminator_use_batch_norm"] = convert_string_to_list_of_bools(
        string=arguments["discriminator_use_batch_norm"], sep=","
    )

    # Fix batch_norm_before_act.
    arguments["generator_batch_norm_before_act"] = convert_string_to_bool(
        string=arguments["generator_batch_norm_before_act"]
    )

    arguments["discriminator_batch_norm_before_act"] = convert_string_to_bool(
        string=arguments["discriminator_batch_norm_before_act"]
    )

    # Fix use_leaky_relu.
    arguments["generator_use_leaky_relu"] = convert_string_to_bool(
        string=arguments["generator_use_leaky_relu"]
    )

    arguments["discriminator_use_leaky_relu"] = convert_string_to_bool(
        string=arguments["discriminator_use_leaky_relu"]
    )

    # Fix clip_gradients.
    arguments["generator_clip_gradients"] = convert_string_to_none_or_float(
        string=arguments["generator_clip_gradients"]
    )

    arguments["discriminator_clip_gradients"] = convert_string_to_none_or_float(
        string=arguments["discriminator_clip_gradients"]
    )

    # Append trial_id to path if we are doing hptuning.
    # This code can be removed if you are not using hyperparameter tuning.
    arguments["output_dir"] = os.path.join(
        arguments["output_dir"],
        json.loads(
            os.environ.get(
                "TF_CONFIG", "{}"
            )
        ).get("task", {}).get("trial", ""))

    # Run the training job.
    model.train_and_evaluate(arguments)
