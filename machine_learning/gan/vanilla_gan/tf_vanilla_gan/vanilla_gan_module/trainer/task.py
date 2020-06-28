import argparse
import json
import os

from . import model


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


def convert_string_to_list_of_ints(string, sep):
    """Converts string to list of ints.

    Args:
        string: str, string to convert.
        sep: str, separator string.

    Returns:
        List of ints conversion of string.
    """
    return [int(x) for x in string.split(sep)]


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
        "--generator_hidden_units",
        help="Hidden layer sizes to use for generator.",
        type=str,
        default="2,4,8"
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
        "--discriminator_hidden_units",
        help="Hidden layer sizes to use for discriminator.",
        type=str,
        default="2,4,8"
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
    parser.add_argument(
        "--label_smoothing",
        help="Multiplier when making real labels instead of all ones.",
        type=float,
        default=0.9
    )

    # Parse all arguments.
    args = parser.parse_args()
    arguments = args.__dict__

    # Unused args provided by service.
    arguments.pop("job_dir", None)
    arguments.pop("job-dir", None)

    # Fix eval steps.
    arguments["eval_steps"] = convert_string_to_none_or_int(
        string=arguments["eval_steps"])

    # Fix hidden_units.
    arguments["generator_hidden_units"] = convert_string_to_list_of_ints(
        string=arguments["generator_hidden_units"], sep=","
    )

    arguments["discriminator_hidden_units"] = convert_string_to_list_of_ints(
        string=arguments["discriminator_hidden_units"], sep=","
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
