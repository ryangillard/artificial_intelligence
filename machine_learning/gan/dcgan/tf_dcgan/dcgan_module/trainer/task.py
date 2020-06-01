import argparse
import json
import os
import shutil

from . import model


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
        "--generator_final_num_filters",
        help="Number of filters for final generator conv layer.",
        type=int,
        default=3
    )
    parser.add_argument(
        "--generator_final_kernel_size",
        help="Kernel sizes for final generator conv layer.",
        type=int,
        default=5
    )
    parser.add_argument(
        "--generator_final_stride",
        help="Strides for final generator conv layer.",
        type=int,
        default=2
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
        default=0.1
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
        "--discriminator_dropout_rates",
        help="Dropout rates for discriminator dropout layers.",
        type=str,
        default="0.3,0.3"
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
        default=0.1
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

    # Parse all arguments.
    args = parser.parse_args()
    arguments = args.__dict__

    # Unused args provided by service.
    arguments.pop("job_dir", None)
    arguments.pop("job-dir", None)

    # Fix eval steps.
    if arguments["eval_steps"] == "None":
        arguments["eval_steps"] = None
    else:
        arguments["eval_steps"] = int(arguments["eval_steps"])

    # Fix generator_projection_dims.
    arguments["generator_projection_dims"] = [
        int(x)
        for x in arguments["generator_projection_dims"].split(",")
    ]

    # Fix num_filters.
    arguments["generator_num_filters"] = [
        int(x)
        for x in arguments["generator_num_filters"].split(",")
    ]

    arguments["discriminator_num_filters"] = [
        int(x)
        for x in arguments["discriminator_num_filters"].split(",")
    ]

    # Fix kernel_sizes.
    arguments["generator_kernel_sizes"] = [
        int(x)
        for x in arguments["generator_kernel_sizes"].split(",")
    ]

    arguments["discriminator_kernel_sizes"] = [
        int(x)
        for x in arguments["discriminator_kernel_sizes"].split(",")
    ]

    # Fix strides.
    arguments["generator_strides"] = [
        int(x)
        for x in arguments["generator_strides"].split(",")
    ]

    arguments["discriminator_strides"] = [
        int(x)
        for x in arguments["discriminator_strides"].split(",")
    ]

    # Fix discriminator_dropout_rates.
    arguments["discriminator_dropout_rates"] = [
        float(x)
        for x in arguments["discriminator_dropout_rates"].split(",")
    ]

    # Fix clip_gradients.
    if arguments["generator_clip_gradients"] == "None":
        arguments["generator_clip_gradients"] = None
    else:
        arguments["generator_clip_gradients"] = float(
            arguments["generator_clip_gradients"]
        )

    if arguments["discriminator_clip_gradients"] == "None":
        arguments["discriminator_clip_gradients"] = None
    else:
        arguments["discriminator_clip_gradients"] = float(
            arguments["discriminator_clip_gradients"]
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

    # Start fresh output directory.
    shutil.rmtree(path=arguments["output_dir"], ignore_errors=True)

    # Run the training job.
    model.train_and_evaluate(arguments)
