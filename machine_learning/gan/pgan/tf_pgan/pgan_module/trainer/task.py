import argparse
import json
import os

from . import model


def calc_generator_discriminator_conv_layer_properties(
        conv_num_filters, conv_kernel_sizes, conv_strides, depth):
    """Calculates generator and discriminator conv layer properties.

    Args:
        num_filters: list, nested list of ints of the number of filters
            for each conv layer.
        kernel_sizes: list, nested list of ints of the kernel sizes for
            each conv layer.
        strides: list, nested list of ints of the strides for each conv
            layer.
        depth: int, depth dimension of images.

    Returns:
        Nested lists of conv layer properties for both generator and
            discriminator.
    """
    def make_generator(num_filters, kernel_sizes, strides, depth):
        """Calculates generator conv layer properties.

        Args:
            num_filters: list, nested list of ints of the number of filters
                for each conv layer.
            kernel_sizes: list, nested list of ints of the kernel sizes for
                each conv layer.
            strides: list, nested list of ints of the strides for each conv
                layer.
            depth: int, depth dimension of images.

        Returns:
            Nested list of conv layer properties for generator.
        """
        # Get the number of growths.
        num_growths = len(num_filters) - 1

        # Make base block.
        in_out = num_filters[0]
        base = [
            [kernel_sizes[0][i]] * 2 + in_out + [strides[0][i]] * 2
            for i in range(len(num_filters[0]))
        ]
        blocks = [base]

        # Add growth blocks.
        for i in range(1, num_growths + 1):
            in_out = [[blocks[i - 1][-1][-3], num_filters[i][0]]]
            block = [[kernel_sizes[i][0]] * 2 + in_out[0] + [strides[i][0]] * 2]
            for j in range(1, len(num_filters[i])):
                in_out.append([block[-1][-3], num_filters[i][j]])
                block.append(
                    [kernel_sizes[i][j]] * 2 + in_out[j] + [strides[i][j]] * 2
                )
            blocks.append(block)

        # Add toRGB conv.
        blocks[-1].append([1, 1, blocks[-1][-1][-3], depth] + [1] * 2)

        return blocks

    def make_discriminator(generator):
        """Calculates discriminator conv layer properties.

        Args:
            generator: list, nested list of conv layer properties for
                generator.

        Returns:
            Nested list of conv layer properties for discriminator.
        """
        # Reverse generator.
        discriminator = generator[::-1]

        # Reverse input and output shapes.
        discriminator = [
            [
                conv[0:2] + conv[2:4][::-1] + conv[-2:]
                for conv in block[::-1]
            ]
            for block in discriminator
        ]

        return discriminator

    # Calculate conv layer properties for generator using args.
    generator = make_generator(
        conv_num_filters, conv_kernel_sizes, conv_strides, depth
    )

    # Calculate conv layer properties for discriminator using generator
    # properties.
    discriminator = make_discriminator(generator)

    return generator, discriminator


def split_up_generator_conv_layer_properties(
        generator, num_filters, strides, depth):
    """Splits up generator conv layer properties into lists.

    Args:
        generator: list, nested list of conv layer properties for
            generator.
        num_filters: list, nested list of ints of the number of filters
            for each conv layer.
        strides: list, nested list of ints of the strides for each conv
            layer.
        depth: int, depth dimension of images.

    Returns:
        Nested lists of conv layer properties for generator.
    """
    generator_base_conv_blocks = [generator[0][0:len(num_filters[0])]]

    generator_growth_conv_blocks = []
    if len(num_filters) > 1:
        generator_growth_conv_blocks = generator[1:-1] + [generator[-1][:-1]]

    generator_to_rgb_layers = [
        [[1] * 2 + [num_filters[i][0]] + [depth] + [strides[i][0]] * 2]
        for i in range(len(num_filters))
    ]

    return (generator_base_conv_blocks,
            generator_growth_conv_blocks,
            generator_to_rgb_layers)


def split_up_discriminator_conv_layer_properties(
        discriminator, num_filters, strides, depth):
    """Splits up discriminator conv layer properties into lists.

    Args:
        discriminator: list, nested list of conv layer properties for
            discriminator.
        num_filters: list, nested list of ints of the number of filters
            for each conv layer.
        strides: list, nested list of ints of the strides for each conv
            layer.
        depth: int, depth dimension of images.

    Returns:
        Nested lists of conv layer properties for discriminator.
    """
    discriminator_from_rgb_layers = [
        [[1] * 2 + [depth] + [num_filters[i][0]] + [strides[i][0]] * 2]
        for i in range(len(num_filters))
    ]

    if len(num_filters) > 1:
        discriminator_base_conv_blocks = [discriminator[-1]]
    else:
        discriminator_base_conv_blocks = [discriminator[-1][1:]]

    discriminator_growth_conv_blocks = []
    if len(num_filters) > 1:
        discriminator_growth_conv_blocks = [discriminator[0][1:]] + discriminator[1:-1]
        discriminator_growth_conv_blocks = discriminator_growth_conv_blocks[::-1]

    return (discriminator_from_rgb_layers,
            discriminator_base_conv_blocks,
            discriminator_growth_conv_blocks)


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

    # Serving parameters.
    parser.add_argument(
        "--exports_to_keep",
        help="Number of exports to keep before overwriting oldest.",
        type=int,
        default=5
    )
    parser.add_argument(
        "--predict_all_resolutions",
        help="If want all resolutions predicted or just largest one.",
        type=str,
        default="True"
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

    # Shared parameters.
    parser.add_argument(
        "--num_steps_until_growth",
        help="Number of steps until layer added to generator & discriminator.",
        type=int,
        default=100
    )
    parser.add_argument(
        "--conv_num_filters",
        help="Number of filters for growth conv layers.",
        type=str,
        default="512,512;512,512"
    )
    parser.add_argument(
        "--conv_kernel_sizes",
        help="Kernel sizes for growth conv layers.",
        type=str,
        default="3,3;3,3"
    )
    parser.add_argument(
        "--conv_strides",
        help="Strides for growth conv layers.",
        type=str,
        default="1,1;1,1"
    )

    # Generator parameters.
    parser.add_argument(
        "--latent_size",
        help="The latent size of the noise vector.",
        type=int,
        default=3
    )
    parser.add_argument(
        "--use_pixel_norm",
        help="If want to use pixel norm op after each convolution.",
        type=str,
        default="True"
    )
    parser.add_argument(
        "--pixel_norm_epsilon",
        help="Small value to add to denominator for numerical stability.",
        type=float,
        default=1e-8
    )
    parser.add_argument(
        "--normalize_latent",
        help="If want to normalize latent vector before projection.",
        type=str,
        default="True"
    )
    parser.add_argument(
        "--generator_projection_dims",
        help="The 3D dimensions to project latent noise vector into.",
        type=str,
        default="8,8,256"
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
        "--use_minibatch_stddev",
        help="If want to use minibatch stddev op before first base conv layer.",
        type=str,
        default="True"
    )
    parser.add_argument(
        "--minibatch_stddev_group_size",
        help="The size of groups to split minibatch examples into.",
        type=int,
        default=4
    )
    parser.add_argument(
        "--minibatch_stddev_averaging",
        help="If want to average across feature maps and pixels for minibatch stddev.",
        type=str,
        default="True"
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
        "--discriminator_gradient_penalty_coefficient",
        help="Coefficient of gradient penalty for discriminator.",
        type=float,
        default=10.0
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
    if arguments["eval_steps"].lower() == "none":
        arguments["eval_steps"] = None
    else:
        arguments["eval_steps"] = int(arguments["eval_steps"])

    # Fix predict_all_resolutions.
    if arguments["predict_all_resolutions"].lower() == "false":
        arguments["predict_all_resolutions"] = False
    else:
        arguments["predict_all_resolutions"] = True

    # Fix conv layer property parameters.
    arguments["conv_num_filters"] = [
        [int(y) for y in x.split(",")]
        for x in arguments["conv_num_filters"].split(";")
    ]

    arguments["conv_kernel_sizes"] = [
        [int(y) for y in x.split(",")]
        for x in arguments["conv_kernel_sizes"].split(";")
    ]

    arguments["conv_strides"] = [
        [int(y) for y in x.split(",")]
        for x in arguments["conv_strides"].split(";")
    ]

    # Make some assertions.
    assert len(arguments["conv_num_filters"]) > 0
    assert len(arguments["conv_num_filters"]) == len(arguments["conv_kernel_sizes"])
    assert len(arguments["conv_num_filters"]) == len(arguments["conv_strides"])

    # Truncate lists if over the 1024x1024 current limit.
    if len(arguments["conv_num_filters"]) > 9:
        arguments["conv_num_filters"] = arguments["conv_num_filters"][0:10]
        arguments["conv_kernel_sizes"] = arguments["conv_kernel_sizes"][0:10]
        arguments["conv_strides"] = arguments["conv_strides"][0:10]

    # Get conv layer properties for generator and discriminator.
    (generator,
     discriminator) = calc_generator_discriminator_conv_layer_properties(
        arguments["conv_num_filters"],
        arguments["conv_kernel_sizes"],
        arguments["conv_strides"],
        arguments["depth"]
    )

    # Split up generator properties into separate lists.
    (generator_base_conv_blocks,
     generator_growth_conv_blocks,
     generator_to_rgb_layers) = split_up_generator_conv_layer_properties(
        generator,
        arguments["conv_num_filters"],
        arguments["conv_strides"],
        arguments["depth"]
    )
    arguments["generator_base_conv_blocks"] = generator_base_conv_blocks
    arguments["generator_growth_conv_blocks"] = generator_growth_conv_blocks
    arguments["generator_to_rgb_layers"] = generator_to_rgb_layers

    # Split up discriminator properties into separate lists.
    (discriminator_from_rgb_layers,
     discriminator_base_conv_blocks,
     discriminator_growth_conv_blocks) = split_up_discriminator_conv_layer_properties(
        discriminator,
        arguments["conv_num_filters"],
        arguments["conv_strides"],
        arguments["depth"]
    )
    arguments["discriminator_from_rgb_layers"] = discriminator_from_rgb_layers
    arguments["discriminator_base_conv_blocks"] = discriminator_base_conv_blocks
    arguments["discriminator_growth_conv_blocks"] = discriminator_growth_conv_blocks

    # Fix normalize_latent.
    if arguments["normalize_latent"].lower() == "false":
        arguments["normalize_latent"] = False
    else:
        arguments["normalize_latent"] = True

    # Fix use_pixel_norm.
    if arguments["use_pixel_norm"].lower() == "false":
        arguments["use_pixel_norm"] = False
    else:
        arguments["use_pixel_norm"] = True

    # Fix generator_projection_dims.
    arguments["generator_projection_dims"] = [
        int(x)
        for x in arguments["generator_projection_dims"].split(",")
    ]

    # Fix use_minibatch_stddev.
    if arguments["use_minibatch_stddev"].lower() == "false":
        arguments["use_minibatch_stddev"] = False
    else:
        arguments["use_minibatch_stddev"] = True

    # Fix clip_gradients.
    if arguments["generator_clip_gradients"].lower() == "none":
        arguments["generator_clip_gradients"] = None
    else:
        arguments["generator_clip_gradients"] = float(
            arguments["generator_clip_gradients"]
        )

    if arguments["discriminator_clip_gradients"].lower() == "none":
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

    # Run the training job.
    model.train_and_evaluate(arguments)
