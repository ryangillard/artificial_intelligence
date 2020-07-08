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
    if not string:
        return []
    return [convert_string_to_bool(x) for x in string.split(sep)]


def convert_string_to_list_of_ints(string, sep):
    """Converts string to list of ints.

    Args:
        string: str, string to convert.
        sep: str, separator string.

    Returns:
        List of ints conversion of string.
    """
    if not string:
        return []
    return [int(x) for x in string.split(sep)]


def convert_string_to_list_of_floats(string, sep):
    """Converts string to list of floats.

    Args:
        string: str, string to convert.
        sep: str, separator string.

    Returns:
        List of floats conversion of string.
    """
    if not string:
        return []
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
    parser.add_argument(
        "--image_pool_capacity",
        help="The capacity of generated image pools. If zero, then no pools.",
        type=int,
        default=50
    )
    parser.add_argument(
        "--use_least_squares_loss",
        help="Whether to use least squares loss.",
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
    parser.add_argument(
        "--preprocess_image_resize_jitter_size",
        help="List of [height, width] to resize and crop to add jitter to image during training.",
        type=str,
        default="286,286"
    )
    parser.add_argument(
        "--preprocess_image_use_random_mirroring",
        help="Whether to randomly mirror image during training.",
        type=str,
        default="True"
    )

    # Generator parameters.
    parser.add_argument(
        "--generator_use_unet",
        help="Whether generator users U-net or resnet.",
        type=str,
        default="False"
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
        "--generator_learning_rate_decay_type",
        help="Decay type of learning rate. Constant, polynomial, exponential, piecewise polynomial.",
        type=str,
        default="piecewise_polynomial"
    )
    parser.add_argument(
        "--generator_learning_rate_constant_steps",
        help="Number of steps to keep learning rate constant.",
        type=int,
        default=100 * 1096
    )
    parser.add_argument(
        "--generator_learning_rate_decay_steps",
        help="Number of steps to keep decay learning rate.",
        type=int,
        default=100 * 1096
    )
    parser.add_argument(
        "--generator_learning_rate_end_learning_rate",
        help="Minimum learning rate to decay to.",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--generator_learning_rate_power",
        help="Polynomial decay power.",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--generator_learning_rate_cycle",
        help="Whether to cycle learning rate for polynomial decay types.",
        type=str,
        default="False"
    )
    parser.add_argument(
        "--generator_learning_rate_decay_rate",
        help="Learning rate decay rate.",
        type=float,
        default=0.99
    )
    parser.add_argument(
        "--generator_learning_rate_staircase",
        help="Whether to staircase learning rate for exponential decay types.",
        type=str,
        default="False"
    )
    parser.add_argument(
        "--generator_learning_rate_alpha",
        help="Minimum learning rate value as a fraction of learning rate for cosine decay types.",
        type=float,
        default=0.0
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
        "--generator_clip_gradients_by_value",
        help="Clipping to prevent gradient to exceed this value for generator.",
        type=str,
        default="None"
    )
    parser.add_argument(
        "--generator_clip_gradients_global_norm",
        help="Global clipping to prevent gradient norm to exceed this value for generator.",
        type=str,
        default="None"
    )
    parser.add_argument(
        "--generator_train_steps",
        help="Number of steps to train generator for per cycle.",
        type=int,
        default=1
    )
    parser.add_argument(
        "--forward_cycle_loss_lambda",
        help="Forward cycle loss lambda weight.",
        type=float,
        default=10.0
    )
    parser.add_argument(
        "--backward_cycle_loss_lambda",
        help="Backward cycle loss lambda weight.",
        type=float,
        default=10.0
    )
    parser.add_argument(
        "--identity_loss_lambda",
        help="Identity loss lambda weight.",
        type=float,
        default=0.5
    )

    # Generator U-net encoder parameters.
    parser.add_argument(
        "--generator_unet_encoder_num_filters",
        help="Number of filters for generator U-net encoder conv layers.",
        type=str,
        default="64,128,256,512,512,512,512,512"
    )
    parser.add_argument(
        "--generator_unet_encoder_kernel_sizes",
        help="Kernel sizes for generator U-net encoder conv layers.",
        type=str,
        default="4,4,4,4,4,4,4,4"
    )
    parser.add_argument(
        "--generator_unet_encoder_strides",
        help="Strides for generator U-net encoder conv layers.",
        type=str,
        default="2,2,2,2,2,2,2,2"
    )
    parser.add_argument(
        "--generator_unet_encoder_downsample",
        help="Whether generator U-net encoder layers use batch norm.",
        type=str,
        default="True,True,True,True,True,True,True,True"
    )
    parser.add_argument(
        "--generator_unet_encoder_pad_type",
        help="Generator U-net encoder padding type: constant, reflection, replication.",
        type=str,
        default="constant"
    )
    parser.add_argument(
        "--generator_unet_encoder_pad_constant",
        help="Generator U-net encoder constant padding constant.",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--generator_unet_encoder_layer_norm_before_act",
        help="Whether generator U-net encoder layers have normalization before activation.",
        type=str,
        default="True"
    )
    parser.add_argument(
        "--generator_unet_encoder_dropout_before_act",
        help="Whether generator U-net encoder layers have dropout before activation.",
        type=str,
        default="False"
    )
    parser.add_argument(
        "--generator_unet_encoder_dropout_before_norm",
        help="Whether generator U-net encoder layers have dropout before normalization.",
        type=str,
        default="False"
    )
    parser.add_argument(
        "--generator_unet_encoder_layer_norm_type",
        help="Generator U-net encoder layer normalization types.",
        type=str,
        default="None,instance,instance,instance,instance,instance,instance,None"
    )
    parser.add_argument(
        "--generator_unet_encoder_dropout_rates",
        help="Generator U-net encoder layer dropout rates.",
        type=str,
        default="0.,0.,0.,0.,0.,0.,0.,0."
    )
    parser.add_argument(
        "--generator_unet_encoder_activation",
        help="Whether generator encoder layers use leaky relu activations.",
        type=str,
        default="leaky_relu,leaky_relu,leaky_relu,leaky_relu,leaky_relu,leaky_relu,leaky_relu,relu"
    )
    parser.add_argument(
        "--generator_unet_encoder_leaky_relu_alpha",
        help="The amount of leakyness of generator encoder's leaky relus.",
        type=float,
        default=0.2
    )

    # Generator U-net decoder parameters.
    parser.add_argument(
        "--generator_unet_decoder_num_filters",
        help="Number of filters for generator U-net decoder conv layers.",
        type=str,
        default="512,512,512,512,256,128,64,3"
    )
    parser.add_argument(
        "--generator_unet_decoder_kernel_sizes",
        help="Kernel sizes for generator U-net decoder conv layers.",
        type=str,
        default="4,4,4,4,4,4,4,4"
    )
    parser.add_argument(
        "--generator_unet_decoder_strides",
        help="Strides for generator U-net decoder conv layers.",
        type=str,
        default="2,2,2,2,2,2,2,2"
    )
    parser.add_argument(
        "--generator_unet_decoder_downsample",
        help="Whether generator U-net decoder layers use batch norm.",
        type=str,
        default="False,False,False,False,False,False,False,True"
    )
    parser.add_argument(
        "--generator_unet_decoder_pad_type",
        help="Generator U-net decoder padding type: constant, reflection, replication.",
        type=str,
        default="constant"
    )
    parser.add_argument(
        "--generator_unet_decoder_pad_constant",
        help="Generator U-net decoder constant padding constant.",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--generator_unet_decoder_layer_norm_before_act",
        help="Whether generator U-net decoder layers have normalization before activation.",
        type=str,
        default="True"
    )
    parser.add_argument(
        "--generator_unet_decoder_dropout_before_act",
        help="Whether generator U-net decoder layers have dropout before activation.",
        type=str,
        default="False"
    )
    parser.add_argument(
        "--generator_unet_decoder_dropout_before_norm",
        help="Whether generator U-net decoder layers have dropout before normalization.",
        type=str,
        default="False"
    )
    parser.add_argument(
        "--generator_unet_decoder_layer_norm_type",
        help="Generator U-net decoder layer normalization types.",
        type=str,
        default="instance,instance,instance,instance,instance,instance,instance,None"
    )
    parser.add_argument(
        "--generator_unet_decoder_dropout_rates",
        help="Generator U-net decoder layer dropout rates.",
        type=str,
        default="0.5,0.5,0.5,0.,0.,0.,0.,0."
    )
    parser.add_argument(
        "--generator_unet_decoder_activation",
        help="Whether generator decoder layers use leaky relu activations.",
        type=str,
        default="relu,relu,relu,relu,relu,relu,relu,tanh"
    )
    parser.add_argument(
        "--generator_unet_decoder_leaky_relu_alpha",
        help="The amount of leakyness of generator decoder's leaky relus.",
        type=float,
        default=0.2
    )

    # Generator resnet encoder parameters.
    parser.add_argument(
        "--generator_resnet_enc_num_filters",
        help="Number of filters for generator resnet encoder conv layers.",
        type=str,
        default="64,128,256"
    )
    parser.add_argument(
        "--generator_resnet_enc_kernel_sizes",
        help="Kernel sizes for generator resnet encoder conv layers.",
        type=str,
        default="7,3,3"
    )
    parser.add_argument(
        "--generator_resnet_enc_strides",
        help="Strides for generator resnet encoder conv layers.",
        type=str,
        default="1,2,2"
    )
    parser.add_argument(
        "--generator_resnet_enc_downsample",
        help="Whether generator resnet encoder layers use batch norm.",
        type=str,
        default="True,True,True"
    )
    parser.add_argument(
        "--generator_resnet_enc_pad_type",
        help="Generator resnet encoder padding type: constant, reflection, replication.",
        type=str,
        default="reflection"
    )
    parser.add_argument(
        "--generator_resnet_enc_pad_constant",
        help="Generator resnet encoder constant padding constant.",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--generator_resnet_enc_layer_norm_before_act",
        help="Whether generator resnet encoder layers have normalization before activation.",
        type=str,
        default="True"
    )
    parser.add_argument(
        "--generator_resnet_enc_dropout_before_act",
        help="Whether generator resnet encoder layers have dropout before activation.",
        type=str,
        default="False"
    )
    parser.add_argument(
        "--generator_resnet_enc_dropout_before_norm",
        help="Whether generator resnet encoder layers have dropout before normalization.",
        type=str,
        default="False"
    )
    parser.add_argument(
        "--generator_resnet_enc_layer_norm_type",
        help="Generator resnet encoder layer normalization types.",
        type=str,
        default="instance,instance,instance"
    )
    parser.add_argument(
        "--generator_resnet_enc_dropout_rates",
        help="Generator resnet encoder layer dropout rates.",
        type=str,
        default="0.,0.,0."
    )
    parser.add_argument(
        "--generator_resnet_enc_activation",
        help="Whether generator encoder layers use leaky relu activations.",
        type=str,
        default="relu,relu,relu"
    )
    parser.add_argument(
        "--generator_resnet_enc_leaky_relu_alpha",
        help="The amount of leakyness of generator encoder's leaky relus.",
        type=float,
        default=0.2
    )

    # Generator resnet res block parameters.
    parser.add_argument(
        "--generator_num_resnet_blocks",
        help="Generator number of resnet residual blocks.",
        type=int,
        default=9
    )
    parser.add_argument(
        "--generator_resnet_res_num_filters",
        help="Number of filters for generator resnet residual block conv layers.",
        type=str,
        default="256,256"
    )
    parser.add_argument(
        "--generator_resnet_res_kernel_sizes",
        help="Kernel sizes for generator resnet residual block conv layers.",
        type=str,
        default="3,3"
    )
    parser.add_argument(
        "--generator_resnet_res_strides",
        help="Strides for generator resnet residual block conv layers.",
        type=str,
        default="1,1"
    )
    parser.add_argument(
        "--generator_resnet_res_downsample",
        help="Whether generator resnet residual block layers use batch norm.",
        type=str,
        default="True,True"
    )
    parser.add_argument(
        "--generator_resnet_res_pad_type",
        help="Generator resnet residual block padding type: constant, reflection, replication.",
        type=str,
        default="reflection"
    )
    parser.add_argument(
        "--generator_resnet_res_pad_constant",
        help="Generator resnet residual block constant padding constant.",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--generator_resnet_res_layer_norm_before_act",
        help="Whether generator resnet residual block layers have normalization before activation.",
        type=str,
        default="True"
    )
    parser.add_argument(
        "--generator_resnet_res_dropout_before_act",
        help="Whether generator resnet residual block layers have dropout before activation.",
        type=str,
        default="False"
    )
    parser.add_argument(
        "--generator_resnet_res_dropout_before_norm",
        help="Whether generator resnet residual block layers have dropout before normalization.",
        type=str,
        default="False"
    )
    parser.add_argument(
        "--generator_resnet_res_layer_norm_type",
        help="Generator resnet residual block layer normalization types.",
        type=str,
        default="instance,instance"
    )
    parser.add_argument(
        "--generator_resnet_res_dropout_rates",
        help="Generator resnet residual block layer dropout rates.",
        type=str,
        default="0.,0."
    )
    parser.add_argument(
        "--generator_resnet_res_activation",
        help="Whether generator residual block layers use leaky relu activations.",
        type=str,
        default="relu,none"
    )
    parser.add_argument(
        "--generator_resnet_res_leaky_relu_alpha",
        help="The amount of leakyness of generator residual block's leaky relus.",
        type=float,
        default=0.2
    )

    # Generator resnet decoder parameters.
    parser.add_argument(
        "--generator_resnet_dec_num_filters",
        help="Number of filters for generator resnet decoder conv layers.",
        type=str,
        default="128,64,3"
    )
    parser.add_argument(
        "--generator_resnet_dec_kernel_sizes",
        help="Kernel sizes for generator resnet decoder conv layers.",
        type=str,
        default="3,3,7"
    )
    parser.add_argument(
        "--generator_resnet_dec_strides",
        help="Strides for generator resnet decoder conv layers.",
        type=str,
        default="2,2,1"
    )
    parser.add_argument(
        "--generator_resnet_dec_downsample",
        help="Whether generator resnet decoder layers use batch norm.",
        type=str,
        default="False,False,True"
    )
    parser.add_argument(
        "--generator_resnet_dec_pad_type",
        help="Generator resnet decoder padding type: constant, reflection, replication.",
        type=str,
        default="reflection"
    )
    parser.add_argument(
        "--generator_resnet_dec_pad_constant",
        help="Generator resnet decoder constant padding constant.",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--generator_resnet_dec_layer_norm_before_act",
        help="Whether generator resnet decoder layers have normalization before activation.",
        type=str,
        default="True"
    )
    parser.add_argument(
        "--generator_resnet_dec_dropout_before_act",
        help="Whether generator resnet decoder layers have dropout before activation.",
        type=str,
        default="False"
    )
    parser.add_argument(
        "--generator_resnet_dec_dropout_before_norm",
        help="Whether generator resnet decoder layers have dropout before normalization.",
        type=str,
        default="False"
    )
    parser.add_argument(
        "--generator_resnet_dec_layer_norm_type",
        help="Generator resnet decoder layer normalization types.",
        type=str,
        default="instance,instance,instance"
    )
    parser.add_argument(
        "--generator_resnet_dec_dropout_rates",
        help="Generator resnet decoder layer dropout rates.",
        type=str,
        default="0.,0.,0."
    )
    parser.add_argument(
        "--generator_resnet_dec_activation",
        help="Whether generator decoder layers use leaky relu activations.",
        type=str,
        default="relu,relu,tanh"
    )
    parser.add_argument(
        "--generator_resnet_dec_leaky_relu_alpha",
        help="The amount of leakyness of generator decoder's leaky relus.",
        type=float,
        default=0.2
    )

    # Discriminator parameters.
    parser.add_argument(
        "--discriminator_num_filters",
        help="Number of filters for discriminator conv layers.",
        type=str,
        default="64,128,256,512,512,1"
    )
    parser.add_argument(
        "--discriminator_kernel_sizes",
        help="Kernel sizes for discriminator conv layers.",
        type=str,
        default="4,4,4,4,4,4"
    )
    parser.add_argument(
        "--discriminator_strides",
        help="Strides for discriminator conv layers.",
        type=str,
        default="2,2,2,2,1,1"
    )
    parser.add_argument(
        "--discriminator_downsample",
        help="Whether discriminator layers use batch norm.",
        type=str,
        default="True,True,True,True,True,True"
    )
    parser.add_argument(
        "--discriminator_pad_type",
        help="Discriminator padding type: constant, reflection, replication.",
        type=str,
        default="reflection"
    )
    parser.add_argument(
        "--discriminator_pad_constant",
        help="Discriminator constant padding constant.",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--discriminator_layer_norm_before_act",
        help="Whether discriminator layers have normalization before activation.",
        type=str,
        default="True"
    )
    parser.add_argument(
        "--discriminator_dropout_before_act",
        help="Whether discriminator layers have dropout before activation.",
        type=str,
        default="False"
    )
    parser.add_argument(
        "--discriminator_dropout_before_norm",
        help="Whether discriminator layers have dropout before normalization.",
        type=str,
        default="False"
    )
    parser.add_argument(
        "--discriminator_layer_norm_type",
        help="Discriminator layer normalization types.",
        type=str,
        default="None,instance,instance,instance,instance,None"
    )
    parser.add_argument(
        "--discriminator_dropout_rates",
        help="Discriminator layer dropout rates.",
        type=str,
        default="0.,0.,0.,0.,0.,0."
    )
    parser.add_argument(
        "--discriminator_activation",
        help="Whether discriminator layers use leaky relu activations.",
        type=str,
        default="leaky_relu,leaky_relu,leaky_relu,leaky_relu,leaky_relu,None"
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
        "--discriminator_learning_rate_decay_type",
        help="Decay type of learning rate. Constant, polynomial, exponential, piecewise polynomial.",
        type=str,
        default="piecewise_polynomial"
    )
    parser.add_argument(
        "--discriminator_learning_rate_constant_steps",
        help="Number of steps to keep learning rate constant.",
        type=int,
        default=100 * 1096
    )
    parser.add_argument(
        "--discriminator_learning_rate_decay_steps",
        help="Number of steps to keep decay learning rate.",
        type=int,
        default=100 * 1096
    )
    parser.add_argument(
        "--discriminator_learning_rate_end_learning_rate",
        help="Minimum learning rate to decay to.",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--discriminator_learning_rate_power",
        help="Polynomial decay power.",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--discriminator_learning_rate_cycle",
        help="Whether to cycle learning rate for polynomial decay types.",
        type=str,
        default="False"
    )
    parser.add_argument(
        "--discriminator_learning_rate_decay_rate",
        help="Learning rate decay rate.",
        type=float,
        default=0.99
    )
    parser.add_argument(
        "--discriminator_learning_rate_staircase",
        help="Whether to staircase learning rate for exponential decay types.",
        type=str,
        default="False"
    )
    parser.add_argument(
        "--discriminator_learning_rate_alpha",
        help="Minimum learning rate value as a fraction of learning rate for cosine decay types.",
        type=float,
        default=0.0
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
        "--discriminator_clip_gradients_by_value",
        help="Clipping to prevent gradient to exceed this value for discriminator.",
        type=str,
        default="None"
    )
    parser.add_argument(
        "--discriminator_clip_gradients_global_norm",
        help="Global clipping to prevent gradient norm to exceed this value for discriminator.",
        type=str,
        default="None"
    )
    parser.add_argument(
        "--discriminator_train_steps",
        help="Number of steps to train discriminator for per cycle.",
        type=int,
        default=1
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

    # Fix use_least_squares_loss.
    arguments["use_least_squares_loss"] = convert_string_to_bool(
        string=arguments["use_least_squares_loss"]
    )

    # Fix eval steps.
    arguments["eval_steps"] = convert_string_to_none_or_int(
        string=arguments["eval_steps"])

    # Fix preprocess_image_resize_jitter_size.
    arguments["preprocess_image_resize_jitter_size"] = convert_string_to_list_of_ints(
        string=arguments["preprocess_image_resize_jitter_size"], sep=","
    )

    # Fix preprocess_image_use_random_mirroring.
    arguments["preprocess_image_use_random_mirroring"] = convert_string_to_bool(
        string=arguments["preprocess_image_use_random_mirroring"]
    )

    # Fix generator_use_unet.
    arguments["generator_use_unet"] = convert_string_to_bool(
        string=arguments["generator_use_unet"]
    )

    # Fix learning_rate_cycle.
    arguments["generator_learning_rate_cycle"] = convert_string_to_bool(
        string=arguments["generator_learning_rate_cycle"]
    )

    arguments["discriminator_learning_rate_cycle"] = convert_string_to_bool(
        string=arguments["discriminator_learning_rate_cycle"]
    )

    # Fix learning_rate_staircase.
    arguments["generator_learning_rate_staircase"] = convert_string_to_bool(
        string=arguments["generator_learning_rate_staircase"]
    )

    arguments["discriminator_learning_rate_staircase"] = convert_string_to_bool(
        string=arguments["discriminator_learning_rate_staircase"]
    )

    # Fix num_filters.
    arguments["generator_unet_encoder_num_filters"] = convert_string_to_list_of_ints(
        string=arguments["generator_unet_encoder_num_filters"], sep=","
    )

    arguments["generator_unet_decoder_num_filters"] = convert_string_to_list_of_ints(
        string=arguments["generator_unet_decoder_num_filters"], sep=","
    )

    arguments["generator_resnet_enc_num_filters"] = convert_string_to_list_of_ints(
        string=arguments["generator_resnet_enc_num_filters"], sep=","
    )

    arguments["generator_resnet_res_num_filters"] = convert_string_to_list_of_ints(
        string=arguments["generator_resnet_res_num_filters"], sep=","
    )

    arguments["generator_resnet_dec_num_filters"] = convert_string_to_list_of_ints(
        string=arguments["generator_resnet_dec_num_filters"], sep=","
    )

    arguments["discriminator_num_filters"] = convert_string_to_list_of_ints(
        string=arguments["discriminator_num_filters"], sep=","
    )

    # Fix kernel_sizes.
    arguments["generator_unet_encoder_kernel_sizes"] = convert_string_to_list_of_ints(
        string=arguments["generator_unet_encoder_kernel_sizes"], sep=","
    )

    arguments["generator_unet_decoder_kernel_sizes"] = convert_string_to_list_of_ints(
        string=arguments["generator_unet_decoder_kernel_sizes"], sep=","
    )

    arguments["generator_resnet_enc_kernel_sizes"] = convert_string_to_list_of_ints(
        string=arguments["generator_resnet_enc_kernel_sizes"], sep=","
    )

    arguments["generator_resnet_res_kernel_sizes"] = convert_string_to_list_of_ints(
        string=arguments["generator_resnet_res_kernel_sizes"], sep=","
    )

    arguments["generator_resnet_dec_kernel_sizes"] = convert_string_to_list_of_ints(
        string=arguments["generator_resnet_dec_kernel_sizes"], sep=","
    )

    arguments["discriminator_kernel_sizes"] = convert_string_to_list_of_ints(
        string=arguments["discriminator_kernel_sizes"], sep=","
    )

    # Fix strides.
    arguments["generator_unet_encoder_strides"] = convert_string_to_list_of_ints(
        string=arguments["generator_unet_encoder_strides"], sep=","
    )

    arguments["generator_unet_decoder_strides"] = convert_string_to_list_of_ints(
        string=arguments["generator_unet_decoder_strides"], sep=","
    )

    arguments["generator_resnet_enc_strides"] = convert_string_to_list_of_ints(
        string=arguments["generator_resnet_enc_strides"], sep=","
    )

    arguments["generator_resnet_res_strides"] = convert_string_to_list_of_ints(
        string=arguments["generator_resnet_res_strides"], sep=","
    )

    arguments["generator_resnet_dec_strides"] = convert_string_to_list_of_ints(
        string=arguments["generator_resnet_dec_strides"], sep=","
    )

    arguments["discriminator_strides"] = convert_string_to_list_of_ints(
        string=arguments["discriminator_strides"], sep=","
    )

    # Fix downsample.
    arguments["generator_unet_encoder_downsample"] = convert_string_to_list_of_bools(
        string=arguments["generator_unet_encoder_downsample"], sep=","
    )

    arguments["generator_unet_decoder_downsample"] = convert_string_to_list_of_bools(
        string=arguments["generator_unet_decoder_downsample"], sep=","
    )

    arguments["generator_resnet_enc_downsample"] = convert_string_to_list_of_bools(
        string=arguments["generator_resnet_enc_downsample"], sep=","
    )

    arguments["generator_resnet_res_downsample"] = convert_string_to_list_of_bools(
        string=arguments["generator_resnet_res_downsample"], sep=","
    )

    arguments["generator_resnet_dec_downsample"] = convert_string_to_list_of_bools(
        string=arguments["generator_resnet_dec_downsample"], sep=","
    )

    arguments["discriminator_downsample"] = convert_string_to_list_of_bools(
        string=arguments["discriminator_downsample"], sep=","
    )

    # Fix layer_norm_before_act.
    arguments["generator_unet_encoder_layer_norm_before_act"] = convert_string_to_bool(
        string=arguments["generator_unet_encoder_layer_norm_before_act"]
    )

    arguments["generator_unet_decoder_layer_norm_before_act"] = convert_string_to_bool(
        string=arguments["generator_unet_decoder_layer_norm_before_act"]
    )

    arguments["generator_resnet_enc_layer_norm_before_act"] = convert_string_to_bool(
        string=arguments["generator_resnet_enc_layer_norm_before_act"]
    )

    arguments["generator_resnet_res_layer_norm_before_act"] = convert_string_to_bool(
        string=arguments["generator_resnet_res_layer_norm_before_act"]
    )

    arguments["generator_resnet_dec_layer_norm_before_act"] = convert_string_to_bool(
        string=arguments["generator_resnet_dec_layer_norm_before_act"]
    )

    arguments["discriminator_encoder_layer_norm_before_act"] = convert_string_to_bool(
        string=arguments["discriminator_layer_norm_before_act"]
    )

    # Fix dropout_before_act.
    arguments["generator_unet_encoder_dropout_before_act"] = convert_string_to_bool(
        string=arguments["generator_unet_encoder_dropout_before_act"]
    )

    arguments["generator_unet_decoder_dropout_before_act"] = convert_string_to_bool(
        string=arguments["generator_unet_decoder_dropout_before_act"]
    )

    arguments["generator_resnet_enc_dropout_before_act"] = convert_string_to_bool(
        string=arguments["generator_resnet_enc_dropout_before_act"]
    )

    arguments["generator_resnet_res_dropout_before_act"] = convert_string_to_bool(
        string=arguments["generator_resnet_res_dropout_before_act"]
    )

    arguments["generator_resnet_dec_dropout_before_act"] = convert_string_to_bool(
        string=arguments["generator_resnet_dec_dropout_before_act"]
    )

    arguments["discriminator_encoder_dropout_before_act"] = convert_string_to_bool(
        string=arguments["discriminator_dropout_before_act"]
    )

    # Fix dropout_before_norm.
    arguments["generator_unet_encoder_dropout_before_norm"] = convert_string_to_bool(
        string=arguments["generator_unet_encoder_dropout_before_norm"]
    )

    arguments["generator_unet_decoder_dropout_before_norm"] = convert_string_to_bool(
        string=arguments["generator_unet_decoder_dropout_before_norm"]
    )

    arguments["generator_resnet_enc_dropout_before_norm"] = convert_string_to_bool(
        string=arguments["generator_resnet_enc_dropout_before_norm"]
    )

    arguments["generator_resnet_res_dropout_before_norm"] = convert_string_to_bool(
        string=arguments["generator_resnet_res_dropout_before_norm"]
    )

    arguments["generator_resnet_dec_dropout_before_norm"] = convert_string_to_bool(
        string=arguments["generator_resnet_dec_dropout_before_norm"]
    )

    arguments["discriminator_encoder_dropout_before_norm"] = convert_string_to_bool(
        string=arguments["discriminator_dropout_before_norm"]
    )

    # Fix layer_norm_type.
    arguments["generator_unet_encoder_layer_norm_type"] = arguments["generator_unet_encoder_layer_norm_type"].split(",")

    arguments["generator_unet_decoder_layer_norm_type"] = arguments["generator_unet_decoder_layer_norm_type"].split(",")

    arguments["generator_resnet_enc_layer_norm_type"] = arguments["generator_resnet_enc_layer_norm_type"].split(",")

    arguments["generator_resnet_res_layer_norm_type"] = arguments["generator_resnet_res_layer_norm_type"].split(",")

    arguments["generator_resnet_dec_layer_norm_type"] = arguments["generator_resnet_dec_layer_norm_type"].split(",")

    arguments["discriminator_layer_norm_type"] = arguments["discriminator_layer_norm_type"].split(",")

    # Fix dropout_rates.
    arguments["generator_unet_encoder_dropout_rates"] = convert_string_to_list_of_floats(
        string=arguments["generator_unet_encoder_dropout_rates"], sep=","
    )

    arguments["generator_unet_decoder_dropout_rates"] = convert_string_to_list_of_floats(
        string=arguments["generator_unet_decoder_dropout_rates"], sep=","
    )

    arguments["generator_resnet_enc_dropout_rates"] = convert_string_to_list_of_floats(
        string=arguments["generator_resnet_enc_dropout_rates"], sep=","
    )

    arguments["generator_resnet_res_dropout_rates"] = convert_string_to_list_of_floats(
        string=arguments["generator_resnet_res_dropout_rates"], sep=","
    )

    arguments["generator_resnet_dec_dropout_rates"] = convert_string_to_list_of_floats(
        string=arguments["generator_resnet_dec_dropout_rates"], sep=","
    )

    arguments["discriminator_dropout_rates"] = convert_string_to_list_of_floats(
        string=arguments["discriminator_dropout_rates"], sep=","
    )

    # Fix activation.
    arguments["generator_unet_encoder_activation"] = arguments["generator_unet_encoder_activation"].split(",")

    arguments["generator_unet_decoder_activation"] = arguments["generator_unet_decoder_activation"].split(",")

    arguments["generator_resnet_enc_activation"] = arguments["generator_resnet_enc_activation"].split(",")

    arguments["generator_resnet_res_activation"] = arguments["generator_resnet_res_activation"].split(",")

    arguments["generator_resnet_dec_activation"] = arguments["generator_resnet_dec_activation"].split(",")

    arguments["discriminator_activation"] = arguments["discriminator_activation"].split(",")

    # Fix clip_gradients_by_value.
    arguments["generator_clip_gradients_by_value"] = convert_string_to_none_or_float(
        string=arguments["generator_clip_gradients_by_value"]
    )

    arguments["discriminator_clip_gradients_by_value"] = convert_string_to_none_or_float(
        string=arguments["discriminator_clip_gradients_by_value"]
    )

    # Fix clip_gradients_global_norm.
    arguments["generator_clip_gradients_global_norm"] = convert_string_to_none_or_float(
        string=arguments["generator_clip_gradients_global_norm"]
    )

    arguments["discriminator_clip_gradients_global_norm"] = convert_string_to_none_or_float(
        string=arguments["discriminator_clip_gradients_global_norm"]
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
