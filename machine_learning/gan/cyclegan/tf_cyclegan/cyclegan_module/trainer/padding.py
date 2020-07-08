import math
import tensorflow as tf

from .print_object import print_obj


class CustomPadding2D(tf.layers.Layer):
    """Custom layer for 2D padding.

    Fields:
        paddings: tensor, rank 2 tensor that stores paddings.
    """
    def __init__(self, conv_inputs, kernel_size, strides, padding, **kwargs):
        """Instantiates custom 2D padding layer.

        Args:
            conv_inputs: tensor, inputs to previous conv layer.
            kernel_size: int, list, or tuple, if int then the kernel size for
                both height and width.
            strides: int, list, or tuple, if int then the stride for both
                height and width.
            padding: str, either same or valid padding.
        """
        # Get kernel sizes.
        if isinstance(kernel_size, (list, tuple)):
            kernel_size = kernel_size
        elif isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        # Get strides.
        if isinstance(strides, (list, tuple)):
            strides = strides
        elif isinstance(strides, int):
            strides = (strides, strides)

        # Get padding type.
        padding = padding.lower()
        assert padding in {"valid", "same"}

        # Create paddings.
        self.paddings = self._create_paddings(
            conv_inputs, kernel_size, strides, padding
        )

        # Pass anything else to base `Layer` class.
        super().__init__(**kwargs)

    def _create_paddings(self, conv_inputs, kernel_size, strides, padding):
        """Creates rank 2 paddings tensor.

        Args:
            conv_inputs: tensor, inputs to previous conv layer.
            kernel_size: int, list, or tuple, if int then the kernel size for
                both height and width.
            strides: int, list, or tuple, if int then the stride for both
                height and width.
            padding: str, either same or valid padding.
        """
        # Get input shape.
        input_shape = [x.value for x in conv_inputs.shape]
        print_obj("calculate_padding", "input_shape", input_shape)
        input_h, input_w = input_shape[1:3]

        # Expand kernel size into height and width.
        kernel_h, kernel_w = kernel_size

        # Expand strides into height and width.
        stride_h, stride_w = strides

        # Only pad if padding is same, otherwise don't if padding is valid.
        if padding == "same":
            # Calculate output shape.
            output_h = int(math.ceil(float(input_h) / float(stride_h)))
            output_w = int(math.ceil(float(input_w) / float(stride_w)))

            # Find out how much is missing in each dimension.
            pad_along_height = max(
                (output_h - 1) * stride_h + kernel_h - input_h, 0
            )
            pad_along_width = max(
                (output_w - 1) * stride_w + kernel_w - input_w, 0
            )

            # Preference to add any extra to bottom and right.
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left
        elif padding == "valid":
            pad_top = 0
            pad_bottom = 0
            pad_left = 0
            pad_right = 0

        # Create paddings.
        return [
            [0, 0],
            [pad_top, pad_bottom],
            [pad_left, pad_right],
            [0, 0]
        ]

    def compute_output_shape(self, input_shape):
        """Returns padded output shape given input shape.

        Args:
            input_shape: tuple, 4-tuple that cotains dimension sizes for rank
                4 input shape.
        Returns:
            4-tuple of computed padded output shape.
        """
        return (
            input_shape[0],
            self.paddings[1][0] + input_shape[1] + self.paddings[1][1],
            self.paddings[2][0] + input_shape[2] + self.paddings[2][1],
            input_shape[3]
        )


class ConstantPadding2D(CustomPadding2D):
    """Custom layer for 2D constant padding.

    Fields:
        constant: float, constant value to pad tensor.
    """
    def __init__(
        self,
        conv_inputs,
        kernel_size,
        strides,
        padding,
        constant=0.0,
        **kwargs
    ):
        """Instantiates replication 2D padding layer.

        Args:
            conv_inputs: tensor, inputs to previous conv layer.
            kernel_size: int, list, or tuple, if int then the kernel size for
                both height and width.
            strides: int, list, or tuple, if int then the stride for both
                height and width.
            padding: str, either same or valid padding.
            constant: float, constant value to pad tensor.
        """
        self.constant = constant
        super().__init__(conv_inputs, kernel_size, strides, padding, **kwargs)

    def call(self, inputs):
        """Returns constant padded input tensor.

        Args:
            conv_inputs: tensor, outputs of previous conv layer with no
                padding.
        Returns:
            Constant padded input tensor.
        """
        return tf.pad(
            tensor=inputs,
            paddings=self.paddings,
            mode="CONSTANT",
            constant_values=self.constant
        )


class ReflectionPadding2D(CustomPadding2D):
    """Custom layer for 2D reflection padding.
    """
    def __init__(self, conv_inputs, kernel_size, strides, padding, **kwargs):
        """Instantiates replication 2D padding layer.

        Args:
            conv_inputs: tensor, inputs to previous conv layer.
            kernel_size: int, list, or tuple, if int then the kernel size for
                both height and width.
            strides: int, list, or tuple, if int then the stride for both
                height and width.
            padding: str, either same or valid padding.
        """
        super().__init__(conv_inputs, kernel_size, strides, padding, **kwargs)

    def call(self, inputs):
        """Returns reflection padded input tensor.

        Args:
            conv_inputs: tensor, outputs of previous conv layer with no
                padding.
        Returns:
            Reflection padded input tensor.
        """
        return tf.pad(tensor=inputs, paddings=self.paddings, mode="REFLECT")


class ReplicationPadding2D(CustomPadding2D):
    """Custom layer for 2D replication padding.
    """
    def __init__(self, conv_inputs, kernel_size, strides, padding, **kwargs):
        """Instantiates replication 2D padding layer.

        Args:
            conv_inputs: tensor, inputs to previous conv layer.
            kernel_size: int, list, or tuple, if int then the kernel size for
                both height and width.
            strides: int, list, or tuple, if int then the stride for both
                height and width.
            padding: str, either same or valid padding.
        """
        super().__init__(conv_inputs, kernel_size, strides, padding, **kwargs)

    def call(self, inputs):
        """Returns replication padded input tensor.

        Args:
            conv_inputs: tensor, outputs of previous conv layer with no
                padding.
        Returns:
            Replication padded input tensor.
        """
        return tf.pad(tensor=inputs, paddings=self.paddings, mode="SYMMETRIC")
