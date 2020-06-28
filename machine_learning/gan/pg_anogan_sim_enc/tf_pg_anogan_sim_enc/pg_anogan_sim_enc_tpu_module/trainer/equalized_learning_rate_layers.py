import tensorflow as tf


class Dense(tf.layers.Dense):
    """Subclassing `Dense` layer to allow equalized learning rate scaling.

    Fields:
        equalized_learning_rate: bool, if want to scale layer weights to
            equalize learning rate each forward pass.
    """
    def __init__(
            self,
            units,
            activation=None,
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            equalized_learning_rate=False,
            name=None,
            **kwargs):
        """Initializes `Dense` layer.

        Args:
            units: Integer or Long, dimensionality of the output space.
            activation: Activation function (callable). Set it to None to maintain a
              linear activation.
            use_bias: Boolean, whether the layer uses a bias.
            kernel_initializer: Initializer function for the weight matrix.
              If `None` (default), weights are initialized using the default
              initializer used by `tf.compat.v1.get_variable`.
            bias_initializer: Initializer function for the bias.
            kernel_regularizer: Regularizer function for the weight matrix.
            bias_regularizer: Regularizer function for the bias.
            activity_regularizer: Regularizer function for the output.
            kernel_constraint: An optional projection function to be applied to the
                kernel after being updated by an `Optimizer` (e.g. used to implement
                norm constraints or value constraints for layer weights). The function
                must take as input the unprojected variable and must return the
                projected variable (which must have the same shape). Constraints are
                not safe to use when doing asynchronous distributed training.
            bias_constraint: An optional projection function to be applied to the
                bias after being updated by an `Optimizer`.
            trainable: Boolean, if `True` also add variables to the graph collection
              `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
            equalized_learning_rate: bool, if want to scale layer weights to
                equalize learning rate each forward pass.
            name: String, the name of the layer. Layers with the same name will
              share weights, but to avoid mistakes we require reuse=True in such cases.
        """
        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,
            **kwargs
        )

        # Whether we will scale weights using He init every forward pass.
        self.equalized_learning_rate = equalized_learning_rate

    def call(self, inputs):
        """Calls layer and returns outputs.

        Args:
            inputs: tensor, input tensor of shape [batch_size, features].
        """
        if self.equalized_learning_rate:
            # Scale kernel weights by He init fade-in constant.
            kernel_shape = self.kernel.shape
            fade_in = kernel_shape[0]
            he_constant = tf.sqrt(x=2. / fade_in)
            kernel = self.kernel * he_constant
        else:
            kernel = self.kernel

        rank = len(inputs.shape)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = tf.tensordot(
                a=inputs, b=kernel, axes=[[rank - 1], [0]]
            )
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(shape=output_shape)
        else:
            inputs = tf.cast(x=inputs, dtype=self._compute_dtype)
            if isinstance(inputs, tf.SparseTensor):
                outputs = tf.sparse_tensor_dense_matmul(sp_a=inputs, b=kernel)
            else:
                outputs = tf.matmul(a=inputs, b=kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(value=outputs, bias=self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs


class Conv2D(tf.layers.Conv2D):
    """Subclassing `Conv2D` layer to allow equalized learning rate scaling.

    Fields:
        equalized_learning_rate: bool, if want to scale layer weights to
            equalize learning rate each forward pass.
    """
    def __init__(
            self,
            filters,
            kernel_size,
            strides=(1, 1),
            padding="valid",
            data_format="channels_last",
            dilation_rate=(1, 1),
            activation=None,
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            equalized_learning_rate=False,
            name=None,
            **kwargs):
        """Initializes `Conv2D` layer.

        Args:
            filters: Integer, the dimensionality of the output space (i.e. the number
              of filters in the convolution).
            kernel_size: An integer or tuple/list of 2 integers, specifying the
              height and width of the 2D convolution window.
              Can be a single integer to specify the same value for
              all spatial dimensions.
            strides: An integer or tuple/list of 2 integers,
              specifying the strides of the convolution along the height and width.
              Can be a single integer to specify the same value for
              all spatial dimensions.
              Specifying any stride value != 1 is incompatible with specifying
              any `dilation_rate` value != 1.
            padding: One of `"valid"` or `"same"` (case-insensitive).
            data_format: A string, one of `channels_last` (default) or `channels_first`.
              The ordering of the dimensions in the inputs.
              `channels_last` corresponds to inputs with shape
              `(batch, height, width, channels)` while `channels_first` corresponds to
              inputs with shape `(batch, channels, height, width)`.
            dilation_rate: An integer or tuple/list of 2 integers, specifying
              the dilation rate to use for dilated convolution.
              Can be a single integer to specify the same value for
              all spatial dimensions.
              Currently, specifying any `dilation_rate` value != 1 is
              incompatible with specifying any stride value != 1.
            activation: Activation function. Set it to None to maintain a
              linear activation.
            use_bias: Boolean, whether the layer uses a bias.
            kernel_initializer: An initializer for the convolution kernel.
            bias_initializer: An initializer for the bias vector. If None, the default
              initializer will be used.
            kernel_regularizer: Optional regularizer for the convolution kernel.
            bias_regularizer: Optional regularizer for the bias vector.
            activity_regularizer: Optional regularizer function for the output.
            kernel_constraint: Optional projection function to be applied to the
                kernel after being updated by an `Optimizer` (e.g. used to implement
                norm constraints or value constraints for layer weights). The function
                must take as input the unprojected variable and must return the
                projected variable (which must have the same shape). Constraints are
                not safe to use when doing asynchronous distributed training.
            bias_constraint: Optional projection function to be applied to the
                bias after being updated by an `Optimizer`.
            trainable: Boolean, if `True` also add variables to the graph collection
              `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
            equalized_learning_rate: bool, if want to scale layer weights to
                equalize learning rate each forward pass.
            name: A string, the name of the layer.
        """
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,
            **kwargs
        )

        # Whether we will scale weights using He init every forward pass.
        self.equalized_learning_rate = equalized_learning_rate

    def call(self, inputs):
        """Calls layer and returns outputs.

        Args:
            inputs: tensor, input tensor of shape
                [batch_size, height, width, channels].
        """
        if self.equalized_learning_rate:
            # Scale kernel weights by He init constant.
            kernel_shape = self.kernel.shape
            fade_in = kernel_shape[0] * kernel_shape[1] * kernel_shape[2]
            he_constant = tf.sqrt(x=2. / fade_in)
            kernel = self.kernel * he_constant
        else:
            kernel = self.kernel

        outputs = self._convolution_op(inputs, kernel)

        if self.use_bias:
            if self.data_format == "channels_first":
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = tf.reshape(
                        tensor=self.bias, shape=(1, self.filters, 1)
                    )
                    outputs += bias
                else:
                    outputs = tf.nn.bias_add(
                        value=outputs, bias=self.bias, data_format="NCHW"
                    )
            else:
                outputs = tf.nn.bias_add(
                    value=outputs, bias=self.bias, data_format="NHWC"
                )

        if self.activation is not None:
            return self.activation(outputs)
        return outputs
