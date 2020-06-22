import tensorflow as tf

from .print_object import print_obj


class Network(object):
    """Network that could be for generator or discriminator.
    Fields:
        name: str, name of `Generator` or `Discriminator`.
        kernel_regularizer: `l1_l2_regularizer` object, regularizar for kernel
            variables.
        bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
            variables.
    """
    def __init__(self, kernel_regularizer, bias_regularizer, name):
        """Instantiates network.
        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.
            name: str, name of generator or discriminator.
        """
        # Set name of generator.
        self.name = name

        # Regularizer for kernel weights.
        self.kernel_regularizer = kernel_regularizer

        # Regularizer for bias weights.
        self.bias_regularizer = bias_regularizer

    def embed_labels(self, labels, params, scope):
        """Embeds labels from integer indices to float vectors.

        Args:
            labels: tensor, labels to condition on of shape
                [cur_batch_size, 1].
            params: dict, user passed parameters.
            scope: str, variable scope.

        Returns:
            Embedded labels tensor of shape
                [cur_batch_size, label_embedding_dimension].
        """
        func_name = "{}_embed_labels".format(scope)

        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            # Create trainable label embedding matrix.
            label_embedding_matrix = tf.get_variable(
                name="label_embedding_matrix",
                shape=[
                    params["num_classes"],
                    params["label_embedding_dimension"]
                ],
                dtype=tf.float32,
                initializer=None,
                regularizer=None,
                trainable=True
            )

            # Get embedding vectors for integer label index.
            label_embeddings = tf.nn.embedding_lookup(
                params=label_embedding_matrix,
                ids=labels,
                name="embedding_lookup"
            )

            # Flatten back into a rank 2 tensor.
            label_vectors = tf.reshape(
                tensor=label_embeddings,
                shape=[-1, params["label_embedding_dimension"]],
                name="label_vectors"
            )
            print_obj(func_name, "label_vectors", label_vectors)

        return label_vectors

    def use_labels(self, features, labels, params, scope):
        """Conditions features using label data.

        Args:
            features: tensor, features tensor, either Z for generator or X for
                discriminator.
            labels: tensor, labels to condition on of shape
                [cur_batch_size, 1].
            params: dict, user passed parameters.
            scope: str, variable scope.

        Returns:
            Feature tensor conditioned on labels.
        """
        func_name = "{}_use_labels".format(scope)

        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            if params["{}_embed_labels".format(scope)]:
                label_vectors = self.embed_labels(
                    labels=labels, params=params, scope=scope
                )
            else:
                label_vectors = tf.one_hot(
                    indices=tf.squeeze(input=labels, axis=-1),
                    depth=params["num_classes"],
                    axis=-1,
                    name="label_vectors_one_hot"
                )
            print_obj(func_name, "label_vectors", label_vectors)

            if params["{}_concatenate_labels".format(scope)]:
                if params["{}_dense_before_concatenate".format(scope)]:
                    label_vectors = tf.layers.dense(
                        inputs=label_vectors,
                        units=params["label_embedding_dimension"],
                        activation=None,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        name="labels_dense_concat"
                    )

                # Concatenate labels & features along feature map dimension.
                network = tf.concat(
                    values=[features, label_vectors], axis=-1,
                    name="concat_labels"
                )
                print_obj(func_name, "network", network)
            else:
                label_vectors = tf.layers.dense(
                    inputs=label_vectors,
                    units=params["latent_size"],
                    activation=None,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name="labels_dense_multiply"
                )
                print_obj(func_name, "label_vectors", label_vectors)

                # Element-wise multiply label vectors with latent vectors.
                network = tf.multiply(
                    x=features, y=label_vectors, name="multiply_labels"
                )
                print_obj(func_name, "network", network)

        return network
