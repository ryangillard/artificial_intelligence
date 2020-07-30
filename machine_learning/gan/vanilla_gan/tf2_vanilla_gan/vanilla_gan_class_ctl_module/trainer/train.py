import tensorflow as tf


class Train(object):
    """Class that contains methods used for only training.
    """
    def __init__(self):
        pass

    def get_variables_and_gradients(self, loss, gradient_tape, scope):
        """Gets variables and gradients from model wrt. loss.

        Args:
            loss: tensor, shape of [].
            gradient_tape: instance of `GradientTape`.
            scope: str, the name of the network of interest.

        Returns:
            Lists of network's variables and gradients.
        """
        # Get trainable variables.
        variables = self.network_models[scope].trainable_variables

        # Get gradients from gradient tape.
        gradients = gradient_tape.gradient(
            target=loss, sources=variables
        )

        # Clip gradients.
        if self.params["{}_clip_gradients".format(scope)]:
            gradients, _ = tf.clip_by_global_norm(
                t_list=gradients,
                clip_norm=params["{}_clip_gradients".format(scope)],
                name="{}_clip_by_global_norm_gradients".format(scope)
            )

        # Add variable names back in for identification.
        gradients = [
            tf.identity(
                input=g,
                name="{}_{}_gradients".format(scope, v.name[:-2])
            )
            if tf.is_tensor(x=g) else g
            for g, v in zip(gradients, variables)
        ]

        return variables, gradients

    def create_variable_and_gradient_histogram_summaries(
        self, variables, gradients, scope
    ):
        """Creates variable and gradient histogram summaries.

        Args:
            variables: list, network's trainable variables.
            gradients: list, gradients of network's trainable variables wrt.
                loss.
            scope: str, the name of the network of interest.
        """
        if self.params["write_summaries"]:
            # Add summaries for TensorBoard.
            with self.summary_file_writer.as_default():
                with tf.summary.record_if(
                    condition=tf.equal(
                        x=tf.math.floormod(
                            x=self.global_step,
                            y=self.params["save_summary_steps"]
                        ), y=0
                    )
                ):
                    for v, g in zip(variables, gradients):
                        tf.summary.histogram(
                            name="{}_variables/{}".format(
                                scope, v.name[:-2]
                            ),
                            data=v,
                            step=self.global_step
                        )
                        if tf.is_tensor(x=g):
                            tf.summary.histogram(
                                name="{}_gradients/{}".format(
                                    scope, v.name[:-2]
                                ),
                                data=g,
                                step=self.global_step
                            )
                    self.summary_file_writer.flush()

    def get_select_loss_variables_and_gradients(self, real_images, scope):
        """Gets selected network's loss, variables, and gradients.

        Args:
            real_images: tensor, real images of shape
                [batch_size, height * width * depth].
            scope: str, the name of the network of interest.

        Returns:
            Selected network's loss, variables, and gradients.
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            # Get fake logits from generator.
            fake_logits, generator_loss = self.generator_loss_phase(
                mode="TRAIN", training=(scope == "generator")
            )

            # Get discriminator loss.
            _, discriminator_loss = self.discriminator_loss_phase(
                real_images, fake_logits, training=(scope == "discriminator")
            )

        # Create empty dicts to hold loss, variables, gradients.
        loss_dict = {}
        vars_dict = {}
        grads_dict = {}

        # Loop over generator and discriminator.
        for (loss, gradient_tape, scope_name) in zip(
            [generator_loss, discriminator_loss],
            [gen_tape, dis_tape],
            ["generator", "discriminator"]
        ):
            # Get variables and gradients from generator wrt. loss.
            variables, gradients = self.get_variables_and_gradients(
                loss, gradient_tape, scope_name
            )

            # Add loss, variables, and gradients to dictionaries.
            loss_dict[scope_name] = loss
            vars_dict[scope_name] = variables
            grads_dict[scope_name] = gradients

            # Create variable and gradient histogram summaries.
            self.create_variable_and_gradient_histogram_summaries(
                variables, gradients, scope_name
            )

        return loss_dict[scope], vars_dict[scope], grads_dict[scope]

    def train_network(self, variables, gradients, scope):
        """Trains network variables using gradients with optimizer.

        Args:
            variables: list, network's trainable variables.
            gradients: list, gradients of network's trainable variables wrt.
                loss.
            scope: str, the name of the network of interest.
        """
        # Zip together gradients and variables.
        grads_and_vars = zip(gradients, variables)

        # Applying gradients to variables using optimizer.
        self.optimizers[scope].apply_gradients(grads_and_vars=grads_and_vars)

    def train_discriminator(self, features):
        """Trains discriminator network.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Discriminator loss tensor.
        """
        # Extract real images from features dictionary.
        real_images = tf.reshape(
            tensor=features["image"],
            shape=[
                -1,
                self.params["height"] * self.params["width"] * self.params["depth"]
            ]
        )

        # Get gradients for training by running inputs through networks.
        loss, variables, gradients = (
            self.get_select_loss_variables_and_gradients(
                real_images, scope="discriminator"
            )
        )

        # Train discriminator network.
        self.train_network(variables, gradients, scope="discriminator")

        return loss

    def train_generator(self, features):
        """Trains generator network.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Generator loss tensor.
        """
        # Extract real images from features dictionary.
        real_images = tf.reshape(
            tensor=features["image"],
            shape=[
                -1,
                self.params["height"] * self.params["width"] * self.params["depth"]
            ]
        )

        # Get gradients for training by running inputs through networks.
        loss, variables, gradients = (
            self.get_select_loss_variables_and_gradients(
                real_images, scope="generator"
            )
        )

        # Train generator network.
        self.train_network(variables, gradients, scope="generator")

        return loss

    def distributed_eager_discriminator_train_step(self, features):
        """Perform one distributed, eager discriminator train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Reduced loss tensor for chosen network across replicas.
        """
        if self.params["tf_version"] > 2.1:
            run_function = self.strategy.run
        else:
            run_function = self.strategy.experimental_run_v2

        per_replica_losses = run_function(
            fn=self.train_discriminator, kwargs={"features": features}
        )

        return self.strategy.reduce(
            reduce_op=tf.distribute.ReduceOp.SUM,
            value=per_replica_losses,
            axis=None
        )

    def non_distributed_eager_discriminator_train_step(self, features):
        """Perform one non-distributed, eager discriminator train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Reduced loss tensor for chosen network across replicas.
        """
        return self.train_discriminator(features=features)

    @tf.function
    def distributed_graph_discriminator_train_step(self, features):
        """Perform one distributed, graph discriminator train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Reduced loss tensor for chosen network across replicas.
        """
        if self.params["tf_version"] > 2.1:
            run_function = self.strategy.run
        else:
            run_function = self.strategy.experimental_run_v2

        per_replica_losses = run_function(
            fn=self.train_discriminator, kwargs={"features": features}
        )

        return self.strategy.reduce(
            reduce_op=tf.distribute.ReduceOp.SUM,
            value=per_replica_losses,
            axis=None
        )

    @tf.function
    def non_distributed_graph_discriminator_train_step(self, features):
        """Perform one non-distributed, graph discriminator train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Reduced loss tensor for chosen network across replicas.
        """
        return self.train_discriminator(features=features)

    def distributed_eager_generator_train_step(self, features):
        """Perform one distributed, eager generator train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Reduced loss tensor for chosen network across replicas.
        """
        if self.params["tf_version"] > 2.1:
            run_function = self.strategy.run
        else:
            run_function = self.strategy.experimental_run_v2

        per_replica_losses = run_function(
            fn=self.train_generator, kwargs={"features": features}
        )

        return self.strategy.reduce(
            reduce_op=tf.distribute.ReduceOp.SUM,
            value=per_replica_losses,
            axis=None
        )

    def non_distributed_eager_generator_train_step(self, features):
        """Perform one non-distributed, eager generator train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Reduced loss tensor for chosen network across replicas.
        """
        return self.train_generator(features=features)

    @tf.function
    def distributed_graph_generator_train_step(self, features):
        """Perform one distributed, graph generator train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Reduced loss tensor for chosen network across replicas.
        """
        if self.params["tf_version"] > 2.1:
            run_function = self.strategy.run
        else:
            run_function = self.strategy.experimental_run_v2

        per_replica_losses = run_function(
            fn=self.train_generator, kwargs={"features": features}
        )

        return self.strategy.reduce(
            reduce_op=tf.distribute.ReduceOp.SUM,
            value=per_replica_losses,
            axis=None
        )

    @tf.function
    def non_distributed_graph_generator_train_step(self, features):
        """Perform one non-distributed, graph generator train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Reduced loss tensor for chosen network across replicas.
        """
        return self.train_generator(features=features)

    def log_step_loss(self, epoch, epoch_step, loss):
        """Logs step information and loss.

        Args:
            epoch: int, current iteration fully through the dataset.
            epoch_step: int, number of batches through epoch.
            loss: float, the loss of the model at the current step.
        """
        if self.global_step % self.params["log_step_count_steps"] == 0:
            print(
                "epoch = {}, global_step = {}, epoch_step = {}, loss = {}".format(
                    epoch, self.global_step, epoch_step, loss
                )
            )
