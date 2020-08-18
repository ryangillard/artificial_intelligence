import tensorflow as tf


class TrainStep(object):
    """Class that contains methods concerning train steps.
    """
    def __init__(self):
        """Instantiate instance of `TrainStep`.
        """
        pass

    def distributed_eager_discriminator_train_step(
        self, features, growth_idx
    ):
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
            fn=self.train_discriminator,
            kwargs={"features": features, "growth_idx": growth_idx}
        )

        return self.strategy.reduce(
            reduce_op=tf.distribute.ReduceOp.SUM,
            value=per_replica_losses,
            axis=None
        )

    def non_distributed_eager_discriminator_train_step(
        self, features, growth_idx
    ):
        """Perform one non-distributed, eager discriminator train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Reduced loss tensor for chosen network across replicas.
        """
        return self.train_discriminator(
            features=features, growth_idx=growth_idx
        )

    @tf.function
    def distributed_graph_discriminator_train_step(
        self, features, growth_idx
    ):
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
            fn=self.train_discriminator,
            kwargs={"features": features, "growth_idx": growth_idx}
        )

        return self.strategy.reduce(
            reduce_op=tf.distribute.ReduceOp.SUM,
            value=per_replica_losses,
            axis=None
        )

    @tf.function
    def non_distributed_graph_discriminator_train_step(
        self, features, growth_idx
    ):
        """Perform one non-distributed, graph discriminator train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Reduced loss tensor for chosen network across replicas.
        """
        return self.train_discriminator(
            features=features, growth_idx=growth_idx
        )

    def distributed_eager_generator_train_step(self, features, growth_idx):
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
            fn=self.train_generator,
            kwargs={"features": features, "growth_idx": growth_idx}
        )

        return self.strategy.reduce(
            reduce_op=tf.distribute.ReduceOp.SUM,
            value=per_replica_losses,
            axis=None
        )

    def non_distributed_eager_generator_train_step(
        self, features, growth_idx
    ):
        """Perform one non-distributed, eager generator train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Reduced loss tensor for chosen network across replicas.
        """
        return self.train_generator(features=features, growth_idx=growth_idx)

    @tf.function
    def distributed_graph_generator_train_step(self, features, growth_idx):
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
            fn=self.train_generator,
            kwargs={"features": features, "growth_idx": growth_idx}
        )

        return self.strategy.reduce(
            reduce_op=tf.distribute.ReduceOp.SUM,
            value=per_replica_losses,
            axis=None
        )

    @tf.function
    def non_distributed_graph_generator_train_step(
        self, features, growth_idx
    ):
        """Perform one non-distributed, graph generator train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Reduced loss tensor for chosen network across replicas.
        """
        return self.train_generator(features=features, growth_idx=growth_idx)

    def get_train_step_functions(self):
        """Gets network model train step functions for strategy and mode.
        """
        if self.strategy:
            if self.params["use_graph_mode"]:
                self.discriminator_train_step_fn = (
                    self.distributed_graph_discriminator_train_step
                )
                self.generator_train_step_fn = (
                    self.distributed_graph_generator_train_step
                )
            else:
                self.discriminator_train_step_fn = (
                    self.distributed_eager_discriminator_train_step
                )
                self.generator_train_step_fn = (
                    self.distributed_eager_generator_train_step
                )
        else:
            if self.params["use_graph_mode"]:
                self.discriminator_train_step_fn = (
                    self.non_distributed_graph_discriminator_train_step
                )
                self.generator_train_step_fn = (
                    self.non_distributed_graph_generator_train_step
                )
            else:
                self.discriminator_train_step_fn = (
                    self.non_distributed_eager_discriminator_train_step
                )
                self.generator_train_step_fn = (
                    self.non_distributed_eager_generator_train_step
                )

    def log_step_loss(self, epoch, loss):
        """Logs step information and loss.

        Args:
            epoch: int, current iteration fully through the dataset.
            loss: float, the loss of the model at the current step.
        """
        if self.global_step % self.params["log_step_count_steps"] == 0:
            start_time = self.previous_timestamp
            self.previous_timestamp = tf.timestamp()
            elapsed_time = self.previous_timestamp - start_time
            print(
                "{} = {}, {} = {}, {} = {}, {} = {}, {} = {}".format(
                    "epoch",
                    epoch,
                    "global_step",
                    self.global_step.numpy(),
                    "epoch_step",
                    self.epoch_step,
                    "steps/sec",
                    float(self.params["log_step_count_steps"]) / elapsed_time,
                    "loss",
                    loss,
                )
            )

    @tf.function
    def increment_global_step(self):
        """Increments global step variable.
        """
        self.global_step.assign_add(
            delta=tf.ones(shape=[], dtype=tf.int64)
        )

    @tf.function
    def increment_alpha_var(self):
        """Increments alpha variable through range [0., 1.] during transition.
        """
        block_idx = self.block_idx
        num_steps_until_growth = (
            self.num_steps_until_growth_schedule[block_idx]
        )
        self.alpha_var.assign(
            value=tf.divide(
                x=tf.cast(
                    x=tf.math.floormod(
                        x=self.global_step, y=num_steps_until_growth
                    ),
                    dtype=tf.float32
                ),
                y=num_steps_until_growth
            )
        )

    def network_model_training_steps(
        self,
        epoch,
        train_step_fn,
        train_steps,
        train_dataset_iter,
        features,
        labels
    ):
        """Trains a network model for so many steps given a set of features.

        Args:
            epoch: int, the current iteration through the dataset.
            train_step_fn: unbound function, trains the given network model
                given a set of features.
            train_steps: int, number of steps to train network model.
            train_dataset_iter: iterator, training dataset iterator.
            features: dict, feature tensors from input function.
            labels: tensor, label tensor from input function.

        Returns:
            Bool that indicates if current growth phase complete,
                dictionary of most recent feature tensors, and most recent
                label tensor.
        """
        for _ in range(train_steps):
            if features is None:
                # Train model on batch of features and get loss.
                features, labels = next(train_dataset_iter)

            loss = train_step_fn(features=features, growth_idx=self.growth_idx)

            # Log step information and loss.
            self.log_step_loss(epoch, loss)

            # Checkpoint model every save_checkpoints_steps steps.
            self.checkpoint_manager.save(
                checkpoint_number=self.global_step, check_interval=True
            )

            # Increment steps.
            self.increment_global_step()
            self.growth_step += 1
            self.epoch_step += 1

            # If this is a growth transition phase.
            if self.growth_idx % 2 == 1:
                # Increment alpha variable.
                self.increment_alpha_var()

            num_steps_until_growth = (
                self.num_steps_until_growth_schedule[self.block_idx]
            )

            if self.growth_step % num_steps_until_growth == 0:
                return True, features, labels
        return False, features, labels
