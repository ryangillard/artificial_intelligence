import tensorflow as tf


class TrainStep(object):
    """Class that contains methods concerning train steps.
    """
    def __init__(self):
        """Instantiate instance of `TrainStep`.
        """
        pass

    def train_batch(self, features):
        """Trains model with a batch of feature data.

        Args:
            features: tensor, rank 2 tensor of feature data.

        Returns:
            Scalar loss tensor.
        """
        # Pass images through ResNet to get feature vectors.
        resnet_feature_vectors = (
            self.resnet_instance.get_image_resnet_feature_vectors(
                images=features
            )
        )

        # Train PCA model.
        self.pca_model.calculate_data_stats(data=resnet_feature_vectors)

        return tf.zeros(shape=(), dtype=tf.float32)

    def distributed_eager_train_step(self, features):
        """Perform one distributed, eager train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Scalar loss of model.
        """
        if self.params["tf_version"] > 2.1:
            run_function = self.strategy.run
        else:
            run_function = self.strategy.experimental_run_v2

        per_replica_losses = run_function(
            fn=self.train_batch,
            kwargs={"features": features}
        )

        return self.strategy.reduce(
            reduce_op=tf.distribute.ReduceOp.SUM,
            value=per_replica_losses,
            axis=None
        )

    def non_distributed_eager_train_step(self, features):
        """Perform one non-distributed, eager train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Scalar loss of model.
        """
        return self.train_batch(features=features)

    @tf.function
    def distributed_graph_train_step(self, features):
        """Perform one distributed, graph train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Scalar loss of model.
        """
        if self.params["tf_version"] > 2.1:
            run_function = self.strategy.run
        else:
            run_function = self.strategy.experimental_run_v2

        per_replica_losses = run_function(
            fn=self.train_batch,
            kwargs={"features": features}
        )

        return self.strategy.reduce(
            reduce_op=tf.distribute.ReduceOp.SUM,
            value=per_replica_losses,
            axis=None
        )

    @tf.function
    def non_distributed_graph_train_step(self, features):
        """Perform one non-distributed, graph train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Scalar loss of model.
        """
        return self.train_batch(features=features)

    def get_train_step_functions(self):
        """Gets model train step functions for strategy and mode.
        """
        if self.strategy:
            if self.params["use_graph_mode"]:
                self.train_step_fn = (
                    self.distributed_graph_train_step
                )
            else:
                self.train_step_fn = (
                    self.distributed_eager_train_step
                )
        else:
            if self.params["use_graph_mode"]:
                self.train_step_fn = (
                    self.non_distributed_graph_train_step
                )
            else:
                self.train_step_fn = (
                    self.non_distributed_eager_train_step
                )

    @tf.function
    def increment_global_step_var(self):
        """Increments global step variable.
        """
        self.global_step.assign_add(
            delta=tf.ones(shape=(), dtype=tf.int64)
        )

    def perform_training_step(self, train_dataset_iterator, train_step_fn):
        """Performs one training step of model.

        Args:
            train_dataset_iterator: iterator, iterator of instance of
                `Dataset` for training data.
            train_step_fn: unbound function, trains the given model
                with a given set of features.
        """
        # Train model on batch of features and get loss.
        features = next(train_dataset_iterator)

        # Train for a step and get loss.
        self.loss = train_step_fn(features=features)

        # Checkpoint model every save_checkpoints_steps steps.
        checkpoint_saved = self.checkpoint_manager.save(
            checkpoint_number=self.global_step, check_interval=True
        )

        if checkpoint_saved:
            print("Checkpoint saved at {}".format(checkpoint_saved))

        # Increment steps.
        self.increment_global_step_var()
