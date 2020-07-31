import datetime
import os
import tensorflow as tf

from . import inputs
from . import instantiate_model
from . import train
from . import train_and_eval


class TrainAndEvaluateLoop(
    train_and_eval.TrainAndEval,
    train.Train,
    instantiate_model.InstantiateModel
):
    """Train and evaluate loop trainer.

    Fields:
        params: dict, user passed parameters.
        network_objects: dict, instances of `Generator` and `Discriminator`
            network objects.
        network_models: dict, instances of Keras `Model`s for each network.
        optimizers: dict, instances of Keras `Optimizer`s for each network.
        strategy: instance of tf.distribute.strategy.
        global_batch_size: int, the global batch size after summing batch
            sizes across replicas.
        global_step: tf.Variable, the global step counter across epochs and
            steps within epoch.
        checkpoint_manager: instance of `tf.train.CheckpointManager`.
        summary_file_writer: instance of tf.summary.create_file_writer for
            summaries for TensorBoard.
    """
    def __init__(self, params):
        """Instantiate trainer.

        Args:
            params: dict, user passed parameters.
        """
        self.params = params

        self.network_objects = {}
        self.network_models = {}
        self.optimizers = {}

        self.strategy = None

        self.global_batch_size = None
        self.global_step = tf.Variable(
            initial_value=tf.zeros(shape=[], dtype=tf.int64),
            trainable=False,
            name="global_step"
        )
        self.checkpoint_manager = None
        self.summary_file_writer = None

    @tf.function
    def increment_global_step(self):
        self.global_step.assign_add(
            delta=tf.ones(shape=[], dtype=tf.int64)
        )

    def get_train_eval_datasets(self, num_replicas):
        """Gets train and eval datasets.

        Args:
            num_replicas: int, number of device replicas.

        Returns:
            Train and eval datasets.
        """
        train_dataset = inputs.read_dataset(
            filename=self.params["train_file_pattern"],
            batch_size=self.params["train_batch_size"] * num_replicas,
            params=self.params,
            training=True
        )()

        eval_dataset = inputs.read_dataset(
            filename=self.params["eval_file_pattern"],
            batch_size=self.params["eval_batch_size"] * num_replicas,
            params=self.params,
            training=False
        )()
        if self.params["eval_steps"]:
            eval_dataset = eval_dataset.take(count=self.params["eval_steps"])

        return train_dataset, eval_dataset

    def create_checkpoint_machinery(self):
        """Creates checkpoint machinery needed to save & restore checkpoints.
        """
        # Create checkpoint instance.
        checkpoint_dir = os.path.join(
            self.params["output_dir"], "checkpoints"
        )
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(
            generator_model=self.network_models["generator"],
            discriminator_model=self.network_models["discriminator"],
            generator_optimizer=self.optimizers["generator"],
            discriminator_optimizer=self.optimizers["discriminator"]
        )

        # Create checkpoint manager.
        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=checkpoint,
            directory=checkpoint_dir,
            max_to_keep=self.params["keep_checkpoint_max"],
            step_counter=self.global_step,
            checkpoint_interval=self.params["save_checkpoints_steps"]
        )

        # Restore any prior checkpoints.
        status = checkpoint.restore(
            save_path=self.checkpoint_manager.latest_checkpoint
        )

    def training_loop(self, steps_per_epoch, train_dataset_iter):
        """Logs step information and loss.

        Args:
            steps_per_epoch: int, number of steps/batches to take each epoch.
            train_dataset_iter: iterator, training dataset iterator.
        """
        # Get correct train function based on parameters.
        if self.strategy:
            if self.params["use_graph_mode"]:
                discriminator_train_step_fn = (
                    self.distributed_graph_discriminator_train_step
                )
                generator_train_step_fn = (
                    self.distributed_graph_generator_train_step
                )
            else:
                discriminator_train_step_fn = (
                    self.distributed_eager_discriminator_train_step
                )
                generator_train_step_fn = (
                    self.distributed_eager_generator_train_step
                )
        else:
            if self.params["use_graph_mode"]:
                discriminator_train_step_fn = (
                    self.non_distributed_graph_discriminator_train_step
                )
                generator_train_step_fn = (
                    self.non_distributed_graph_generator_train_step
                )
            else:
                discriminator_train_step_fn = (
                    self.non_distributed_eager_discriminator_train_step
                )
                generator_train_step_fn = (
                    self.non_distributed_eager_generator_train_step
                )

        for epoch in range(self.params["num_epochs"]):
            for epoch_step in range(steps_per_epoch):
                # Train model on batch of features and get loss.
                features, labels = next(train_dataset_iter)

                # Determine if it is time to train generator or discriminator.
                cycle_step = self.global_step % (
                    self.params["discriminator_train_steps"] + self.params["generator_train_steps"]
                )

                # Conditionally choose to train generator or discriminator subgraph.
                if cycle_step < self.params["discriminator_train_steps"]:
                    loss = discriminator_train_step_fn(features=features)
                else:
                    loss = generator_train_step_fn(features=features)

                # Log step information and loss.
                self.log_step_loss(epoch, epoch_step, loss)

                # Checkpoint model every save_checkpoints_steps steps.
                self.checkpoint_manager.save(
                    checkpoint_number=self.global_step, check_interval=True
                )

                # Increment global step.
                self.increment_global_step()

    def training_loop_end_save_model(self):
        """Saving model when training loop ends.
        """
        # Write final checkpoint.
        self.checkpoint_manager.save(
            checkpoint_number=self.global_step, check_interval=False
        )

        # Export SavedModel for serving.
        export_path = os.path.join(
            self.params["output_dir"],
            "export",
            datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        )

        # Signature will be serving_default.
        tf.saved_model.save(
            obj=self.network_models["generator"], export_dir=export_path
        )

    def train_block(self, train_dataset, eval_dataset):
        """Training block setups training, then loops through datasets.

        Args:
            train_dataset: instance of `Dataset` for training data.
            eval_dataset: instance of `Dataset` for evaluation data.
        """
        # Create iterators of datasets.
        train_dataset_iter = iter(train_dataset)
        eval_dataset_iter = iter(eval_dataset)

        steps_per_epoch = (
            self.params["train_dataset_length"] // self.global_batch_size
        )

        # Instantiate model objects.
        self.instantiate_model_objects()

        # Create checkpoint machinery to save/restore checkpoints.
        self.create_checkpoint_machinery()

        # Create summary file writer.
        self.summary_file_writer = tf.summary.create_file_writer(
            logdir=os.path.join(self.params["output_dir"], "summaries"),
            name="summary_file_writer"
        )

        # Run training loop.
        self.training_loop(steps_per_epoch, train_dataset_iter)

        # Save model at end of training loop.
        self.training_loop_end_save_model()

    def train_and_evaluate(self):
        """Trains and evaluates Keras model.

        Args:
            args: dict, user passed parameters.

        Returns:
            Generator's `Model` object for in-memory predictions.
        """
        if self.params["distribution_strategy"]:
            # If the list of devices is not specified in the
            # Strategy constructor, it will be auto-detected.
            if self.params["distribution_strategy"] == "Mirrored":
                self.strategy = tf.distribute.MirroredStrategy()
            print(
                "Number of devices = {}".format(
                    self.strategy.num_replicas_in_sync
                )
            )

            # Set global batch size for training.
            self.global_batch_size = (
                self.params["train_batch_size"] * self.strategy.num_replicas_in_sync
            )

            # Get input datasets. Batch size is split evenly between replicas.
            train_dataset, eval_dataset = self.get_train_eval_datasets(
                num_replicas=self.strategy.num_replicas_in_sync
            )

            with self.strategy.scope():
                # Create distributed datasets.
                train_dist_dataset = (
                    self.strategy.experimental_distribute_dataset(
                        dataset=train_dataset
                    )
                )
                eval_dist_dataset = (
                    self.strategy.experimental_distribute_dataset(
                        dataset=eval_dataset
                    )
                )

                # Training block setups training, then loops through datasets.
                self.train_block(
                    train_dataset=train_dist_dataset,
                    eval_dataset=eval_dist_dataset
                )
        else:
            # Set global batch size for training.
            self.global_batch_size = self.params["train_batch_size"]

            # Get input datasets.
            train_dataset, eval_dataset = self.get_train_eval_datasets(
                num_replicas=1
            )

            # Training block setups training, then loops through datasets.
            self.train_block(
                train_dataset=train_dataset, eval_dataset=eval_dataset
            )
