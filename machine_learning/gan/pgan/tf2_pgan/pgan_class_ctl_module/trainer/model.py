import tensorflow as tf

from . import export
from . import inputs
from . import instantiate_model
from . import train
from . import train_and_eval
from . import train_step
from . import training_loop


class TrainAndEvaluateModel(
    train_and_eval.TrainAndEval,
    train.Train,
    instantiate_model.InstantiateModel,
    train_step.TrainStep,
    training_loop.TrainingLoop,
    export.Export
):
    """Train and evaluate loop trainer for model.

    Attributes:
        params: dict, user passed parameters.
        network_objects: dict, instances of `Generator` and `Discriminator`
            network objects.
        network_models: dict, instances of Keras `Model`s for each network.
        optimizers: dict, instances of Keras `Optimizer`s for each network.
        strategy: instance of tf.distribute.strategy.
        discriminator_train_step_fn: unbound function, function for a
            dicriminator train step using correct strategy and mode.
        generator_train_step_fn: unbound function, function for a
            generator train step using correct strategy and mode.
        global_batch_size: int, the global batch size after summing batch
            sizes across replicas.
        global_step: tf.Variable, the global step counter across epochs and
            steps within epoch.
        alpha_var: tf.Variable, used in growth transition network's weighted
            sum.
        summary_file_writer: instance of tf.summary.create_file_writer for
            summaries for TensorBoard.
        growth_idx: int, current growth index model has progressed to.
        epoch_step: int, the current step through current epoch.
        previous_timestamp: float, the previous timestamp for profiling the
            steps/sec rate.
    """
    def __init__(self, params):
        """Instantiate trainer.

        Args:
            params: dict, user passed parameters.
        """
        super().__init__()
        self.params = params

        self.network_objects = {}
        self.network_models = {}
        self.optimizers = {}

        self.strategy = None

        self.discriminator_train_step_fn = None
        self.generator_train_step_fn = None

        self.global_batch_size = None

        self.global_step = tf.Variable(
            initial_value=tf.zeros(shape=[], dtype=tf.int64),
            trainable=False,
            name="global_step"
        )

        self.alpha_var = tf.Variable(
            initial_value=tf.zeros(shape=[], dtype=tf.float32),
            trainable=False,
            name="alpha_var"
        )

        self.summary_file_writer = None

        self.growth_idx = 0
        self.epoch_step = 0
        self.previous_timestamp = 0.0

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

        # Instantiate models, create checkpoints, create summary file writer.
        self.prepare_training_components()

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
