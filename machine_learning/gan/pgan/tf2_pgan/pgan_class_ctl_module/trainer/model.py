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
        optimizers: dict, instances of Keras `Optimizer`s for each network.
        strategy: instance of tf.distribute.strategy.
        global_batch_size_schedule: list, schedule of ints for the global
            batch size after summing batch sizes across replicas.
        train_datasets: list, instances of `Dataset` for each resolution
            block for training.
        eval_datasets: list, instances of `Dataset` for each resolution
            block for evaluation.
        discriminator_train_step_fn: unbound function, function for a
            dicriminator train step using correct strategy and mode.
        generator_train_step_fn: unbound function, function for a
            generator train step using correct strategy and mode.
        global_step: tf.Variable, the global step counter across epochs and
            steps within epoch.
        alpha_var: tf.Variable, used in growth transition network's weighted
            sum.
        summary_file_writer: instance of tf.summary.create_file_writer for
            summaries for TensorBoard.
        num_growths: int, number of growth phases to train over.
        num_steps_until_growth_schedule: list, ints representing a schedule of
            the number of steps/batches until the next growth.
        unique_trainable_variables: dict, list of unique trainable variables
         for each model type unioned across all growths.
        growth_idx: int, current growth index model has progressed to.
        block_idx: int, current growth block/resolution model is in.
        growth_step: int, current number of steps since last growth.
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
        self.optimizers = {}

        self.strategy = None
        self.global_batch_size_schedule = []

        self.train_datasets = []
        self.eval_datasets = []

        self.discriminator_train_step_fn = None
        self.generator_train_step_fn = None

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

        self.checkpoint_manager = None
        self.summary_file_writer = None

        # Calculate number of growths. Each progression involves 2 growths,
        # a transition phase and stablization phase.
        self.num_growths = len(self.params["conv_num_filters"]) * 2 - 1
        self.num_steps_until_growth_schedule = (
            self.params["num_steps_until_growth_schedule"]
        )

        self.unique_trainable_variables = {}

        self.growth_idx = 0
        self.block_idx = 0
        self.growth_step = 0
        self.epoch_step = 0
        self.previous_timestamp = 0.0

    def get_train_eval_datasets(self, num_replicas, block_idx):
        """Gets train and eval datasets.

        Args:
            num_replicas: int, number of device replicas.
            block_idx: int, resolution block index.

        Returns:
            Train and eval datasets.
        """
        train_batch_size = self.params["train_batch_size_schedule"][block_idx]
        train_dataset = inputs.read_dataset(
            filename=self.params["train_file_pattern"],
            batch_size=train_batch_size * num_replicas,
            params=self.params,
            training=True
        )()

        eval_batch_size = self.params["eval_batch_size_schedule"][block_idx]
        eval_dataset = inputs.read_dataset(
            filename=self.params["eval_file_pattern"],
            batch_size=eval_batch_size * num_replicas,
            params=self.params,
            training=False
        )()
        if self.params["eval_steps"]:
            eval_dataset = eval_dataset.take(count=self.params["eval_steps"])

        return train_dataset, eval_dataset

    def train_block(self):
        """Training block setups training, then loops through datasets.
        """
        # Create iterators of datasets.
        self.train_datasets = [iter(x) for x in self.train_datasets]
        self.eval_datasets = [iter(x) for x in self.eval_datasets]

        # Instantiate models, create checkpoints, create summary file writer.
        self.prepare_training_components()

        # Run training loop.
        self.training_loop()

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
            self.global_batch_size_schedule = [
                x * self.strategy.num_replicas_in_sync
                for x in self.params["train_batch_size_schedule"]
            ]

            # Shorten growth schedule due to parallel work from replicas.
            self.num_steps_until_growth_schedule = [
                x // self.strategy.num_replicas_in_sync
                for x in self.params["num_steps_until_growth_schedule"]
            ]

            # Get input datasets. Batch size is split evenly between replicas.
            num_blocks = (self.num_growths + 1) // 2
            for block_idx in range(num_blocks):
                train_dataset, eval_dataset = self.get_train_eval_datasets(
                    num_replicas=self.strategy.num_replicas_in_sync,
                    block_idx=block_idx
                )
                self.train_datasets.append(train_dataset)
                self.eval_datasets.append(eval_dataset)

            with self.strategy.scope():
                # Create distributed datasets.
                self.train_datasets = [
                    self.strategy.experimental_distribute_dataset(dataset=x)
                    for x in self.train_datasets
                ]

                self.eval_datasets = [
                    self.strategy.experimental_distribute_dataset(dataset=x)
                    for x in self.eval_datasets
                ]

                # Training block setups training, then loops through datasets.
                self.train_block()
        else:
            # Set global batch size for training.
            self.global_batch_size_schedule = self.params["train_batch_size_schedule"]

            # Get input datasets.
            num_blocks = (self.num_growths + 1) // 2
            for block_idx in range(num_blocks):
                train_dataset, eval_dataset = self.get_train_eval_datasets(
                    num_replicas=1,
                    block_idx=block_idx
                )
                self.train_datasets.append(train_dataset)
                self.eval_datasets.append(eval_dataset)

            # Training block setups training, then loops through datasets.
            self.train_block()
