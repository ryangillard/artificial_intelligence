import os
import tensorflow as tf

from . import checkpoints
from . import export
from . import pca
from . import resnet
from . import train_step
from . import training_inputs
from . import training_loop


class TrainModel(
    checkpoints.Checkpoints,
    train_step.TrainStep,
    training_loop.TrainingLoop,
    export.Export
):
    """Class that trains a model.

    Attributes:
        params: dict, user passed parameters.
        resnet_instance: instance or `ResNet` class.
        pca_model: instance of `PCA` class.
        strategy: instance of tf.distribute.strategy.
        global_batch_size: int, global batch size after summing batch sizes
            across replicas.
        train_dataset_iterator: iterator, iterator of instance of `Dataset`
            for training data.
        train_step_fn: unbound function, function for a train step using
            correct strategy and mode.
        global_step: tf.Variable, the global step counter.
        checkpoint: instance of tf.train.Checkpoint, for saving and restoring
            checkpoints.
        checkpoint_manager: instance of tf.train.CheckpointManager, for
            managing checkpoint path, how often to write, etc.
    """
    def __init__(self, params):
        """Instantiate trainer.

        Args:
            params: dict, user passed parameters.
        """
        super().__init__()
        self.params = params

        self.resnet_instance = resnet.ResNet(
            params={
                "image_height": self.params["image_height"],
                "image_width": self.params["image_width"],
                "image_depth": self.params["image_depth"],
                "resnet_weights": self.params["resnet_weights"],
                "resnet_layer_name": self.params["resnet_layer_name"],
                "preprocess_input": self.params["preprocess_input"]
            }
        )

        self.pca_model = pca.PCA(
            params={
                "num_cols": self.params["num_cols"],
                "use_sample_covariance": self.params["use_sample_covariance"],
                "top_k_pc": self.params["top_k_pc"]
            }
        )

        self.strategy = None
        self.global_batch_size = []

        self.train_dataset_iterator = None

        self.train_step_fn = None

        self.global_step = tf.Variable(
            initial_value=tf.zeros(shape=[], dtype=tf.int64),
            trainable=False,
            name="global_step"
        )

        self.checkpoint = None
        self.checkpoint_manager = None

    def get_train_dataset(self, num_replicas):
        """Gets train dataset.

        Args:
            num_replicas: int, number of device replicas.

        Returns:
            `tf.data.Dataset` for training data.
        """
        return training_inputs.read_dataset(
            file_pattern=self.params["train_file_pattern"],
            batch_size=self.params["train_batch_size"] * num_replicas,
            params=self.params
        )()

    def train_block(self, train_dataset):
        """Training block setups training, then loops through datasets.

        Args:
            train_dataset: instance of `Dataset` for training data.
        """
        # Create iterators of datasets.
        self.train_dataset_iterator = iter(train_dataset)

        # Create checkpoint machinery to save/restore checkpoints.
        self.create_checkpoint_machinery()

        # Run training loop.
        self.training_loop()

    def train_model(self):
        """Trains Keras model.

        Args:
            args: dict, user passed parameters.
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

            # Get input dataset. Batch size is split evenly between replicas.
            train_dataset = self.get_train_dataset(
                num_replicas=self.strategy.num_replicas_in_sync
            )

            with self.strategy.scope():
                # Create distributed datasets.
                train_dataset = (
                    self.strategy.experimental_distribute_dataset(
                        dataset=train_dataset
                    )
                )

                # Training block setups training, then loops through datasets.
                self.train_block(train_dataset=train_dataset)
        else:
            # Set global batch size for training.
            self.global_batch_size = self.params["train_batch_size"]

            # Get input datasets.
            train_dataset = self.get_train_dataset(num_replicas=1)

            # Training block setups training, then loops through datasets.
            self.train_block(train_dataset=train_dataset)
