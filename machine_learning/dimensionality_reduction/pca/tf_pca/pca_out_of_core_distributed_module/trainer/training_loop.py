import tensorflow as tf


class TrainingLoop(object):
    """Class that contains methods for training loop.
    """
    def __init__(self):
        """Instantiate instance of `TrainStep`.
        """
        pass

    def training_loop(self):
        """Loops through training dataset to train model.
        """
        # Get correct train function based on parameters.
        self.get_train_step_functions()

        num_steps = (
            self.params["train_dataset_length"] // self.global_batch_size
        )

        while self.global_step.numpy() < num_steps:
            # Train model.
            self.perform_training_step(
                train_dataset_iterator=self.train_dataset_iterator,
                train_step_fn=self.train_step_fn
            )

        self.training_loop_end_save_model()
