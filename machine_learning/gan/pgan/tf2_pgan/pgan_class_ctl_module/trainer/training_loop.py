import tensorflow as tf


class TrainingLoop(object):
    """Class that contains methods for training loop.
    """
    def __init__(self):
        """Instantiate instance of `TrainStep`.
        """
        pass

    def training_loop(self, steps_per_epoch, train_dataset_iter):
        """Loops through training dataset to train model.

        Args:
            steps_per_epoch: int, number of steps/batches to take each epoch.
            train_dataset_iter: iterator, training dataset iterator.
        """
        # Get correct train function based on parameters.
        self.get_train_step_functions()

        # Calculate number of growths. Each progression involves 2 growths,
        # a transition phase and stablization phase.
        num_growths = len(self.params["conv_num_filters"]) * 2 - 1

        for self.growth_idx in range(num_growths):
            print("\ngrowth_idx = {}".format(self.growth_idx))

            # Set active generator and discriminator `Model`s.
            self.set_active_network_models()

            for epoch in range(self.params["num_epochs"]):
                self.previous_timestamp = tf.timestamp()

                self.epoch_step = 0
                while self.epoch_step < steps_per_epoch:
                    # Train discriminator.
                    (growth_phase_complete,
                     features,
                     labels) = self.network_model_training_steps(
                        epoch=epoch,
                        train_step_fn=self.discriminator_train_step_fn,
                        train_steps=self.params["discriminator_train_steps"],
                        train_dataset_iter=train_dataset_iter,
                        features=None,
                        labels=None
                    )

                    if growth_phase_complete:
                        break  # break while loop

                    # Train generator.
                    (growth_phase_complete,
                     _,
                     _) = self.network_model_training_steps(
                        epoch=epoch,
                        train_step_fn=self.generator_train_step_fn,
                        train_steps=self.params["generator_train_steps"],
                        train_dataset_iter=None,
                        features=features,
                        labels=labels
                    )

                    if growth_phase_complete:
                        break  # break while loop

                if growth_phase_complete:
                    break  # break epoch for loop

            if self.params["export_every_growth_phase"]:
                self.export_saved_model()
