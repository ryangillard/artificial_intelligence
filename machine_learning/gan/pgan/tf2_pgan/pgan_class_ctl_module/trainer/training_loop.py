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

        for self.growth_idx in range(self.num_growths):
            self.growth_step = 0
            self.block_idx = (self.growth_idx + 1) // 2
            print(
                "\nblock_idx = {}, growth_idx = {}".format(
                    self.block_idx, self.growth_idx
                )
            )
            print(
                "\ngenerator_model = {}".format(
                    self.network_objects["generator"].models[self.growth_idx].summary()
                )
            )
            print(
                "\ndiscriminator_model = {}".format(
                    self.network_objects["discriminator"].models[self.growth_idx].summary()
                )
            )

            global_batch_size = (
                self.global_batch_size_schedule[self.block_idx]
            )
            steps_per_epoch = (
                self.params["train_dataset_length"] // global_batch_size
            )

            for epoch in range(self.params["num_epochs"]):
                print("\ngrowth_idx = {}, epoch = {}".format(self.growth_idx, epoch))
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
                        train_dataset_iter=(
                            self.train_datasets[self.block_idx]
                        ),
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
