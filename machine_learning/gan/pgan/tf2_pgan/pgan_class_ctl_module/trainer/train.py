import os
import tensorflow as tf


class Train(object):
    """Class that contains methods used for only training.
    """
    def __init__(self):
        """Instantiate instance of `Train`.
        """
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
        variables = (
            self.network_objects[scope].models[self.growth_idx].trainable_variables
        )

        # Get gradients from gradient tape.
        gradients = gradient_tape.gradient(
            target=loss, sources=variables
        )

        # Clip gradients.
        if self.params["{}_clip_gradients".format(scope)]:
            gradients, _ = tf.clip_by_global_norm(
                t_list=gradients,
                clip_norm=self.params["{}_clip_gradients".format(scope)],
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

    def get_network_losses_variables_and_gradients(self, real_images):
        """Gets losses, variables, and gradients for each network.

        Args:
            real_images: tensor, real images of shape
                [batch_size, height, width, depth].

        Returns:
            Dictionaries of network losses, variables, and gradients.
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            # Get fake logits from generator.
            (fake_images,
             fake_logits,
             generator_loss) = self.generator_loss_phase(training=True)

            # Get discriminator loss.
            _, discriminator_loss = self.discriminator_loss_phase(
                fake_images, real_images, fake_logits, training=True
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

        return loss_dict, vars_dict, grads_dict

    def train_network(self, variables, gradients, scope):
        """Trains network variables using gradients with optimizer.

        Args:
            variables: dict, lists for each network's trainable variables.
            gradients: dict, lists for each network's gradients of loss wrt
                network's trainable variables.
            scope: str, the name of the network of interest.
        """
        # Zip together gradients and variables.
        grads_and_vars = zip(gradients[scope], variables[scope])

        # Applying gradients to variables using optimizer.
        self.optimizers[scope].apply_gradients(grads_and_vars=grads_and_vars)

    def resize_real_images(self, images):
        """Resizes real images to match the GAN's current size.

        Args:
            images: tensor, original images.

        Returns:
            Resized image tensor.
        """
        block_idx = (self.growth_idx + 1) // 2
        height, width = self.params["generator_projection_dims"][0:2]
        resized_image = tf.image.resize(
            images=images,
            size=[
                height * (2 ** block_idx), width * (2 ** block_idx)
            ],
            method="nearest",
            name="resized_real_image_{}".format(self.growth_idx)
        )

        return resized_image

    def train_discriminator(self, features, growth_idx):
        """Trains discriminator network.

        Args:
            features: dict, feature tensors from input function.
            growth_idx: int, current growth index model has progressed to.

        Returns:
            Dictionary of scalar losses for each network.
        """
        # Extract real images from features dictionary.
        real_images = self.resize_real_images(images=features["image"])

        # Get gradients for training by running inputs through networks.
        losses, variables, gradients = (
            self.get_network_losses_variables_and_gradients(real_images)
        )

        # Train discriminator network.
        self.train_network(variables, gradients, scope="discriminator")

        return losses

    def train_generator(self, features, growth_idx):
        """Trains generator network.

        Args:
            features: dict, feature tensors from input function.
            growth_idx: int, current growth index model has progressed to.

        Returns:
            Dictionary of scalar losses for each network.
        """
        # Extract real images from features dictionary.
        real_images = self.resize_real_images(images=features["image"])

        # Get gradients for training by running inputs through networks.
        losses, variables, gradients = (
            self.get_network_losses_variables_and_gradients(real_images)
        )

        # Train generator network.
        self.train_network(variables, gradients, scope="generator")

        return losses

    def create_checkpoint_machinery(self):
        """Creates checkpoint machinery needed to save & restore checkpoints.
        """
        # Create checkpoint instance.
        checkpoint_dir = os.path.join(
            self.params["output_dir"], "checkpoints"
        )
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

        checkpoint = tf.train.Checkpoint(
            generator_model=(
                self.network_objects["generator"].models[self.num_growths - 1]
            ),
            discriminator_model=(
                self.network_objects["discriminator"].models[self.num_growths - 1]
            ),
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

    def prepare_training_components(self):
        """Prepares all components necessary for training.
        """
        # Instantiate model objects.
        self.instantiate_model_objects()

        # Create checkpoint machinery to save/restore checkpoints.
        self.create_checkpoint_machinery()

        # Create summary file writer.
        self.summary_file_writer = tf.summary.create_file_writer(
            logdir=os.path.join(self.params["output_dir"], "summaries"),
            name="summary_file_writer"
        )
