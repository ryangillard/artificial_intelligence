import tensorflow as tf


class TrainAndEval(object):
    """Class that contains methods used for both training and evaluation.
    """
    def __init__(self):
        pass

    def generator_loss_phase(self, mode, training):
        """Gets fake logits and loss for generator.

        Args:
            mode: str, what mode currently in: TRAIN or EVAL.
            training: bool, if model should be training.

        Returns:
            Fake logits of shape [batch_size, 1] and generator loss of shape
                [].
        """
        batch_size = (
            self.params["train_batch_size"]
            if mode == "TRAIN"
            else self.params["eval_batch_size"]
        )

        # Create random noise latent vector for each batch example.
        Z = tf.random.normal(
            shape=[batch_size, self.params["latent_size"]],
            mean=0.0,
            stddev=1.0,
            dtype=tf.float32
        )

        # Get generated image from generator network from gaussian noise.
        fake_images = self.network_models["generator"](
            inputs=Z, training=training
        )

        if self.params["write_summaries"] and mode == "TRAIN":
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
                    tf.summary.image(
                        name="fake_images",
                        data=tf.reshape(
                            tensor=fake_images,
                            shape=[
                                -1,
                                self.params["height"],
                                self.params["width"],
                                self.params["depth"]
                            ]
                        ),
                        step=self.global_step,
                        max_outputs=5
                    )
                    self.summary_file_writer.flush()

        # Get fake logits from discriminator using generator's output image.
        fake_logits = self.network_models["discriminator"](
            inputs=fake_images, training=training
        )

        # Get generator total loss.
        generator_total_loss = (
            self.network_objects["generator"].get_generator_loss(
                global_batch_size=self.global_batch_size,
                fake_logits=fake_logits,
                global_step=self.global_step,
                summary_file_writer=self.summary_file_writer
            )
        )

        return fake_logits, generator_total_loss

    def discriminator_loss_phase(self, real_images, fake_logits, training):
        """Gets real logits and loss for discriminator.

        Args:
            real_images: tensor, real images of shape
                [batch_size, height * width * depth].
            fake_logits: tensor, discriminator logits of fake images of shape
                [batch_size, 1].
            training: bool, if in training mode.

        Returns:
            Real logits of shape [batch_size, 1] and discriminator loss of
                shape [].
        """
        # Get real logits from discriminator using real image.
        real_logits = self.network_models["discriminator"](
            inputs=real_images, training=training
        )

        # Get discriminator total loss.
        discriminator_total_loss = (
            self.network_objects["discriminator"].get_discriminator_loss(
                global_batch_size=self.global_batch_size,
                fake_logits=fake_logits,
                real_logits=real_logits,
                global_step=self.global_step,
                summary_file_writer=self.summary_file_writer
            )
        )

        return real_logits, discriminator_total_loss
