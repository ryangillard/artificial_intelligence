import tensorflow as tf

from . import regularization
from . import vector_to_image
from .print_object import print_obj


class Generator(vector_to_image.VectorToImage):
    """Generator that takes latent vector input and outputs image.

    Fields:
        name: str, name of `Generator`.
    """
    def __init__(self, kernel_regularizer, bias_regularizer, params, name):
        """Instantiates and builds generator network.

        Args:
            kernel_regularizer: `l1_l2_regularizer` object, regularizar for
                kernel variables.
            bias_regularizer: `l1_l2_regularizer` object, regularizar for bias
                variables.
            params: dict, user passed parameters.
            name: str, name of `Generator`.
        """
        # Set name of `Generator`.
        self.name = name

        # Set kind of `VectorToImage`.
        kind = "generator"

        # Initialize base class.
        super().__init__(kernel_regularizer, bias_regularizer, params, kind)

    ##########################################################################
    ##########################################################################
    ##########################################################################

    def get_generator_loss(self, fake_logits, params):
        """Gets generator loss.

        Args:
            fake_logits: tensor, shape of [cur_batch_size, 1] that came from
                discriminator having processed generator's output image.
            params: dict, user passed parameters.

        Returns:
            Generator's total loss tensor of shape [].
        """
        func_name = "get_generator_loss"

        # Calculate base generator loss.
        generator_loss = -tf.reduce_mean(
            input_tensor=fake_logits,
            name="{}_loss".format(self.name)
        )
        print_obj("\n" + func_name, "generator_loss", generator_loss)

        # Get generator regularization losses.
        generator_reg_loss = regularization.get_regularization_loss(
            lambda1=params["generator_l1_regularization_scale"],
            lambda2=params["generator_l2_regularization_scale"],
            scope=self.name
        )
        print_obj(func_name, "generator_reg_loss", generator_reg_loss)

        # Combine losses for total losses.
        generator_total_loss = tf.math.add(
            x=generator_loss,
            y=generator_reg_loss,
            name="{}_total_loss".format(self.name)
        )
        print_obj(func_name, "generator_total_loss", generator_total_loss)

        return generator_total_loss
