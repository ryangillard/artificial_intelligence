import tensorflow as tf

from . import discriminators
from . import generators


class InstantiateModel(object):
    """Class that contains methods used for instantiating model objects.
    """
    def __init__(self):
        pass

    def instantiate_network_objects(self):
        """Instantiates generator and discriminator with parameters.
        """
        # Instantiate generator.
        self.network_objects["generator"] = generators.Generator(
            kernel_regularizer=tf.keras.regularizers.l1_l2(
                l1=self.params["generator_l1_regularization_scale"],
                l2=self.params["generator_l2_regularization_scale"]
            ),
            bias_regularizer=None,
            name="generator",
            params=self.params,
            alpha_var=self.alpha_var
        )

        # Instantiate discriminator.
        self.network_objects["discriminator"] = discriminators.Discriminator(
            kernel_regularizer=tf.keras.regularizers.l1_l2(
                l1=self.params["discriminator_l1_regularization_scale"],
                l2=self.params["discriminator_l2_regularization_scale"]
            ),
            bias_regularizer=None,
            name="discriminator",
            params=self.params,
            alpha_var=self.alpha_var
        )

    def instantiate_optimizer(self, scope):
        """Instantiates optimizer with parameters.

        Args:
            scope: str, the name of the network of interest.
        """
        # Create optimizer map.
        optimizers = {
            "Adadelta": tf.keras.optimizers.Adadelta,
            "Adagrad": tf.keras.optimizers.Adagrad,
            "Adam": tf.keras.optimizers.Adam,
            "Adamax": tf.keras.optimizers.Adamax,
            "Ftrl": tf.keras.optimizers.Ftrl,
            "Nadam": tf.keras.optimizers.Nadam,
            "RMSprop": tf.keras.optimizers.RMSprop,
            "SGD": tf.keras.optimizers.SGD
        }

        # Get optimizer and instantiate it.
        if self.params["{}_optimizer".format(scope)] == "Adam":
            optimizer = optimizers[self.params["{}_optimizer".format(scope)]](
                learning_rate=self.params["{}_learning_rate".format(scope)],
                beta_1=self.params["{}_adam_beta1".format(scope)],
                beta_2=self.params["{}_adam_beta2".format(scope)],
                epsilon=self.params["{}_adam_epsilon".format(scope)],
                name="{}_{}_optimizer".format(
                    scope, self.params["{}_optimizer".format(scope)].lower()
                )
            )
        else:
            optimizer = optimizers[self.params["{}_optimizer".format(scope)]](
                learning_rate=self.params["{}_learning_rate".format(scope)],
                name="{}_{}_optimizer".format(
                    scope, self.params["{}_optimizer".format(scope)].lower()
                )
            )

        self.optimizers[scope] = optimizer

    def instantiate_model_objects(self):
        """Instantiate model network objects, network models, and optimizers.
        """
        # Instantiate generator and discriminator objects.
        self.instantiate_network_objects()

        # Instantiate generator optimizer.
        self.instantiate_optimizer(scope="generator")

        # Instantiate discriminator optimizer.
        self.instantiate_optimizer(scope="discriminator")
