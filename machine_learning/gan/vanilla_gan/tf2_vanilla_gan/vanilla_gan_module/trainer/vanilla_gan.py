import tensorflow as tf

from . import discriminators
from . import generators


def instantiate_network_objects(params):
    """Instantiates generator and discriminator with parameters.

    Args:
        params: dict, user passed parameters.

    Returns:
        Dictionary of instance of `Generator` and instance of `Discriminator`.
    """
    # Instantiate generator.
    generator = generators.Generator(
        input_shape=(params["latent_size"]),
        kernel_regularizer=tf.keras.regularizers.l1_l2(
            l1=params["generator_l1_regularization_scale"],
            l2=params["generator_l2_regularization_scale"]
        ),
        bias_regularizer=None,
        name="generator",
        params=params
    )

    # Instantiate discriminator.
    discriminator = discriminators.Discriminator(
        input_shape=(
            params["height"] * params["width"] * params["depth"]
        ),
        kernel_regularizer=tf.keras.regularizers.l1_l2(
            l1=params["discriminator_l1_regularization_scale"],
            l2=params["discriminator_l2_regularization_scale"]
        ),
        bias_regularizer=None,
        name="discriminator",
        params=params
    )

    return {"generator": generator, "discriminator": discriminator}


def instantiate_optimizer(params, scope):
    """Instantiates optimizer with parameters.

    Args:
        params: dict, user passed parameters.
        scope: str, the name of the network of interest.

    Returns:
        Instance of `Optimizer`.
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
    if params["{}_optimizer".format(scope)] == "Adam":
        optimizer = optimizers[params["{}_optimizer".format(scope)]](
            learning_rate=params["{}_learning_rate".format(scope)],
            beta_1=params["{}_adam_beta1".format(scope)],
            beta_2=params["{}_adam_beta2".format(scope)],
            epsilon=params["{}_adam_epsilon".format(scope)],
            name="{}_{}_optimizer".format(
                scope, params["{}_optimizer".format(scope)].lower()
            )
        )
    else:
        optimizer = optimizers[params["{}_optimizer".format(scope)]](
            learning_rate=params["{}_learning_rate".format(scope)],
            name="{}_{}_optimizer".format(
                scope, params["{}_optimizer".format(scope)].lower()
            )
        )

    return optimizer


def vanilla_gan_model(params):
    """Vanilla GAN custom Estimator model function.

    Args:
        params: dict, user passed parameters.

    Returns:
        Dictionary of network objects, dictionary of models objects, and
            dictionary of optimizer objects.
    """
    # Instantiate generator and discriminator objects.
    network_dict = instantiate_network_objects(params)

    # Instantiate generator optimizer.
    generator_optimizer = instantiate_optimizer(params, scope="generator")

    # Instantiate discriminator optimizer.
    discriminator_optimizer = instantiate_optimizer(
        params, scope="discriminator"
    )

    return (
        network_dict,
        {
            "generator": generator_optimizer,
            "discriminator": discriminator_optimizer
        }
    )
