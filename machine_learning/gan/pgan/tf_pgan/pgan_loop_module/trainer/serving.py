import tensorflow as tf

from .print_object import print_obj


def serving_input_fn(params):
    """Serving input function.

    Args:
        params: dict, user passed parameters.

    Returns:
        ServingInputReceiver object containing features and receiver tensors.
    """
    # Create placeholders to accept data sent to the model at serving time.
    # shape = (batch_size,)
    feature_placeholders = {
        "Z": tf.placeholder(
            dtype=tf.float32,
            shape=[None, params["latent_size"]],
            name="serving_input_placeholder_Z"
        )
    }

    print_obj(
        "\nserving_input_fn",
        "feature_placeholders",
        feature_placeholders
    )

    # Create clones of the feature placeholder tensors so that the SavedModel
    # SignatureDef will point to the placeholder.
    features = {
        key: tf.identity(
            input=value,
            name="serving_input_fn_identity_placeholder_{}".format(key)
        )
        for key, value in feature_placeholders.items()
    }

    print_obj(
        "serving_input_fn",
        "features",
        features
    )

    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors=feature_placeholders
    )
