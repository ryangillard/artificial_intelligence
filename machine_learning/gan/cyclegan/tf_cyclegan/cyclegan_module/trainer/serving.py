import tensorflow as tf

from . import image_utils
from .print_object import print_obj


def serving_input_fn(params):
    """Serving input function.

    Args:
        params: dict, user passed parameters.

    Returns:
        ServingInputReceiver object containing features and receiver tensors.
    """
    func_name = "serving_input_fn"
    # Create placeholders to accept data sent to the model at serving time.
    # shape = [batch_size, height, width, depth]
    feature_placeholders = {
        "source_image": tf.placeholder(
            dtype=tf.float32,
            shape=[None, params["height"], params["width"], params["depth"]],
            name="serving_input_placeholder_source_image"
        ),
        "target_domain_index": tf.placeholder(
            dtype=tf.int32,
            shape=[None],
            name="serving_input_placeholder_target_domain_index"
        )
    }
    print_obj("\n" + func_name, "feature_placeholders", feature_placeholders)

    # Create clones of the feature placeholder tensors so that the SavedModel
    # SignatureDef will point to the placeholder.
    features = {
        key: tf.identity(
            input=value,
            name="{}_identity_placeholder_{}".format(func_name, key)
        )
        for key, value in feature_placeholders.items()
    }
    print_obj(func_name, "features", features)

    # Apply same preprocessing as before.
    features["source_image"] = image_utils.preprocess_image(
        image=features["source_image"],
        mode=tf.estimator.ModeKeys.PREDICT,
        params=params
    )
    print_obj(func_name, "features", features)

    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors=feature_placeholders
    )
