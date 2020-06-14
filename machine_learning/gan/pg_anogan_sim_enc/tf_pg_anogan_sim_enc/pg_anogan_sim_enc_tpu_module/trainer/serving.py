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
    # shape = (batch_size, height, width, depth)
    feature_placeholders = {
        "query_image": tf.placeholder(
            dtype=tf.uint8,
            shape=[None, params["height"], params["width"], params["depth"]],
            name="{}_placeholder_query_image".format(func_name)
        )
    }

    # If we also want to predict G(z) using serving input z.
    if params["predict_g_z"]:
        feature_placeholders.update(
            {
                "Z": tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, params["latent_size"]],
                    name="{}_placeholder_Z".format(func_name)
                )
            }
        )

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

    # Apply same processing to image as train and eval.
    features["query_image"] = image_utils.preprocess_image(
        image=features["query_image"], params=params
    )
    print_obj(func_name, "features", features)

    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors=feature_placeholders
    )
