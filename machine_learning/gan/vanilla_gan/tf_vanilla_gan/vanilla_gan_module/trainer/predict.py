import tensorflow as tf

from .print_object import print_obj


def get_predictions_and_export_outputs(features, generator, params):
    """Gets predictions and serving export outputs.

    Args:
        features: dict, feature tensors from serving input function.
        generator: instance of `Generator`.
        params: dict, user passed parameters.

    Returns:
        Predictions dictionary and export outputs dictionary.
    """
    func_name = "get_predictions_and_export_outputs"
    # Extract given latent vectors from features dictionary.
    Z = tf.cast(x=features["Z"], dtype=tf.float32)
    print_obj("\n" + func_name, "Z", Z)

    # Establish generator network subgraph.
    fake_images = generator.get_fake_images(Z=Z, params=params)

    # Reshape into a rank 4 image.
    generated_images = tf.reshape(
        tensor=fake_images,
        shape=[-1, params["height"], params["width"], params["depth"]]
    )

    # Create predictions dictionary.
    predictions_dict = {
        "generated_images": generated_images
    }

    # Create export outputs.
    export_outputs = {
        "predict_export_outputs": tf.estimator.export.PredictOutput(
            outputs=predictions_dict)
    }

    return predictions_dict, export_outputs
