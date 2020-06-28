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
    # Extract latent vectors from features dictionary.
    Z = features["Z"]
    print_obj("\n" + func_name, "Z", Z)

    # Extract labels from features dictionary & expand from vector to matrix.
    labels = tf.expand_dims(input=features["label"], axis=-1)
    print_obj(func_name, "labels", labels)

    # Establish generator network subgraph.
    fake_images = generator.get_fake_images(Z=Z, labels=labels, params=params)
    print_obj(func_name, "fake_images", fake_images)

    # Reshape into a rank 4 image.
    generated_images = tf.reshape(
        tensor=fake_images,
        shape=[-1, params["height"], params["width"], params["depth"]]
    )
    print_obj(func_name, "generated_images", generated_images)

    # Create predictions dictionary.
    predictions_dict = {
        "generated_images": generated_images
    }
    print_obj(func_name, "predictions_dict", predictions_dict)

    # Create export outputs.
    export_outputs = {
        "predict_export_outputs": tf.estimator.export.PredictOutput(
            outputs=predictions_dict)
    }
    print_obj(func_name, "export_outputs", export_outputs)

    return predictions_dict, export_outputs
