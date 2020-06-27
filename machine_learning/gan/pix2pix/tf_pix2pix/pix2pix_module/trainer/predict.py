import tensorflow as tf

from . import image_utils
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
    source_images = tf.cast(x=features["source_image"], dtype=tf.float32)
    print_obj("\n" + func_name, "source_images", source_images)

    # Get generated images from generator using latent vector.
    generated_images = generator.get_fake_images(
        source_images=source_images, params=params
    )
    print_obj(func_name, "generated_images", generated_images)

    # Resize generated images to match real image sizes.
    generated_images = image_utils.resize_fake_images(
        fake_images=generated_images, params=params
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
