import tensorflow as tf

from . import image_utils
from .print_object import print_obj


def get_predictions_and_export_outputs(
        features, generator_domain_a2b, generator_domain_b2a, params):
    """Gets predictions and serving export outputs.

    Args:
        features: dict, feature tensors from serving input function.
        generator_domain_a2b: instance of `Generator` for domain a to b.
        generator_domain_b2a: instance of `Generator` for domain b to a.
        params: dict, user passed parameters.

    Returns:
        Predictions dictionary and export outputs dictionary.
    """
    func_name = "get_predictions_and_export_outputs"

    # Extract given source images from features dictionary.
    source_images = features["source_image"]
    print_obj("\n" + func_name, "source_images", source_images)

    # Extract domain indices from features dictionary.
    target_domain_indices = features["target_domain_index"]
    print_obj(func_name, "target_domain_indices", target_domain_indices)

    # Get images from generator network for domain a.
    generated_images_domain_b = generator_domain_a2b.get_fake_images(
        source_images=source_images, params=params
    )
    print_obj(
        func_name, "generated_images_domain_b", generated_images_domain_b
    )

    # Get images from generator network for domain b.
    generated_images_domain_a = generator_domain_b2a.get_fake_images(
        source_images=source_images, params=params
    )
    print_obj(
        func_name, "generated_images_domain_a", generated_images_domain_a
    )

    # Interweave domain a and b generated images based on domain index.
    generated_images = tf.where(
        condition=tf.equal(x=target_domain_indices, y=0),
        x=generated_images_domain_a,
        y=generated_images_domain_b,
        name="prediction_where_generated_images"
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
