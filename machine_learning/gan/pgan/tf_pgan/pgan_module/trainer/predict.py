import tensorflow as tf

from .print_object import print_obj

def get_predictions(params):
    """Gets predictions and serves the export output.

    Args:
        params: dict, user passed parameters.

    Returns:
        Predictions dictionary and export outputs dictionary.
    """
    # Extract given latent vectors from features dictionary.
    Z = tf.cast(x=features["Z"], dtype=tf.float32)
    print_obj("get_predictions", "Z", Z)

    # Get predictions from generator.
    generated_images = pgan_generator.get_predict_generator_outputs(
        Z=Z, params=params
    )
    print_obj("get_predictions", "generated_images", generated_images)

    # Create predictions dictionary.
    if params["predict_all_resolutions"]:
        predictions_dict = {
            "generated_images_{}x{}".format(
                4 * 2 ** i, 4 * 2 ** i
            ): generated_images[i]
            for i in range(len(params["conv_num_filters"]))
        }
    else:
        predictions_dict = {
            "generated_images": generated_images
        }
    print_obj("get_predictions", "predictions_dict", predictions_dict)

    # Create export outputs.
    export_outputs = {
        "predict_export_outputs": tf.estimator.export.PredictOutput(
            outputs=predictions_dict)
    }
    print_obj("get_predictions", "export_outputs", export_outputs)

    return predictions_dict, export_outputs
