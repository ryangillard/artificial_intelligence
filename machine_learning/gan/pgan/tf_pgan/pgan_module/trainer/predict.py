import tensorflow as tf

from .print_object import print_obj


def get_predictions(Z, generator, params, block_idx):
    """Gets predictions from latent vectors Z.

    Args:
        Z: tensor, latent vectors of shape [cur_batch_size, latent_size].
        generator: instance of `Generator`.
        params: dict, user passed parameters.
        block_idx: int, current conv layer block's index.

    Returns:
        Predictions dictionary of generated images from generator.
    """
    # Get predictions from generator.
    generated_images = generator.get_predict_generator_outputs(
        Z=Z, params=params, block_idx=block_idx
    )
    print_obj("\nget_predictions", "generated_images", generated_images)

    # Calculate image size for returned dict keys.
    image_dim = 4 * 2 ** block_idx
    image_size = "{}x{}".format(image_dim, image_dim)

    return {
        "generated_images_{}".format(image_size): generated_images
    }


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

    loop_end = len(params["conv_num_filters"])
    loop_start = 0 if params["predict_all_resolutions"] else loop_end - 1
    print_obj(func_name, "loop_start", loop_start)
    print_obj(func_name, "loop_end", loop_end)

    # Create predictions dictionary.
    predictions_dict = {}
    for i in range(loop_start, loop_end):
        predictions = get_predictions(
            Z=Z,
            generator=generator,
            params=params,
            block_idx=i
        )
        predictions_dict.update(predictions)
    print_obj(func_name, "predictions_dict", predictions_dict)

    # Create export outputs.
    export_outputs = {
        "predict_export_outputs": tf.estimator.export.PredictOutput(
            outputs=predictions_dict)
    }
    print_obj(func_name, "export_outputs", export_outputs)

    return predictions_dict, export_outputs
