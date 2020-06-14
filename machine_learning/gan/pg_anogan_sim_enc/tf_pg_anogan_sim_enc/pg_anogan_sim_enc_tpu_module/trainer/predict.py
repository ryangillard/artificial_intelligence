import tensorflow as tf

from . import anomaly
from . import image_utils
from .print_object import print_obj


def get_predictions(features, generator, encoder, params, block_idx):
    """Gets predictions from query image.

    Args:
        features: dict, feature tensors from serving input function.
        generator: instance of `Generator`.
        encoder: instance of `Encoder`.
        params: dict, user passed parameters.
        block_idx: int, current conv layer block's index.

    Returns:
        Predictions dictionary of encoded images from generator, anomaly
            scores, and anomaly flags.
    """
    func_name = "get_predictions"
    print_obj("\n" + func_name, "features", features)

    # Extract given query image from features dictionary.
    query_images = features["query_image"]

    # Resize query image.
    resized_query_images = image_utils.resize_real_image(
        image=query_images, params=params, block_idx=block_idx
    )
    print_obj(func_name, "resized_query_images", resized_query_images)

    # Get encoder logits using query images.
    print(
        "\nCall encoder with resized_query_images = {}.".format(
            resized_query_images
        )
    )
    encoder_logits = encoder.get_predict_encoder_logits(
        X=resized_query_images, params=params, block_idx=block_idx
    )
    print_obj(func_name, "encoder_logits", encoder_logits)

    # Get encoded images from generator using encoder's logits.
    print(
        "\nCall generator with encoder_logits = {}.".format(
            encoder_logits
        )
    )
    encoded_images = generator.get_predict_vec_to_img_outputs(
        Z=encoder_logits, params=params, block_idx=block_idx
    )
    print_obj(func_name, "encoded_images", encoded_images)

    # Perform anomaly detection.
    anomaly_scores, anomaly_flags = anomaly.anomaly_detection(
        resized_query_images, encoder_logits, encoded_images, params
    )
    print_obj(func_name, "anomaly_scores", anomaly_scores)
    print_obj(func_name, "anomaly_flags", anomaly_flags)

    # Calculate image size for returned dict keys.
    image_dim = 4 * 2 ** block_idx
    image_size = "{}x{}".format(image_dim, image_dim)

    # Create predictions dictionary.
    predictions_dict = {
        "encoded_images_{}".format(image_size): encoded_images,
        "anomaly_scores_{}".format(image_size): anomaly_scores,
        "anomaly_flags_{}".format(image_size): anomaly_flags
    }

    # If we also want to predict G(z) using serving input z.
    if params["predict_g_z"]:
        # Extract given Z from features dictionary.
        Z = features["Z"]

        # Get generated images from generator using Z latent vector.
        print("\nCall generator with Z = {}.".format(Z))
        generated_images = generator.get_predict_vec_to_img_outputs(
            Z=Z, params=params, block_idx=block_idx
        )
        print_obj(func_name, "generated_images", generated_images)

        # Update predictions dictionary.
        predictions_dict.update(
            {
                "generated_images_{}".format(image_size): generated_images
            }
        )

    return predictions_dict


def get_predictions_and_export_outputs(features, generator, encoder, params):
    """Gets predictions and serving export outputs.

    Args:
        features: dict, feature tensors from serving input function.
        generator: instance of generator.`Generator`.
        encoder: instance of encoder.`Encoder`.
        params: dict, user passed parameters.

    Returns:
        Predictions dictionary and export outputs dictionary.
    """
    func_name = "get_predictions_and_export_outputs"
    print_obj("\n" + func_name, "features", features)

    loop_end = len(params["conv_num_filters"])
    loop_start = 0 if params["predict_all_resolutions"] else loop_end - 1
    print_obj(func_name, "loop_start", loop_start)
    print_obj(func_name, "loop_end", loop_end)

    # Create predictions dictionary.
    predictions_dict = {}
    for i in range(loop_start, loop_end):
        predictions = get_predictions(
            features=features,
            generator=generator,
            encoder=encoder,
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
