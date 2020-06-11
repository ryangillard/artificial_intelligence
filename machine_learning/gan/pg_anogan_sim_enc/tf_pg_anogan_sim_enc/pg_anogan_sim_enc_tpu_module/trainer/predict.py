import tensorflow as tf

from . import anomaly
from . import image_utils
from .print_object import print_obj


def get_predictions(query_images, generator, encoder, params, block_idx):
    """Gets predictions from query image.

    Args:
        image: tensor, tf.float32 query image of shape
            [None, height, width, depth].
        generator: instance of generator.`Generator`.
        encoder: instance of encoder.`Encoder`.
        params: dict, user passed parameters.
        block_idx: int, current conv layer block's index.

    Returns:
        Predictions dictionary of encoded images from generator, anomaly
            scores, and anomaly flags.
    """
    print_obj("\nget_predictions", "query_images", query_images)
    # Resize query image.
    resized_query_images = image_utils.resize_real_image(
        image=query_images, params=params, block_idx=block_idx
    )
    print_obj("get_predictions", "resized_query_images", resized_query_images)

    # Get encoder logits using query images.
    print(
        "\nCall encoder with resized_query_images = {}.".format(
            resized_query_images
        )
    )
    encoder_logits = encoder.get_predict_encoder_logits(
        X=resized_query_images, params=params, block_idx=block_idx
    )
    print_obj("get_predictions", "encoder_logits", encoder_logits)

    # Get encoded images from generator using encoder's logits.
    print(
        "\nCall generator with encoder_logits = {}.".format(
            encoder_logits
        )
    )
    encoded_images = generator.get_predict_vec_to_img_outputs(
        Z=encoder_logits, params=params, block_idx=block_idx
    )
    print_obj("get_predictions", "encoded_images", encoded_images)

    # Perform anomaly detection.
    anomaly_scores, anomaly_flags = anomaly.anomaly_detection(
        resized_query_images, encoder_logits, encoded_images, params
    )
    print_obj("get_predictions", "anomaly_scores", anomaly_scores)
    print_obj("get_predictions", "anomaly_flags", anomaly_flags)

    # Calculate image size for returned dict keys.
    image_dim = 4 * 2 ** block_idx
    image_size = "{}x{}".format(image_dim, image_dim)

    return {
        "encoded_images_{}".format(image_size): encoded_images,
        "anomaly_scores_{}".format(image_size): anomaly_scores,
        "anomaly_flags_{}".format(image_size): anomaly_flags
    }


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
    # Extract given query image from features dictionary.
    query_images = features["query_image"]
    print_obj(
        "\nget_predictions_and_export_outputs", "query_images", query_images
    )

    loop_end = len(params["conv_num_filters"])
    loop_start = 0 if params["predict_all_resolutions"] else loop_end - 1
    print_obj("get_predictions_and_export_outputs", "loop_start", loop_start)
    print_obj("get_predictions_and_export_outputs", "loop_end", loop_end)

    # Create predictions dictionary.
    predictions_dict = {}
    for i in range(loop_start, loop_end):
        predictions = get_predictions(
            query_images=query_images,
            generator=generator,
            encoder=encoder,
            params=params,
            block_idx=i
        )
        predictions_dict.update(predictions)
    print_obj(
        "get_predictions_and_export_outputs",
        "predictions_dict",
        predictions_dict
    )

    # Create export outputs.
    export_outputs = {
        "predict_export_outputs": tf.estimator.export.PredictOutput(
            outputs=predictions_dict)
    }
    print_obj(
        "get_predictions_and_export_outputs", "export_outputs", export_outputs
    )

    return predictions_dict, export_outputs
