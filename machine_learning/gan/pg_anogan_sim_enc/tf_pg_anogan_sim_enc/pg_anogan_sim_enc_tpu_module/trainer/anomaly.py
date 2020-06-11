import tensorflow as tf

from .print_object import print_obj


def minmax_normalization(X):
    """Min-max normalizes images.

    Args:
        X: tensor, image tensor of rank 4.

    Returns:
        Min-max normalized image tensor.
    """
    min_x = tf.reduce_min(input_tensor=X, name="minmax_normalization_min")

    max_x = tf.reduce_min(input_tensor=X, name="minmax_normalization_max")

    X_normalized = tf.math.divide_no_nan(
        x=X - min_x, y=max_x, name="minmax_normalization_normalized"
    )

    return X_normalized


def get_residual_loss(query_images, encoded_images, params):
    """Gets residual loss between query and encoded images.

    Args:
        query_images: tensor, query image input for predictions.
        encoded_images: tensor, image from generator from encoder's logits.
        params: dict, user passed parameters.

    Returns:
        Residual loss scalar tensor.
    """
    # Minmax normalize query images.
    query_images_normalized = minmax_normalization(X=query_images)
    print_obj(
        "\nquery_images_normalized",
        "query_images_normalized",
        query_images_normalized
    )

    # Minmax normalize encoded images.
    encoded_images_normalized = minmax_normalization(X=encoded_images)

    # Find pixel difference between the normalized query and encoded images.
    image_difference = tf.subtract(
        x=query_images_normalized,
        y=encoded_images_normalized,
        name="image_difference"
    )
    print_obj("get_residual_loss", "image_difference", image_difference)

    # Take L2 norm of difference.
    image_difference_l2_norm = tf.reduce_sum(
        input_tensor=tf.square(x=image_difference),
        axis=[1, 2, 3],
        name="image_difference_l2_norm"
    )
    print_obj(
        "get_residual_loss",
        "image_difference_l2_norm",
        image_difference_l2_norm
    )

    # Scale by image dimension sizes to get residual loss.
    residual_loss = tf.divide(
        x=image_difference_l2_norm,
        y=tf.cast(
            x=params["height"] * params["width"] * params["depth"],
            dtype=tf.float32
        ),
        name="residual_loss"
    )
    print_obj("get_residual_loss", "residual_loss", residual_loss)

    return residual_loss


def get_origin_distance_loss(encoder_logits, params):
    """Gets origin distance loss measuring distance of z-hat from origin.

    Args:
        encoder_logits: tensor, encoder's logits encoded from query images.
        params: dict, user passed parameters.

    Returns:
        Origin distance loss scalar tensor.
    """
    # Take L2 norm of difference.
    z_hat_l2_norm = tf.sqrt(
        x=tf.reduce_sum(
            input_tensor=tf.square(x=encoder_logits),
            axis=-1
        ),
        name="z_hat_l2_norm"
    )
    print_obj("\nget_origin_distance_loss", "z_hat_l2_norm", z_hat_l2_norm)

    # Scale by latent size to get origin distance loss.
    origin_distance_loss = tf.divide(
        x=-z_hat_l2_norm,
        y=tf.sqrt(x=tf.cast(x=params["latent_size"], dtype=tf.float32)),
        name="origin_distance_loss"
    )
    print_obj(
        "get_origin_distance_loss",
        "origin_distance_loss",
        origin_distance_loss
    )

    return origin_distance_loss


def get_anomaly_scores(query_images, encoder_logits, encoded_images, params):
    """Gets anomaly scores from query and encoded images.

    Args:
        query_images: tensor, query image input for predictions.
        encoder_logits: tensor, encoder's logits encoded from query images.
        encoded_images: tensor, image from generator from encoder's logits.
        params: dict, user passed parameters.

    Returns:
        Predictions dictionary and export outputs dictionary.
    """
    # Get residual loss.
    residual_loss = get_residual_loss(query_images, encoded_images, params)
    print_obj("\nget_anomaly_scores", "residual_loss", residual_loss)

    # Get origin distance loss.
    origin_dist_loss = get_origin_distance_loss(encoder_logits, params)
    print_obj("get_anomaly_scores", "origin_dist_loss", origin_dist_loss)

    # Get anomaly scores.
    residual_scl = params["anom_convex_combo_factor"] * residual_loss
    origin_scl = (1. - params["anom_convex_combo_factor"]) * origin_dist_loss
    anomaly_scores = tf.add(
        x=residual_scl, y=origin_scl, name="anomaly_scores"
    )
    print_obj("get_anomaly_scores", "anomaly_scores", anomaly_scores)

    return anomaly_scores


def anomaly_detection(query_images, encoder_logits, encoded_images, params):
    """Gets anomaly scores from query and encoded images.

    Args:
        query_images: tensor, query image input for predictions.
        encoder_logits: tensor, encoder's logits encoded from query images.
        encoded_images: tensor, image from generator from encoder's logits.
        params: dict, user passed parameters.

    Returns:
        Predictions dictionary and export outputs dictionary.
    """
    # Get anomaly scores.
    anomaly_scores = get_anomaly_scores(
        query_images, encoder_logits, encoded_images, params
    )
    print_obj("\nanomaly_detection", "anomaly_scores", anomaly_scores)

    # Get anomaly flags.
    anomaly_flags = tf.cast(
        x=tf.greater(x=anomaly_scores, y=params["anomaly_threshold"]),
        dtype=tf.int32
    )
    print_obj("anomaly_detection", "anomaly_flags", anomaly_flags)

    return anomaly_scores, anomaly_flags
