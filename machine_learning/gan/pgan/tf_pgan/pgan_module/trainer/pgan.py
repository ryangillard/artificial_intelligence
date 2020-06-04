import tensorflow as tf

from . import discriminator
from . import generator
from .print_object import print_obj


def train_network(loss, global_step, alpha_var, params, scope):
    """Trains network and returns loss and train op.

    Args:
        loss: tensor, shape of [].
        global_step: tensor, the current training step or batch in the
            training loop.
        alpha_var: variable, alpha for weighted sum of fade-in of layers.
        params: dict, user passed parameters.
        scope: str, the variables that to train.

    Returns:
        Loss tensor and training op.
    """
    # Create optimizer map.
    optimizers = {
        "Adam": tf.train.AdamOptimizer,
        "Adadelta": tf.train.AdadeltaOptimizer,
        "AdagradDA": tf.train.AdagradDAOptimizer,
        "Adagrad": tf.train.AdagradOptimizer,
        "Ftrl": tf.train.FtrlOptimizer,
        "GradientDescent": tf.train.GradientDescentOptimizer,
        "Momentum": tf.train.MomentumOptimizer,
        "ProximalAdagrad": tf.train.ProximalAdagradOptimizer,
        "ProximalGradientDescent": tf.train.ProximalGradientDescentOptimizer,
        "RMSProp": tf.train.RMSPropOptimizer
    }

    # Get gradients.
    gradients = tf.gradients(
        ys=loss,
        xs=tf.trainable_variables(scope=scope),
        name="{}_gradients".format(scope)
    )

    # Clip gradients.
    if params["{}_clip_gradients".format(scope)]:
        gradients, _ = tf.clip_by_global_norm(
            t_list=gradients,
            clip_norm=params["{}_clip_gradients".format(scope)],
            name="{}_clip_by_global_norm_gradients".format(scope)
        )

    # Zip back together gradients and variables.
    grads_and_vars = zip(gradients, tf.trainable_variables(scope=scope))

    # Get optimizer and instantiate it.
    optimizer = optimizers[params["{}_optimizer".format(scope)]](
        learning_rate=params["{}_learning_rate".format(scope)]
    )

    # Create train op by applying gradients to variables and incrementing
    # global step.
    train_op = optimizer.apply_gradients(
        grads_and_vars=grads_and_vars,
        global_step=global_step,
        name="{}_apply_gradients".format(scope)
    )

    # Update alpha variable to linearly scale from 0 to 1 based on steps.
    alpha_var_update_op = tf.assign(
        ref=alpha_var,
        value=tf.divide(
            x=tf.cast(
                x=tf.mod(x=global_step, y=params["num_steps_until_growth"]),
                dtype=tf.float32
            ),
            y=params["num_steps_until_growth"]
        )
    )

    # Ensure alpha variable gets updated.
    with tf.control_dependencies(control_inputs=[alpha_var_update_op]):
        loss = tf.identity(input=loss, name="train_network_loss_identity")

    return loss, train_op


def resize_real_image(block_idx, image, params):
    """Resizes real images to match the GAN's current size.

    Args:
        block_idx: int, index of current block.
        image: tensor, original image.
        params: dict, user passed parameters.

    Returns:
        Resized image tensor.
    """
    print_obj("\nresize_real_image", "block_idx", block_idx)
    print_obj("resize_real_image", "image", image)

    # Resize image to match GAN size at current block index.
    resized_image = tf.image.resize(
        images=image,
        size=[
            params["generator_projection_dims"][0] * (2 ** block_idx),
            params["generator_projection_dims"][1] * (2 ** block_idx)
        ],
        method="nearest",
        name="resize_real_images_resized_image_{}".format(block_idx)
    )
    print_obj("resize_real_images", "resized_image", resized_image)

    return resized_image


def resize_real_images(image, params):
    """Resizes real images to match the GAN's current size.

    Args:
        image: tensor, original image.
        params: dict, user passed parameters.

    Returns:
        Resized image tensor.
    """
    print_obj("\nresize_real_images", "image", image)
    # Resize real image for each block.
    num_stages = params["train_steps"] // params["num_steps_until_growth"]
    if (num_stages <= 0 or len(params["conv_num_filters"]) == 1):
        print(
            "\nresize_real_images: NEVER GOING TO GROW, SKIP SWITCH CASE!"
        )
        # If we never are going to grow, no sense using the switch case.
        resized_image = resize_real_image(0, image, params)  # 4x4
    else:
        # Find growth index based on global step and growth frequency.
        growth_index = tf.cast(
            x=tf.floordiv(
                x=tf.train.get_or_create_global_step(),
                y=params["num_steps_until_growth"],
                name="resize_real_images_global_step_floordiv"
            ),
            dtype=tf.int32,
            name="resize_real_images_growth_index"
        )

        # Switch to case based on number of steps for resized image.
        resized_image = tf.switch_case(
            branch_index=growth_index,
            branch_fns=[
                lambda: resize_real_image(0, image, params),  # 4x4
                lambda: resize_real_image(1, image, params),  # 8x8
                lambda: resize_real_image(2, image, params),  # 16x16
                lambda: resize_real_image(3, image, params),  # 32x32
                lambda: resize_real_image(4, image, params),  # 64x64
                lambda: resize_real_image(5, image, params),  # 128x128
                lambda: resize_real_image(6, image, params),  # 256x256
                lambda: resize_real_image(7, image, params),  # 512x512
                lambda: resize_real_image(8, image, params),  # 1024x1024
            ],
            name="resize_real_images_switch_case_resized_image"
        )
        print_obj(
            "resize_real_images", "selected resized_image", resized_image
        )

    return resized_image


def pgan_model(features, labels, mode, params):
    """Progressively Growing GAN custom Estimator model function.

    Args:
        features: dict, keys are feature names and values are feature tensors.
        labels: tensor, label data.
        mode: tf.estimator.ModeKeys with values of either TRAIN, EVAL, or
            PREDICT.
        params: dict, user passed parameters.

    Returns:
        Instance of `tf.estimator.EstimatorSpec` class.
    """
    print_obj("\npgan_model", "features", features)
    print_obj("pgan_model", "labels", labels)
    print_obj("pgan_model", "mode", mode)
    print_obj("pgan_model", "params", params)

    # Loss function, training/eval ops, etc.
    predictions_dict = None
    loss = None
    train_op = None
    eval_metric_ops = None
    export_outputs = None

    # Create alpha variable to use for weighted sum for smooth fade-in.
    alpha_var = tf.get_variable(
        name="alpha_var",
        dtype=tf.float32,
        initializer=tf.zeros(shape=[], dtype=tf.float32),
        trainable=False
    )
    print_obj("pgan_model", "alpha_var", alpha_var)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Extract given latent vectors from features dictionary.
        Z = tf.cast(x=features["Z"], dtype=tf.float32)

        # Get predictions from generator.
        generated_images = generator.generator_network(
            Z=Z, alpha_var=alpha_var, params=params
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
    else:
        # Extract image from features dictionary.
        X = features["image"]

        # Get dynamic batch size in case of partial batch.
        cur_batch_size = tf.shape(
            input=X,
            out_type=tf.int32,
            name="pgan_model_cur_batch_size"
        )[0]

        # Create random noise latent vector for each batch example.
        Z = tf.random.normal(
            shape=[cur_batch_size, params["latent_size"]],
            mean=0.0,
            stddev=1.0,
            dtype=tf.float32
        )

        # Get generated image from generator network from gaussian noise.
        print("\nCall generator with Z = {}.".format(Z))
        generator_outputs = generator.generator_network(
            Z=Z, alpha_var=alpha_var, params=params
        )

        # Get fake logits from discriminator using generator's output image.
        print("\nCall discriminator with generator_outputs = {}.".format(
            generator_outputs
        ))

        fake_logits = discriminator.discriminator_network(
            X=generator_outputs, alpha_var=alpha_var, params=params
        )
        
        # Resize real images based on the current size of the GAN.
        real_image = resize_real_images(X, params)

        # Get real logits from discriminator using real image.
        print("\nCall discriminator with real_image = {}.".format(
            real_image
        ))

        real_logits = discriminator.discriminator_network(
            X=real_image, alpha_var=alpha_var, params=params
        )

        # Get generator total loss.
        generator_total_loss = generator.get_generator_loss(
            fake_logits=fake_logits, params=params
        )

        # Get discriminator total loss.
        discriminator_total_loss = discriminator.get_discriminator_loss(
            fake_logits=fake_logits, real_logits=real_logits, params=params
        )

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Get global step.
            global_step = tf.train.get_or_create_global_step()

            # Determine if it is time to train generator or discriminator.
            cycle_step = tf.mod(
                x=global_step,
                y=tf.cast(
                    x=tf.add(
                        x=params["generator_train_steps"],
                        y=params["discriminator_train_steps"]
                    ),
                    dtype=tf.int64
                ),
                name="pgan_model_cycle_step"
            )

            # Create choose generator condition.
            condition = tf.less(
                x=cycle_step, y=params["generator_train_steps"]
            )

            # Needed for batch normalization, but has no effect otherwise.
            update_ops = tf.get_collection(key=tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(control_inputs=update_ops):
                # Conditionally choose to train generator or discriminator.
                loss, train_op = tf.cond(
                    pred=condition,
                    true_fn=lambda: train_network(
                        loss=generator_total_loss,
                        global_step=global_step,
                        alpha_var=alpha_var,
                        params=params,
                        scope="generator"
                    ),
                    false_fn=lambda: train_network(
                        loss=discriminator_total_loss,
                        global_step=global_step,
                        alpha_var=alpha_var,
                        params=params,
                        scope="discriminator"
                    )
                )
        else:
            loss = discriminator_total_loss

            # Concatenate discriminator logits and labels.
            discriminator_logits = tf.concat(
                values=[real_logits, fake_logits],
                axis=0,
                name="discriminator_concat_logits"
            )

            discriminator_labels = tf.concat(
                values=[
                    tf.ones_like(tensor=real_logits),
                    tf.zeros_like(tensor=fake_logits)
                ],
                axis=0,
                name="discriminator_concat_labels"
            )

            # Calculate discriminator probabilities.
            discriminator_probabilities = tf.nn.sigmoid(
                x=discriminator_logits, name="discriminator_probabilities"
            )

            # Create eval metric ops dictionary.
            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(
                    labels=discriminator_labels,
                    predictions=discriminator_probabilities,
                    name="pgan_model_accuracy"
                ),
                "precision": tf.metrics.precision(
                    labels=discriminator_labels,
                    predictions=discriminator_probabilities,
                    name="pgan_model_precision"
                ),
                "recall": tf.metrics.recall(
                    labels=discriminator_labels,
                    predictions=discriminator_probabilities,
                    name="pgan_model_recall"
                ),
                "auc_roc": tf.metrics.auc(
                    labels=discriminator_labels,
                    predictions=discriminator_probabilities,
                    num_thresholds=200,
                    curve="ROC",
                    name="pgan_model_auc_roc"
                ),
                "auc_pr": tf.metrics.auc(
                    labels=discriminator_labels,
                    predictions=discriminator_probabilities,
                    num_thresholds=200,
                    curve="PR",
                    name="pgan_model_auc_pr"
                )
            }

    # Return EstimatorSpec
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs=export_outputs
    )
