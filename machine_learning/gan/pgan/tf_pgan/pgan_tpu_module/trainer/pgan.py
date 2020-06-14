import tensorflow as tf

from . import discriminator
from . import eval_metrics
from . import generator
from . import image_utils
from . import predict
from . import train
from . import train_and_eval
from .print_object import print_obj


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
    func_name = "pgan_model"
    print_obj("\n" + func_name, "features", features)
    print_obj(func_name, "labels", labels)
    print_obj(func_name, "mode", mode)
    print_obj(func_name, "params", params)

    # Loss function, training/eval ops, etc.
    predictions_dict = None
    loss = None
    train_op = None
    eval_metric_ops = None
    export_outputs = None

    # Instantiate generator.
    pgan_generator = generator.Generator(
        kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(
            scale_l1=params["generator_l1_regularization_scale"],
            scale_l2=params["generator_l2_regularization_scale"]
        ),
        bias_regularizer=None,
        params=params,
        name="generator"
    )

    # Instantiate discriminator.
    pgan_discriminator = discriminator.Discriminator(
        kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(
            scale_l1=params["discriminator_l1_regularization_scale"],
            scale_l2=params["discriminator_l2_regularization_scale"]
        ),
        bias_regularizer=None,
        params=params,
        name="discriminator"
    )

    # Create alpha variable to use for weighted sum for smooth fade-in.
    alpha_var = tf.get_variable(
        name="alpha_var",
        dtype=tf.float32,
        # When the initializer is a function, tensorflow can place it
        # "outside of the control flow context" to make sure it always runs.
        initializer=lambda: tf.zeros(shape=[], dtype=tf.float32),
        trainable=False
    )
    print_obj(func_name, "alpha_var", alpha_var)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Get predictions and export outputs.
        (predictions_dict,
         export_outputs) = predict.get_predictions_and_export_outputs(
            features=features,
            generator=pgan_generator,
            params=params
        )
    else:
        # Get logits and losses from networks for train and eval modes.
        (real_logits,
         fake_logits,
         generator_total_loss,
         discriminator_total_loss) = train_and_eval.get_logits_and_losses(
            features=features,
            generator=pgan_generator,
            discriminator=pgan_discriminator,
            alpha_var=alpha_var,
            params=params
        )

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Get loss and train op for EstimatorSpec.
            loss, train_op = train.get_loss_and_train_op(
                generator_total_loss=generator_total_loss,
                discriminator_total_loss=discriminator_total_loss,
                alpha_var=alpha_var,
                params=params
            )
        else:
            loss = discriminator_total_loss

            if params["use_tpu"]:
                eval_metric_ops = (
                    eval_metrics.get_eval_metric_ops,
                    {"real_logits": real_logits, "fake_logits": fake_logits}
                )
            else:
                eval_metric_ops = eval_metrics.get_eval_metric_ops(
                    real_logits, fake_logits
                )

    if params["eval_on_tpu"]:
        # Return TPUEstimatorSpec
        return tf.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions=predictions_dict,
            loss=loss,
            train_op=train_op,
            eval_metrics=eval_metric_ops,
            export_outputs=export_outputs
        )
    else:
        # Return EstimatorSpec
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions_dict,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            export_outputs=export_outputs
        )
