import tensorflow as tf

from . import discriminator
from . import eval_metrics
from . import generator
from . import predict
from . import train
from . import train_and_eval
from .print_object import print_obj


def create_generator_discriminator_pair(name_suffix, params):
    """CycleGAN Unpaired Image Translation custom Estimator model function.

    Args:
        name_suffix: str, suffix to add to the end of network names.
        params: dict, user passed parameters.

    Returns:
        Instances of `Generator` and `Discriminator` classes.
    """
    # Instantiate generator.
    cyclegan_generator = generator.Generator(
        kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(
            scale_l1=params["generator_l1_regularization_scale"],
            scale_l2=params["generator_l2_regularization_scale"]
        ),
        bias_regularizer=None,
        name="generator/domain_{}".format(name_suffix)
    )

    # Instantiate discriminator.
    cyclegan_discriminator = discriminator.Discriminator(
        kernel_regularizer=tf.contrib.layers.l1_l2_regularizer(
            scale_l1=params["discriminator_l1_regularization_scale"],
            scale_l2=params["discriminator_l2_regularization_scale"]
        ),
        bias_regularizer=None,
        name="discriminator/domain_{}".format(name_suffix[-1])
    )

    return cyclegan_generator, cyclegan_discriminator


def cyclegan_model(features, labels, mode, params):
    """CycleGAN Unpaired Image Translation custom Estimator model function.

    Args:
        features: dict, keys are feature names and values are feature tensors.
        labels: tensor, label data.
        mode: tf.estimator.ModeKeys with values of either TRAIN, EVAL, or
            PREDICT.
        params: dict, user passed parameters.

    Returns:
        Instance of `tf.estimator.EstimatorSpec` class.
    """
    func_name = "cyclegan_model"
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

    # Instantiate generator and discriminator pair for domain a.
    cyclegan_generator_domain_a2b, cyclegan_discriminator_domain_b = (
        create_generator_discriminator_pair(
            name_suffix="a2b", params=params
        )
    )

    # Instantiate generator and discriminator pair for domain b.
    cyclegan_generator_domain_b2a, cyclegan_discriminator_domain_a = (
        create_generator_discriminator_pair(
            name_suffix="b2a", params=params
        )
    )

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Get predictions and export outputs.
        (predictions_dict,
         export_outputs) = predict.get_predictions_and_export_outputs(
            features=features,
            generator_domain_a2b=cyclegan_generator_domain_a2b,
            generator_domain_b2a=cyclegan_generator_domain_b2a,
            params=params
        )
    else:
        # Get logits and losses from networks for train and eval modes.
        (real_logits_domain_b,
         fake_logits_domain_b,
         generator_domain_a2b_total_loss,
         discriminator_domain_b_total_loss,
         real_logits_domain_a,
         fake_logits_domain_a,
         generator_domain_b2a_total_loss,
         discriminator_domain_a_total_loss) = (
            train_and_eval.get_logits_and_losses_combined_domains(
                features=features,
                generator_domain_a2b=cyclegan_generator_domain_a2b,
                discriminator_domain_b=cyclegan_discriminator_domain_b,
                generator_domain_b2a=cyclegan_generator_domain_b2a,
                discriminator_domain_a=cyclegan_discriminator_domain_a,
                mode=mode,
                params=params
            )
        )

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Create nested dictionary of losses.
            loss_dict = {
                "generator": {
                    "a2b": generator_domain_a2b_total_loss,
                    "b2a": generator_domain_b2a_total_loss
                },
                "discriminator": {
                    "a": discriminator_domain_a_total_loss,
                    "b": discriminator_domain_b_total_loss
                }
            }

            # Create variable and gradient histogram summaries.
            train.create_variable_and_gradient_histogram_summaries(
                loss_dict=loss_dict, params=params
            )

            # Get loss and train op for EstimatorSpec.
            loss, train_op = train.get_loss_and_train_op(
                loss_dict=loss_dict, params=params
            )
        else:
            # Set eval loss.
            loss = tf.add(
                x=discriminator_domain_a_total_loss,
                y=discriminator_domain_b_total_loss
            )

            # Get eval metrics.
            eval_metric_ops = eval_metrics.get_eval_metric_ops(
                real_logits=tf.add(
                    x=real_logits_domain_a, y=real_logits_domain_b
                ),
                fake_logits=tf.add(
                    x=fake_logits_domain_a, y=fake_logits_domain_b
                ),
                params=params
            )

    # Return EstimatorSpec
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions_dict,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs=export_outputs
    )
