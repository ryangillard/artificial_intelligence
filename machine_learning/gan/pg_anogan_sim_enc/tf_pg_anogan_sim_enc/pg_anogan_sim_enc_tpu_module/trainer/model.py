import os
import tensorflow as tf

from . import input
from . import serving
from . import pg_anogan_sim_enc

from .print_object import print_obj


def instantiate_estimator(args, config):
    """Instantiates `TPUEstimator`.

    Args:
        args: dict, user passed parameters.
        config: instance of `tf.contrib.tpu.RunConfig`.

    Returns:
        `TPUEstimator` object.
    """
    # Create our custom estimator using our model function.
    estimator = tf.estimator.tpu.TPUEstimator(
        model_fn=pg_anogan_sim_enc.pg_anogan_sim_enc_model,
        model_dir=args["output_dir"],
        config=config,
        params=args,
        use_tpu=args["use_tpu"],
        train_batch_size=args["train_batch_size"],
        eval_batch_size=args["eval_batch_size"],
        eval_on_tpu=args["eval_on_tpu"],
        export_to_tpu=args["export_to_tpu"],
        export_to_cpu=args["export_to_cpu"]
    )

    return estimator


def train_estimator(args, estimator, steps):
    """Trains custom Estimator model.

    Args:
        args: dict, user passed parameters.
        estimator: instance of `TPUEstimator`.
        steps: int, number of steps to train for.
    """
    print(
        "CALLING TRAIN WITH GROWTH_IDX {}".format(args["growth_idx"])
    )
    estimator.train(
        input_fn=input.read_dataset(
            filename=args["train_file_pattern"],
            mode=tf.estimator.ModeKeys.TRAIN,
            batch_size=args["train_batch_size"],
            params=args
        ),
        steps=steps
    )


def export_saved_model(args, estimator):
    """Exports SavedModel.

    Args:
        args: dict, user passed parameters.
        estimator: instance of `TPUEstimator`.
    """
    tf.logging.info("Starting to export model.")
    estimator.export_savedmodel(
        export_dir_base=os.path.join(
            args["output_dir"], "export/exporter"
        ),
        serving_input_receiver_fn=lambda: serving.serving_input_fn(
            args
        )
    )


def train_loop_iteration(args, config, steps):
    """Performs one training loop iteration.

    Args:
        args: dict, user passed parameters.
        config: instance of `tf.contrib.tpu.RunConfig`.
        steps: int, number of steps to train for.
    """
    # Instantiate new `TPUEstimator` instance.
    estimator = instantiate_estimator(args, config)

    # Train estimator.
    train_estimator(args, estimator, steps)

    # Export SavedModel.
    export_saved_model(args, estimator)


def progressive_train_loop(args, config):
    """Progressively trains model in a loop.

    Args:
        args: dict, user passed parameters.
        config: instance of `tf.contrib.tpu.RunConfig`.
    """
    func_name = "progressive_train_loop"

    # Detrmine number of stages.
    args["growth_idx"] = 0 if not args["growth_idx"] else args["growth_idx"]
    new_stages = ((args["train_steps"] - 1) // args["num_steps_until_growth"])
    min_potential_stages = min(
        args["growth_idx"] + new_stages + 1,
        17
    )
    print_obj("\n" + func_name, "min_potential_stages", min_potential_stages)

    min_possible_stages = min(
        min_potential_stages, len(args["conv_num_filters"]) * 2 - 1
    )
    print_obj(func_name, "min_possible_stages", min_possible_stages)

    num_stages = min_possible_stages - 1
    print_obj(func_name, "num_stages", num_stages)
    # Growth phases.
    for i in range(num_stages):
        # Perfom one training loop iteration.
        train_loop_iteration(
            args, config, steps=args["num_steps_until_growth"]
        )

        args["growth_idx"] += 1

    # Steady phase for any remaining steps.
    growth_steps = num_stages * args["num_steps_until_growth"]
    print_obj(func_name, "growth_steps", growth_steps)
    remaining_steps = args["train_steps"] - growth_steps
    print_obj(func_name, "remaining_steps", remaining_steps)
    if remaining_steps > 0:
        # Perfom one training loop iteration.
        train_loop_iteration(args, config, steps=remaining_steps)


def train_and_evaluate(args):
    """Trains and evaluates custom Estimator model.

    Args:
        args: dict, user passed parameters.
    """
    print_obj("train_and_evaluate", "args", args)
    # Ensure filewriter cache is clear for TensorBoard events file.
    tf.summary.FileWriterCache.clear()

    # Set logging to be level of INFO.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Create TPU config.
    if args["use_tpu"]:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)

        # Create TPU RunConfig.
        config = tf.contrib.tpu.RunConfig(
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=args["num_steps_until_growth"],
                per_host_input_for_training=True
            ),
            cluster=tpu_cluster_resolver,
            model_dir=args["output_dir"],
            save_summary_steps=args["save_summary_steps"],
            save_checkpoints_steps=args["save_checkpoints_steps"],
            keep_checkpoint_max=args["keep_checkpoint_max"]
        )

        # Run training loop.
        progressive_train_loop(args, config)
    else:
        # Create TPU RunConfig.
        config = tf.contrib.tpu.RunConfig(
            model_dir=args["output_dir"],
            save_summary_steps=args["save_summary_steps"],
            save_checkpoints_steps=args["save_checkpoints_steps"],
            keep_checkpoint_max=args["keep_checkpoint_max"]
        )

        if args["use_estimator_train_and_evaluate"]:
            # Create our custom estimator using our model function.
            estimator = tf.estimator.tpu.TPUEstimator(
                model_fn=pg_anogan_sim_enc.pg_anogan_sim_enc_model,
                model_dir=args["output_dir"],
                config=config,
                params=args,
                use_tpu=False,
                train_batch_size=args["train_batch_size"],
                eval_batch_size=args["eval_batch_size"],
                eval_on_tpu=False,
                export_to_tpu=False,
                export_to_cpu=True
            )

            # Create train spec to read in our training data.
            train_spec = tf.estimator.TrainSpec(
                input_fn=input.read_dataset(
                    filename=args["train_file_pattern"],
                    mode=tf.estimator.ModeKeys.TRAIN,
                    batch_size=args["train_batch_size"],
                    params=args
                ),
                max_steps=args["train_steps"]
            )

            # Create exporter to save out the complete model to disk.
            exporter = tf.estimator.LatestExporter(
                name="exporter",
                serving_input_receiver_fn=lambda: serving.serving_input_fn(
                    args
                ),
                exports_to_keep=args["exports_to_keep"]
            )

            # Create eval spec to read validation data and export our model.
            eval_spec = tf.estimator.EvalSpec(
                input_fn=input.read_dataset(
                    filename=args["eval_file_pattern"],
                    mode=tf.estimator.ModeKeys.EVAL,
                    batch_size=args["eval_batch_size"],
                    params=args
                ),
                steps=args["eval_steps"],
                start_delay_secs=args["start_delay_secs"],
                throttle_secs=args["throttle_secs"],
                exporters=exporter
            )

            # Create train and evaluate loop to train & evaluate our estimator.
            tf.estimator.train_and_evaluate(
                estimator=estimator,
                train_spec=train_spec,
                eval_spec=eval_spec
            )
        else:
            progressive_train_loop(args, config)
