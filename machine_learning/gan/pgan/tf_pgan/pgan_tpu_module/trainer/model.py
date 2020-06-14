import os
import tensorflow as tf

from . import input
from . import serving
from . import pgan

from .print_object import print_obj


def train_and_evaluate(args):
    """Trains and evaluates custom Estimator model.

    Args:
        args: dict, user passed parameters.

    Returns:
        `Estimator` object.
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
            cluster=tpu_cluster_resolver,
            model_dir=args["output_dir"],
            save_checkpoints_steps=100,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=args["num_steps_until_growth"],
                per_host_input_for_training=True
            )
        )
    else:
        config = tf.contrib.tpu.RunConfig(save_checkpoints_steps=100)

    if args["use_tpu"]:
        # Detrmine number of stages.
        min_potential_stages = min(
            ((args["train_steps"] - 1) // args["num_steps_until_growth"]) +1,
            17
        )

        min_possible_stages = min(
            min_potential_stages, len(args["conv_num_filters"]) * 2 - 1
        )

        num_stages = min_possible_stages - 1

        # Growth phases.
        args["growth_idx"] = 0
        for i in range(num_stages):
            # Create our custom estimator using our model function.
            estimator = tf.estimator.tpu.TPUEstimator(
                model_fn=pgan.pgan_model,
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
                steps=args["num_steps_until_growth"]
            )

            # Export SavedModel.
            tf.logging.info("Starting to export model.")
            estimator.export_savedmodel(
                export_dir_base=os.path.join(
                    args["output_dir"], "export/exporter"
                ),
                serving_input_receiver_fn=lambda: serving.serving_input_fn(
                    args
                )
            )
            
            args["growth_idx"] += 1

        # Steady phase for any remaining steps.
        growth_steps = num_stages * args["num_steps_until_growth"]
        remaining_steps = args["train_steps"] - growth_steps
        if remaining_steps > 0:
            # Create our custom estimator using our model function.
            estimator = tf.estimator.tpu.TPUEstimator(
                model_fn=pgan.pgan_model,
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
                steps=remaining_steps
            )

        # Export SavedModel.
        tf.logging.info("Starting to export model.")
        estimator.export_savedmodel(
            export_dir_base=os.path.join(
                args["output_dir"], "export/exporter"
            ),
            serving_input_receiver_fn=lambda: serving.serving_input_fn(args)
        )
    else:
        # Create our custom estimator using our model function.
        estimator = tf.estimator.tpu.TPUEstimator(
            model_fn=pgan.pgan_model,
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
            serving_input_receiver_fn=lambda: serving.serving_input_fn(args),
            exports_to_keep=args["exports_to_keep"]
        )

        # Create eval spec to read in our validation data and export our model.
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

        # Create train and evaluate loop to train and evaluate our estimator.
        tf.estimator.train_and_evaluate(
            estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)
