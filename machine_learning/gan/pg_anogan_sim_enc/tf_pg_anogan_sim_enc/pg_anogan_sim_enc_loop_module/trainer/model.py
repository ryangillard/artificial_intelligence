import tensorflow as tf

from . import input
from . import serving
from . import pg_anogan_sim_enc

from .print_object import print_obj


def train_and_evaluate(args):
    """Trains and evaluates custom Estimator model.

    Args:
        args: dict, user passed parameters.

    Returns:
        `Estimator` object.
    """
    print_obj("train_and_evaluate", "args", args)
    # Set logging to be level of INFO.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Create TPU config.
    if args["use_tpu"]:
        STEPS_PER_EVAL = args["train_steps"]
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
#         print("All devices: ", tf.config.list_logical_devices('TPU'))
        config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
#             master=,
            model_dir=args["output_dir"],
            save_checkpoints_steps=STEPS_PER_EVAL,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=STEPS_PER_EVAL,
                per_host_input_for_training=True
            )
        )
#   is_per_host = contrib_tpu.InputPipelineConfig.PER_HOST_V2
#   run_config = contrib_tpu.RunConfig(
#       cluster=tpu_cluster_resolver,
#       master=FLAGS.master,
#       model_dir=FLAGS.output_dir,
#       save_checkpoints_steps=FLAGS.save_checkpoints_steps,
#       tpu_config=contrib_tpu.TPUConfig(
#           iterations_per_loop=FLAGS.iterations_per_loop,
#           num_shards=FLAGS.num_tpu_cores,
#           per_host_input_for_training=is_per_host))
    else:
        config = tf.contrib.tpu.RunConfig()
    
    if args["use_tpu"]:
        args["growth_idx"] = -1
        for i in range(500):
            if i % args["num_steps_until_growth"]:
                args["growth_idx"] += 1
            if i % 2 == 0:
                args["training_phase"] = "discriminator"
            else:
                args["training_phase"] = "encoder"

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

            print("CALLING TRAIN WITH TRAINING_PHASE {} AND GROWTH_IDX {} FOR STEP".format(args["training_phase"], args["growth_idx"]))
            estimator.train(
                input_fn=input.read_dataset(
                    filename=args["train_file_pattern"],
                    mode=tf.estimator.ModeKeys.TRAIN,
                    batch_size=args["train_batch_size"],
                    params=args
                ),
                steps=args["train_steps"]
            )

        # export similar to Cloud ML Engine / TF Serving convention
        tf.logging.info('Starting to export model.')
        estimator.export_savedmodel(
            export_dir_base=os.path.join(args["output_dir"], "export/exporter"),
            serving_input_receiver_fn=lambda: serving.serving_input_fn(args))
    else:
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
