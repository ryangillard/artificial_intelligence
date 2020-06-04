import tensorflow as tf

from . import input
from . import serving
from . import pgan


def train_and_evaluate(args):
    """Trains and evaluates custom Estimator model.

    Args:
        args: dict, user passed parameters.
    """
    # Set logging to be level of INFO.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Create exporter to save out the complete model to disk.
    exporter = tf.estimator.LatestExporter(
        name="exporter",
        serving_input_receiver_fn=lambda: serving.serving_input_fn(args)
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

    # Determine number of training stages.
    num_stages = min(
        args["train_steps"] // args["num_steps_until_growth"] + 1,
        len(args["conv_num_filters"])
    )

    # Train estimator for each stage.
    for i in range(num_stages):
        # Determine number of training steps from last checkpoint.
        train_steps = args["num_steps_until_growth"] * (i + 1)

        # Create train spec to read in our training data.
        train_spec = tf.estimator.TrainSpec(
            input_fn=input.read_dataset(
                filename=args["train_file_pattern"],
                mode=tf.estimator.ModeKeys.TRAIN,
                batch_size=args["train_batch_size"],
                params=args
            ),
            max_steps=train_steps
        )
        
        # Set growth index.
        args["growth_index"] = i

        print(
            "\n\n\nTRAINING MODEL FOR {} STEPS WITH GROWTH INDEX = {}".format(
                args["num_steps_until_growth"], args["growth_index"]
            )
        )

        # Instantiate estimator.
        estimator = tf.estimator.Estimator(
            model_fn=pgan.pgan_model,
            model_dir=args["output_dir"],
            params=args
        )

        # Create train and evaluate loop to train and evaluate our estimator.
        tf.estimator.train_and_evaluate(
            estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)

    # Determine if more training is needed using final stage block.
    train_steps_completed = num_stages * args["num_steps_until_growth"]
    train_steps_remaining = args["train_steps"] - train_steps_completed

    # Train for any remaining steps.
    if train_steps_remaining > 0:
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

        # Set growth index.
        args["growth_index"] = -1 if len(args["conv_num_filters"]) > 1 else 0

        print(
            "\n\n\nTRAINING MODEL MORE USING FINAL BLOCK FOR REMAINING {} STEPS AT GROWTH INDEX {}".format(
                train_steps_remaining, args["growth_index"]
            )
        )

        # Instantiate estimator.
        estimator = tf.estimator.Estimator(
            model_fn=pgan.pgan_model,
            model_dir=args["output_dir"],
            params=args
        )

        # Create train and evaluate loop to train and evaluate our estimator.
        tf.estimator.train_and_evaluate(
            estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)
