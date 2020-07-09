import datetime
import os
import tensorflow as tf

from . import input
from . import vanilla_gan
from . import train


def distributed_train_step(
    strategy,
    global_batch_size,
    features,
    network_dict,
    optimizer_dict,
    params,
    global_step,
    summary_file_writer
):
    """Perform one distributed train step.

    Args:
        strategy: instance of `tf.distribute.Strategy`.
        global_batch_size: int, global batch size for distribution.
        features: dict, feature tensors from input function.
        network_dict: dict, dictionary of network objects.
        optimizer_dict: dict, dictionary of optimizer objects.
        params: dict, user passed parameters.
        global_step: int, current global step for training.
        summary_file_writer: summary file writer.

    Returns:
        Reduced loss tensor for chosen network across replicas.
    """
    if params["tf_version"] > 2.1:
        per_replica_losses = strategy.run(
            fn=train.train_step,
            kwargs={
                "global_batch_size": global_batch_size,
                "features": features,
                "network_dict": network_dict,
                "optimizer_dict": optimizer_dict,
                "params": params,
                "global_step": global_step,
                "summary_file_writer": summary_file_writer
            }
        )
    else:
        per_replica_losses = strategy.experimental_run_v2(
            fn=train.train_step,
            kwargs={
                "global_batch_size": global_batch_size,
                "features": features,
                "network_dict": network_dict,
                "optimizer_dict": optimizer_dict,
                "params": params,
                "global_step": global_step,
                "summary_file_writer": summary_file_writer
            }
        )

    return strategy.reduce(
        reduce_op=tf.distribute.ReduceOp.SUM,
        value=per_replica_losses,
        axis=None
    )


def train_and_evaluate(args):
    """Trains and evaluates Keras model.

    Args:
        args: dict, user passed parameters.
    """
    # If the list of devices is not specified in the
    # `tf.distribute.MirroredStrategy` constructor, it will be auto-detected.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices = {}".format(strategy.num_replicas_in_sync))

    # Get input datasets. Batch size will be split evenly between replicas.
    train_dataset = input.read_dataset(
        filename=args["train_file_pattern"],
        batch_size=args["train_batch_size"] * strategy.num_replicas_in_sync,
        params=args,
        training=True
    )()

    eval_dataset = input.read_dataset(
        filename=args["eval_file_pattern"],
        batch_size=args["eval_batch_size"] * strategy.num_replicas_in_sync,
        params=args,
        training=False
    )()
    if args["eval_steps"]:
        eval_dataset = eval_dataset.take(count=args["eval_steps"])

    with strategy.scope():
        # Create distributed datasets.
        train_dist_dataset = strategy.experimental_distribute_dataset(
            dataset=train_dataset
        )
        eval_dist_dataset = strategy.experimental_distribute_dataset(
            dataset=eval_dataset
        )

        # Create iterators of distributed datasets.
        train_dist_iter = iter(train_dist_dataset)
        eval_dist_iter = iter(eval_dist_dataset)

        steps_per_epoch = args["train_dataset_length"] // args["train_batch_size"]

        # Instantiate model objects.
        network_dict, optimizer_dict = vanilla_gan.vanilla_gan_model(params=args)

        # Create checkpoint instance.
        checkpoint_dir = os.path.join(args["output_dir"], "checkpoints")
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(
            generator_model=network_dict["generator"].get_model(),
            discriminator_model=network_dict["discriminator"].get_model(),
            generator_optimizer=optimizer_dict["generator"],
            discriminator_optimizer=optimizer_dict["discriminator"]
        )

        # Create checkpoint manager.
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=checkpoint,
            directory=checkpoint_dir,
            max_to_keep=args["keep_checkpoint_max"]
        )

        # Restore any prior checkpoints.
        status = checkpoint.restore(
            save_path=checkpoint_manager.latest_checkpoint
        )

        # Create summary file writer.
        summary_file_writer = tf.summary.create_file_writer(
            logdir=os.path.join(args["output_dir"], "summaries"),
            name="summary_file_writer"
        )

        # Loop over datasets to perform training.
        global_step = 0
        for epoch in range(args["num_epochs"]):
            for epoch_step in range(steps_per_epoch):
                features, labels = next(train_dist_iter)

                loss = distributed_train_step(
                    strategy=strategy,
                    global_batch_size=(
                        args["train_batch_size"] * strategy.num_replicas_in_sync
                    ),
                    features=features,
                    network_dict=network_dict,
                    optimizer_dict=optimizer_dict,
                    params=args,
                    global_step=global_step,
                    summary_file_writer=summary_file_writer
                )

                if global_step % args["log_step_count_steps"] == 0:
                    print(
                        "epoch = {}, global_step = {}, epoch_step = {}, loss = {}".format(
                            epoch, global_step, epoch_step, loss
                        )
                    )
                global_step += 1

            # Checkpoint model every so many steps.
            if global_step % args["save_checkpoints_steps"] == 0:
                checkpoint_manager.save(checkpoint_number=global_step)

        # Write final checkpoint.
        checkpoint_manager.save(checkpoint_number=global_step)

        # Export SavedModel for serving.
        export_path = os.path.join(
            args["output_dir"],
            "export",
            datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        )

        # Signature will be serving_default.
        tf.saved_model.save(
            obj=network_dict["generator"].get_model(), export_dir=export_path
        )
