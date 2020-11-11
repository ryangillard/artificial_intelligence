import os
import tensorflow as tf


class Checkpoints(object):
    """Class that contains methods used for training checkpoints.
    """
    def __init__(self):
        """Instantiate instance of `Checkpoints`.
        """
        pass

    def create_checkpoint_manager(self):
        """Creates checkpoint manager for reading and writing checkpoints.
        """
        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=os.path.join(
                self.params["output_dir"], "checkpoints"
            ),
            max_to_keep=self.params["keep_checkpoint_max"],
            checkpoint_name="ckpt",
            step_counter=self.global_step,
            checkpoint_interval=self.params["save_checkpoints_steps"]
        )

    def create_checkpoint_machinery(self):
        """Creates checkpoint machinery needed to save & restore checkpoints.
        """
        # Create checkpoint instance.
        self.checkpoint = tf.train.Checkpoint(
            global_step=self.global_step,
            seen_example_count=self.pca_model.seen_example_count,
            col_means_vector=self.pca_model.col_means_vector,
            covariance_matrix=self.pca_model.covariance_matrix,
            eigenvalues=self.pca_model.eigenvalues,
            eigenvectors=self.pca_model.eigenvectors
        )

        # Create initial checkpoint manager.
        self.create_checkpoint_manager()

        # Restore any prior checkpoints.
        print(
            "Loading latest checkpoint: {}".format(
                self.checkpoint_manager.latest_checkpoint
            )
        )
        status = self.checkpoint.restore(
            save_path=self.checkpoint_manager.latest_checkpoint
        )

        if self.checkpoint_manager.latest_checkpoint:
            status.assert_consumed()
