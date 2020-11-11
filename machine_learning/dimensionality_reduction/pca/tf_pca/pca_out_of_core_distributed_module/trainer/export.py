import datetime
import os
import tensorflow as tf


class Export(object):
    """Class that contains methods used for exporting model objects.
    """
    def __init__(self):
        """Instantiate instance of `Export`.
        """
        pass

    def create_serving_model(self):
        """Creates Keras `Model` for serving.

        Returns:
            `tf.Keras.Model` for serving predictions.
        """
        # Create input layer for raw images.
        input_layer = tf.keras.Input(
            shape=(
                self.params["image_height"],
                self.params["image_width"],
                self.params["image_depth"]
            ),
            name="serving_inputs",
            dtype=tf.uint8
        )

        # Pass images through ResNet to get feature vectors.
        resnet_feature_vectors = (
            self.resnet_instance.get_image_resnet_feature_vectors(
                images=input_layer
            )
        )

        # Project ResNet feature vectors using PCA eigenvectors.
        pca_projections = tf.identity(
            input=self.pca_model.pca_projection_to_top_k_pc(
                data=resnet_feature_vectors
            ),
            name="pca_projections"
        )

        return tf.keras.Model(
            inputs=input_layer,
            outputs=pca_projections,
            name="serving_model"
        )

    def export_saved_model(self):
        """Exports SavedModel to output directory for serving.
        """
        # Build export path.
        export_path = os.path.join(
            self.params["output_dir"],
            "export",
            datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        )

        # Create serving models.
        serving_model = self.create_serving_model()

        # Signature will be serving_default.
        tf.saved_model.save(
            obj=serving_model,
            export_dir=export_path
        )

    def training_loop_end_save_model(self):
        """Saving model when training loop ends.
        """
        # Write final checkpoint.
        checkpoint_saved = self.checkpoint_manager.save(
            checkpoint_number=self.global_step, check_interval=False
        )

        if checkpoint_saved:
            print("Checkpoint saved at {}".format(checkpoint_saved))

        # Export SavedModel for serving.
        self.export_saved_model()
