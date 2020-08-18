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

    def export_saved_model(self):
        """Exports SavedModel to output directory for serving.
        """
        export_path = os.path.join(
            self.params["output_dir"],
            "export",
            datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        )

        # Signature will be serving_default.
        tf.saved_model.save(
            obj=self.network_objects["generator"].models[self.growth_idx],
            export_dir=export_path
        )

    def training_loop_end_save_model(self):
        """Saving model when training loop ends.
        """
        # Write final checkpoint.
        self.checkpoint_manager.save(
            checkpoint_number=self.global_step, check_interval=False
        )

        # Export SavedModel for serving.
        self.export_saved_model()
