import tensorflow as tf


class ResNet(object):
    """Class that contains methods that preprocess images through ResNet.

    Attributes:
        params: dict, user passed parameters.
    """
    def __init__(self, params):
        """Initializes `ResNet` class instance.

        Args:
            params: dict, user passed parameters.
        """
        self.params = params

        self.resnet_model, self.pooling_layer = self.get_resnet_layers(
            input_shape=(
                self.params["image_height"],
                self.params["image_width"],
                self.params["image_depth"]
            )
        )

    def get_resnet_layers(self, input_shape):
        """Gets ResNet layers from ResNet50 model.

        Args:
            input_shape: tuple, input shape of images.
        """
        # Load the ResNet50 model.
        resnet50_model = tf.keras.applications.resnet50.ResNet50(
            include_top=False,
            weights=self.params["resnet_weights"],
            input_shape=input_shape
        )
        resnet50_model.trainable = False

        # Create a new Model based on original resnet50 model ended after the
        # chosen residual block.
        layer_name = self.params["resnet_layer_name"]
        resnet50 = tf.keras.Model(
            inputs=resnet50_model.input,
            outputs=resnet50_model.get_layer(layer_name).output
        )

        # Add adaptive mean-spatial pooling after the new model.
        adaptive_mean_spatial_layer = tf.keras.layers.GlobalAvgPool2D()

        return resnet50, adaptive_mean_spatial_layer

    def preprocess_image_batch(self, images):
        """Preprocesses batch of images.

        Args:
            images: tensor, rank 4 image tensor of shape
                (batch_size, image_height, image_width, image_depth).

        Returns:
            Preprocessed images tensor.
        """
        images = tf.cast(x=images, dtype=tf.float32)

        if self.params["preprocess_input"]:
            images = tf.keras.applications.resnet50.preprocess_input(x=images)

        return images

    def get_image_resnet_feature_vectors(self, images):
        """Gets image ResNet feature vectors.

        Args:
            images: tensor, rank 4 image tensor of shape
                (batch_size, image_height, image_width, image_depth).

        Returns:
            Processed ResNet feature rank 1 tensor for each image.
        """
        preprocessed_images = self.preprocess_image_batch(images=images)
        resnet_feature_image = self.resnet_model(inputs=preprocessed_images)
        resnet_feature_vector = self.pooling_layer(inputs=resnet_feature_image)

        return resnet_feature_vector
