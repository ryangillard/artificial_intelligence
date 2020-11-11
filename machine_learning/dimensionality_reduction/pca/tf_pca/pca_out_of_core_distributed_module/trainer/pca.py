import tensorflow as tf

from . import covariance


class PCA(covariance.CovarianceMatrix):
    """Class that performs PCA projection and reconstruction.

    Attributes:
        rollover_singleton_example: tensor, rank 2 tensor of shape
            (1, num_cols) containing a rollover singleton example in case the
            data batch size begins at 1. This avoids NaN covariances.
        eigenvalues: tf.Variable, rank 1 of shape (num_cols,) containing the
            eigenvalues of the covariance matrix.
        eigenvectors: tf.Variable, rank 2 of shape (num_cols, num_cols)
            containing the eigenvectors of the covariance matrix.
        top_k_eigenvectors: tensor, rank 2 tensor of shape
            (num_cols, top_k_pc) containing the eigenvectors associated with
            the top_k_pc eigenvalues.
    """
    def __init__(self, params):
        """Initializes `PCA` class instance.

        Args:
            params: dict, user passed parameters.
        """
        super().__init__(params=params)
        self.params = params

        self.rollover_singleton_example = None

        self.eigenvalues = tf.Variable(
            initial_value=tf.zeros(
                shape=(self.params["num_cols"],), dtype=tf.float32
            ),
            trainable=False
        )

        self.eigenvectors = tf.Variable(
            initial_value=tf.zeros(
                shape=(self.params["num_cols"], self.params["num_cols"]),
                dtype=tf.float32
            ),
            trainable=False
        )

        self.top_k_eigenvectors = tf.zeros(
            shape=(self.params["num_cols"], self.params["top_k_pc"])
        )

    @tf.function
    def assign_eigenvalues(self, eigenvalues):
        """Assigns covariance matrix eigenvalues tf.Variable.

        Args:
            eigenvalues: tensor, rank 1 of shape (num_cols,) containing the
            eigenvalues of the covariance matrix.
        """
        self.eigenvalues.assign(value=eigenvalues)

    @tf.function
    def assign_eigenvectors(self, eigenvectors):
        """Assigns covariance matrix eigenvectors tf.Variable.

        Args:
            eigenvectors: tensor, rank 2 of shape (num_cols, num_cols)
            containing the eigenvectors of the covariance matrix.
        """
        self.eigenvectors.assign(value=eigenvectors)

    def calculate_eigenvalues_and_eigenvectors(self):
        """Calculates eigenvalues and eigenvectors of data.
        """
        # shape = (num_cols,) & (num_cols, num_cols)
        eigenvalues, eigenvectors = tf.linalg.eigh(
            tensor=self.covariance_matrix
        )

        self.assign_eigenvalues(eigenvalues=eigenvalues)
        self.assign_eigenvectors(eigenvectors=eigenvectors)

    def pca_projection_to_top_k_pc(self, data):
        """Projects data down to top_k principal components.

        Args:
            data: tensor, rank 2 tensor of shape (num_examples, num_cols)
                containing batch of input data.

        Returns:
            Rank 2 tensor of shape (num_examples, top_k_pc) containing
                projected centered data.
        """
        # shape = (num_cols, top_k_pc)
        self.top_k_eigenvectors = (
            self.eigenvectors[:, -self.params["top_k_pc"]:]
        )

        # shape = (num_examples, num_cols)
        centered_data = data - self.col_means_vector

        # shape = (num_examples, top_k_pc)
        projected_centered_data = tf.matmul(
            a=centered_data,
            b=self.top_k_eigenvectors
        )

        return projected_centered_data

    def pca_reconstruction_from_top_k_pc(self, data):
        """Reconstructs data up from top_k principal components.

        Args:
            data: tensor, rank 2 tensor of shape (num_examples, num_cols)
                containing batch of input data.

        Returns:
            Rank 2 tensor of shape (num_examples, num_cols) containing
                lossy, reconstructed input data.
        """
        # shape = (num_examples, top_k_pc)
        projected_centered_data = self.pca_projection_to_top_k_pc(data=data)

        # shape = (num_examples, num_cols)
        unprojected_centered_data = tf.matmul(
            a=projected_centered_data,
            b=self.top_k_eigenvectors,
            transpose_b=True
        )

        # shape = (num_examples, num_cols)
        data_reconstructed = unprojected_centered_data + self.col_means_vector

        return data_reconstructed
