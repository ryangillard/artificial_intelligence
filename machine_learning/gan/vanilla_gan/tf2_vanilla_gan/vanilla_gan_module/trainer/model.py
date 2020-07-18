import tensorflow as tf

from . import discriminators
from . import generators
from . import inputs


class TrainAndEvaluateLoop(object):
    """Train and evaluate loop trainer.

    Fields:
        params: dict, user passed parameters.
        network_objects: dict, instances of `Generator` and `Discriminator`
            network objects.
        network_models: dict, instances of Keras `Model`s for each network.
        optimizers: dict, instances of Keras `Optimizer`s for each network.
        strategy: instance of tf.distribute.strategy.
        global_batch_size: int, the global batch size after summing batch
            sizes across replicas.
        global_step: int, the global step counter across epochs and steps
            within epoch.
        summary_file_writer: instance of tf.summary.create_file_writer for
            summaries for TensorBoard.
    """
    def __init__(self, params):
        """Instantiate trainer.

        Args:
            params: dict, user passed parameters.
        """
        self.params = params

        self.network_objects = {}
        self.network_models = {}
        self.optimizers = {}

        self.strategy = None

        self.global_batch_size = None
        self.global_step = None
        self.summary_file_writer = None

    # train_and_eval.py
    def generator_loss_phase(self, mode, training):
        """Gets fake logits and loss for generator.

        Args:
            mode: str, what mode currently in: TRAIN or EVAL.
            training: bool, if model should be training.

        Returns:
            Fake logits of shape [batch_size, 1] and generator loss.
        """
        batch_size = (
            self.params["train_batch_size"]
            if mode == "TRAIN"
            else self.params["eval_batch_size"]
        )

        # Create random noise latent vector for each batch example.
        Z = tf.random.normal(
            shape=[batch_size, self.params["latent_size"]],
            mean=0.0,
            stddev=1.0,
            dtype=tf.float32
        )

        # Get generated image from generator network from gaussian noise.
        fake_images = self.network_models["generator"](
            inputs=Z, training=training
        )

        if mode == "TRAIN":
            # Add summaries for TensorBoard.
            with self.summary_file_writer.as_default():
                with tf.summary.record_if(
                    self.global_step % self.params["save_summary_steps"] == 0
                ):
                    tf.summary.image(
                        name="fake_images",
                        data=tf.reshape(
                            tensor=fake_images,
                            shape=[
                                -1,
                                self.params["height"],
                                self.params["width"],
                                self.params["depth"]
                            ]
                        ),
                        step=self.global_step,
                        max_outputs=5,
                    )
                    self.summary_file_writer.flush()

        # Get fake logits from discriminator using generator's output image.
        fake_logits = self.network_models["discriminator"](
            inputs=fake_images, training=False
        )

        # Get generator total loss.
        generator_total_loss = (
            self.network_objects["generator"].get_generator_loss(
                global_batch_size=self.global_batch_size,
                fake_logits=fake_logits,
                global_step=self.global_step,
                summary_file_writer=self.summary_file_writer
            )
        )

        return fake_logits, generator_total_loss

    def discriminator_loss_phase(self, real_images, fake_logits, training):
        """Gets real logits and loss for discriminator.

        Args:
            real_images: tensor, real images of shape
                [batch_size, height * width * depth].
            fake_logits: tensor, discriminator logits of fake images of shape
                [batch_size, 1].
            training: bool, if in training mode.

        Returns:
            Real logits and discriminator loss.
        """
        # Get real logits from discriminator using real image.
        real_logits = self.network_models["discriminator"](
            inputs=real_images, training=training
        )

        # Get discriminator total loss.
        discriminator_total_loss = (
            self.network_objects["discriminator"].get_discriminator_loss(
                global_batch_size=self.global_batch_size,
                fake_logits=fake_logits,
                real_logits=real_logits,
                global_step=self.global_step,
                summary_file_writer=self.summary_file_writer
            )
        )

        return real_logits, discriminator_total_loss

    # train.py
    def get_variables_and_gradients(self, loss, gradient_tape, scope):
        """Gets variables and gradients from model wrt. loss.

        Args:
            loss: tensor, shape of [].
            gradient_tape: instance of `GradientTape`.
            scope: str, the name of the network of interest.

        Returns:
            Lists of network's variables and gradients.
        """
        # Get trainable variables.
        variables = self.network_models[scope].trainable_variables

        # Get gradients from gradient tape.
        gradients = gradient_tape.gradient(
            target=loss, sources=variables
        )

        # Clip gradients.
        if self.params["{}_clip_gradients".format(scope)]:
            gradients, _ = tf.clip_by_global_norm(
                t_list=gradients,
                clip_norm=params["{}_clip_gradients".format(scope)],
                name="{}_clip_by_global_norm_gradients".format(scope)
            )

        # Add variable names back in for identification.
        gradients = [
            tf.identity(
                input=g,
                name="{}_{}_gradients".format(scope, v.name[:-2])
            )
            if tf.is_tensor(x=g) else g
            for g, v in zip(gradients, variables)
        ]

        return variables, gradients

    def get_generator_loss_variables_and_gradients(self):
        """Gets generator's loss, variables, and gradients.

        Returns:
            Generator's loss, variables, and gradients.
        """
        with tf.GradientTape() as generator_tape:
            # Get generator loss.
            _, generator_loss = self.generator_loss_phase(
                mode="TRAIN", training=True
            )

        # Get variables and gradients from generator wrt. loss.
        variables, gradients = self.get_variables_and_gradients(
            loss=generator_loss,
            gradient_tape=generator_tape,
            scope="generator"
        )

        return generator_loss, variables, gradients

    def get_discriminator_loss_variables_and_gradients(self, real_images):
        """Gets discriminator's loss, variables, and gradients.

        Args:
            real_images: tensor, real images of shape
                [batch_size, height * width * depth].

        Returns:
            Discriminator's loss, variables, and gradients.
        """
        with tf.GradientTape() as discriminator_tape:
            # Get fake logits from generator.
            fake_logits, _ = self.generator_loss_phase(
                mode="TRAIN", training=False
            )

            # Get discriminator loss.
            _, discriminator_loss = self.discriminator_loss_phase(
                real_images, fake_logits, training=True
            )

        # Get variables and gradients from discriminator wrt. loss.
        variables, gradients = self.get_variables_and_gradients(
            loss=discriminator_loss,
            gradient_tape=discriminator_tape,
            scope="discriminator"
        )

        return discriminator_loss, variables, gradients

    def create_variable_and_gradient_histogram_summaries(
        self, variables, gradients, scope
    ):
        """Creates variable and gradient histogram summaries.

        Args:
            variables: list, network's trainable variables.
            gradients: list, gradients of network's trainable variables wrt.
                loss.
            params: dict, user passed parameters.
            global_step: int, current global step for training.
            summary_file_writer: summary file writer.
            scope: str, the name of the network of interest.
        """
        # Add summaries for TensorBoard.
        with self.summary_file_writer.as_default():
            with tf.summary.record_if(
                self.global_step % self.params["save_summary_steps"] == 0
            ):
                for v, g in zip(variables, gradients):
                    tf.summary.histogram(
                        name="{}_variables/{}".format(scope, v.name[:-2]),
                        data=v,
                        step=self.global_step
                    )
                    if tf.is_tensor(x=g):
                        tf.summary.histogram(
                            name="{}_gradients/{}".format(scope, v.name[:-2]),
                            data=g,
                            step=self.global_step
                        )
                self.summary_file_writer.flush()

    def get_select_loss_variables_and_gradients(self, real_images, scope):
        """Gets selected network's loss, variables, and gradients.

        Args:
            real_images: tensor, real images of shape
                [batch_size, height * width * depth].
            scope: str, the name of the network of interest.

        Returns:
            Selected network's loss, variables, and gradients.
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            # Get fake logits from generator.
            fake_logits, generator_loss = self.generator_loss_phase(
                mode="TRAIN",
                training=(scope == "generator")
            )

            # Get discriminator loss.
            _, discriminator_loss = self.discriminator_loss_phase(
                real_images, fake_logits, training=(scope == "discriminator")
            )

        # Create empty dicts to hold loss, variables, gradients.
        loss_dict = {}
        vars_dict = {}
        grads_dict = {}

        # Loop over generator and discriminator.
        for (loss, gradient_tape, scope) in zip(
            [generator_loss, discriminator_loss],
            [gen_tape, dis_tape],
            ["generator", "discriminator"]
        ):
            # Get variables and gradients from generator wrt. loss.
            variables, gradients = self.get_variables_and_gradients(
                loss, gradient_tape, scope
            )

            # Add loss, variables, and gradients to dictionaries.
            loss_dict[scope] = loss
            vars_dict[scope] = variables
            grads_dict[scope] = gradients

            # Create variable and gradient histogram summaries.
            self.create_variable_and_gradient_histogram_summaries(
                variables, gradients, scope
            )

        return loss_dict[scope], vars_dict[scope], grads_dict[scope]

    def train_network(self, variables, gradients, scope):
        """Trains network variables using gradients with optimizer.

        Args:
            variables: list, network's trainable variables.
            gradients: list, gradients of network's trainable variables wrt.
                loss.
            scope: str, the name of the network of interest.
        """
        # Zip together gradients and variables.
        grads_and_vars = zip(gradients, variables)

        # Applying gradients to variables using optimizer.
        self.optimizers[scope].apply_gradients(grads_and_vars=grads_and_vars)

    def train_discriminator(self, features):
        """Trains discriminator network.

        Args:
            global_batch_size: int, global batch size for distribution.
            features: dict, feature tensors from input function.
            generator: instance of `Generator`.
            discriminator: instance of `Discriminator`.
            discriminator_optimizer: instance of `Optimizer`, discriminator's
                optimizer.
            params: dict, user passed parameters.
            global_step: int, current global step for training.
            summary_file_writer: summary file writer.

        Returns:
            Discriminator loss tensor.
        """
        # Extract real images from features dictionary.
        real_images = tf.reshape(
            tensor=features["image"],
            shape=[
                -1,
                self.params["height"] * self.params["width"] * self.params["depth"]
            ]
        )

        # Get gradients for training by running inputs through networks.
        if self.global_step % self.params["save_summary_steps"] == 0:
            # More computation, but needed for ALL histogram summaries.
            loss, variables, gradients = (
                self.get_select_loss_variables_and_gradients(
                    real_images, scope="discriminator"
                )
            )
        else:
            # More efficient computation.
            loss, variables, gradients = (
                self.get_discriminator_loss_variables_and_gradients(
                    real_images
                )
            )

        # Train discriminator network.
        self.train_network(variables, gradients, scope="discriminator")

        return loss

    def train_generator(self, features):
        """Trains generator network.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Generator loss tensor.
        """
        # Get gradients for training by running inputs through networks.
        if self.global_step % self.params["save_summary_steps"] == 0:
            # Extract real images from features dictionary.
            real_images = tf.reshape(
                tensor=features["image"],
                shape=[
                    -1,
                    self.params["height"] * self.params["width"] * self.params["depth"]
                ]
            )

            # More computation, but needed for ALL histogram summaries.
            loss, variables, gradients = (
                self.get_select_loss_variables_and_gradients(
                    real_images, scope="generator"
                )
            )
        else:
            # More efficient computation.
            loss, variables, gradients = (
                self.get_generator_loss_variables_and_gradients()
            )

        # Train generator network.
        self.train_network(variables, gradients, scope="generator")

        return loss

    def train_step(self, features):
        """Perform one train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Loss tensor for chosen network.
        """
        # Determine if it is time to train generator or discriminator.
        cycle_step = self.global_step % (
            self.params["discriminator_train_steps"] + self.params["generator_train_steps"]
        )

        # Conditionally choose to train generator or discriminator subgraph.
        if cycle_step < self.params["discriminator_train_steps"]:
            loss = self.train_discriminator(features=features)
        else:
            loss = self.train_generator(features=features)

        return loss

    # vanilla_gan.py
    def instantiate_network_objects(self):
        """Instantiates generator and discriminator with parameters.
        """
        # Instantiate generator.
        self.network_objects["generator"] = generators.Generator(
            input_shape=(self.params["latent_size"]),
            kernel_regularizer=tf.keras.regularizers.l1_l2(
                l1=self.params["generator_l1_regularization_scale"],
                l2=self.params["generator_l2_regularization_scale"]
            ),
            bias_regularizer=None,
            name="generator",
            params=self.params
        )

        # Instantiate discriminator.
        self.network_objects["discriminator"] = discriminators.Discriminator(
            input_shape=(
                self.params["height"] * self.params["width"] * self.params["depth"]
            ),
            kernel_regularizer=tf.keras.regularizers.l1_l2(
                l1=self.params["discriminator_l1_regularization_scale"],
                l2=self.params["discriminator_l2_regularization_scale"]
            ),
            bias_regularizer=None,
            name="discriminator",
            params=self.params
        )

    def instantiate_optimizer(self, scope):
        """Instantiates optimizer with parameters.

        Args:
            scope: str, the name of the network of interest.
        """
        # Create optimizer map.
        optimizers = {
            "Adadelta": tf.keras.optimizers.Adadelta,
            "Adagrad": tf.keras.optimizers.Adagrad,
            "Adam": tf.keras.optimizers.Adam,
            "Adamax": tf.keras.optimizers.Adamax,
            "Ftrl": tf.keras.optimizers.Ftrl,
            "Nadam": tf.keras.optimizers.Nadam,
            "RMSprop": tf.keras.optimizers.RMSprop,
            "SGD": tf.keras.optimizers.SGD
        }

        # Get optimizer and instantiate it.
        if self.params["{}_optimizer".format(scope)] == "Adam":
            optimizer = optimizers[self.params["{}_optimizer".format(scope)]](
                learning_rate=self.params["{}_learning_rate".format(scope)],
                beta_1=self.params["{}_adam_beta1".format(scope)],
                beta_2=self.params["{}_adam_beta2".format(scope)],
                epsilon=self.params["{}_adam_epsilon".format(scope)],
                name="{}_{}_optimizer".format(
                    scope, self.params["{}_optimizer".format(scope)].lower()
                )
            )
        else:
            optimizer = optimizers[self.params["{}_optimizer".format(scope)]](
                learning_rate=self.params["{}_learning_rate".format(scope)],
                name="{}_{}_optimizer".format(
                    scope, self.params["{}_optimizer".format(scope)].lower()
                )
            )

        self.optimizers[scope] = optimizer

    def vanilla_gan_model(self):
        """Vanilla GAN custom Estimator model function.
        """
        # Instantiate generator and discriminator objects.
        self.instantiate_network_objects()

        self.network_models["generator"] = (
            self.network_objects["generator"].get_model()
        )

        self.network_models["discriminator"] = (
            self.network_objects["discriminator"].get_model()
        )

        # Instantiate generator optimizer.
        self.instantiate_optimizer(scope="generator")

        # Instantiate discriminator optimizer.
        self.instantiate_optimizer(scope="discriminator")

    # model.py
    def get_train_eval_datasets(self, num_replicas):
        """Gets train and eval datasets.

        Args:
            num_replicas: int, number of device replicas.

        Returns:
            Train and eval datasets.
        """
        train_dataset = inputs.read_dataset(
            filename=self.params["train_file_pattern"],
            batch_size=self.params["train_batch_size"] * num_replicas,
            params=self.params,
            training=True
        )()

        eval_dataset = inputs.read_dataset(
            filename=self.params["eval_file_pattern"],
            batch_size=self.params["eval_batch_size"] * num_replicas,
            params=self.params,
            training=False
        )()
        if self.params["eval_steps"]:
            eval_dataset = eval_dataset.take(count=self.params["eval_steps"])

        return train_dataset, eval_dataset

    def create_checkpoint_machinery(self):
        """Creates checkpoint machinery needed to save & restore checkpoints.

        Returns:
            Instance of `CheckpointManager`.
        """
        # Create checkpoint instance.
        checkpoint_dir = os.path.join(
            self.params["output_dir"], "checkpoints"
        )
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(
            generator_model=self.network_models["generator"],
            discriminator_model=self.network_models["discriminator"],
            generator_optimizer=self.optimizers["generator"],
            discriminator_optimizer=self.optimizers["discriminator"]
        )

        # Create checkpoint manager.
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=checkpoint,
            directory=checkpoint_dir,
            max_to_keep=self.params["keep_checkpoint_max"]
        )

        # Restore any prior checkpoints.
        status = checkpoint.restore(
            save_path=checkpoint_manager.latest_checkpoint
        )

        return checkpoint_manager

    def distributed_eager_train_step(self, features):
        """Perform one distributed, eager train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Reduced loss tensor for chosen network across replicas.
        """
        if self.params["tf_version"] > 2.1:
            per_replica_losses = self.strategy.run(
                fn=self.train_step,
                kwargs={
                    "features": features
                }
            )
        else:
            per_replica_losses = self.strategy.experimental_run_v2(
                fn=self.train_step,
                kwargs={
                    "features": features
                }
            )

        return self.strategy.reduce(
            reduce_op=tf.distribute.ReduceOp.SUM,
            value=per_replica_losses,
            axis=None
        )

    def non_distributed_eager_train_step(self, features):
        """Perform one non-distributed, eager train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Reduced loss tensor for chosen network across replicas.
        """
        return self.train_step(features=features)

    @tf.function
    def distributed_graph_train_step(self, features):
        """Perform one distributed, graph train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Reduced loss tensor for chosen network across replicas.
        """
        if self.params["tf_version"] > 2.1:
            per_replica_losses = self.strategy.run(
                fn=self.train_step,
                kwargs={
                    "features": features
                }
            )
        else:
            per_replica_losses = self.strategy.experimental_run_v2(
                fn=self.train_step,
                kwargs={
                    "features": features
                }
            )

        return self.strategy.reduce(
            reduce_op=tf.distribute.ReduceOp.SUM,
            value=per_replica_losses,
            axis=None
        )

    @tf.function
    def non_distributed_graph_train_step(self, features):
        """Perform one non-distributed, graph train step.

        Args:
            features: dict, feature tensors from input function.

        Returns:
            Reduced loss tensor for chosen network across replicas.
        """
        return self.train_step(features=features)

    def log_step_loss(self, epoch, epoch_step, loss):
        """Logs step information and loss.

        Args:
            epoch: int, current iteration fully through the dataset.
            epoch_step: int, number of batches through epoch.
            loss: float, the loss of the model at the current step.
        """
        if self.global_step % self.params["log_step_count_steps"] == 0:
            print(
                "epoch = {}, global_step = {}, epoch_step = {}, loss = {}".format(
                    epoch, self.global_step, epoch_step, loss
                )
            )

    def training_loop(
        self, steps_per_epoch, train_dataset_iter, checkpoint_manager
    ):
        """Logs step information and loss.

        Args:
            steps_per_epoch: int, number of steps/batches to take each epoch.
            train_dataset_iter: iterator, training dataset iterator.
            checkpoint_manager: instance of `CheckpointManager`.
        """
        # Get correct train function based on parameters.
        if self.strategy:
            if self.params["use_graph_mode"]:
                train_step_fn = self.distributed_graph_train_step
            else:
                train_step_fn = self.distributed_eager_train_step
        else:
            if self.params["use_graph_mode"]:
                train_step_fn = self.non_distributed_graph_train_step
            else:
                train_step_fn = self.non_distributed_eager_train_step

        self.global_step = 0
        for epoch in range(self.params["num_epochs"]):
            for epoch_step in range(steps_per_epoch):
                features, labels = next(train_dataset_iter)

                # Train model on batch of features and get loss.
                loss = train_step_fn(features=features)

                # Log step information and loss.
                self.log_step_loss(epoch, epoch_step, loss)

                self.global_step += 1

            # Checkpoint model every so many steps.
            if self.global_step % self.params["save_checkpoints_steps"] == 0:
                checkpoint_manager.save(checkpoint_number=self.global_step)

    def training_loop_end_save_model(self, checkpoint_manager):
        """Saving model when training loop ends.

        Args:
            checkpoint_manager: instance of `CheckpointManager`.
        """
        # Write final checkpoint.
        checkpoint_manager.save(checkpoint_number=self.global_step)

        # Export SavedModel for serving.
        export_path = os.path.join(
            self.params["output_dir"],
            "export",
            datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        )

        # Signature will be serving_default.
        tf.saved_model.save(
            obj=self.network_models["generator"], export_dir=export_path
        )

    def train_block(self, train_dataset, eval_dataset):
        """Training block setups training, then loops through datasets.

        Args:
            train_dataset: instance of `Dataset` for training data.
            eval_dataset: instance of `Dataset` for evaluation data.
        """
        # Create iterators of datasets.
        train_dataset_iter = iter(train_dataset)
        eval_dataset_iter = iter(eval_dataset)

        steps_per_epoch = (
            self.params["train_dataset_length"] // self.global_batch_size
        )

        # Instantiate model objects.
        self.vanilla_gan_model()

        # Create checkpoint machinery to save/restore checkpoints.
        checkpoint_manager = self.create_checkpoint_machinery()

        # Create summary file writer.
        self.summary_file_writer = tf.summary.create_file_writer(
            logdir=os.path.join(self.params["output_dir"], "summaries"),
            name="summary_file_writer"
        )

        # Run training loop.
        self.training_loop(
            steps_per_epoch, train_dataset_iter, checkpoint_manager
        )

        # Save model at end of training loop.
        self.training_loop_end_save_model(checkpoint_manager)

    def train_and_evaluate(self):
        """Trains and evaluates Keras model.

        Args:
            args: dict, user passed parameters.

        Returns:
            Generator's `Model` object for in-memory predictions.
        """
        if self.params["distribution_strategy"]:
            # If the list of devices is not specified in the
            # Strategy constructor, it will be auto-detected.
            if self.params["distribution_strategy"] == "Mirrored":
                self.strategy = tf.distribute.MirroredStrategy()
            print(
                "Number of devices = {}".format(
                    self.strategy.num_replicas_in_sync
                )
            )

            # Set global batch size for training.
            self.global_batch_size = (
                self.params["train_batch_size"] * self.strategy.num_replicas_in_sync
            )

            # Get input datasets. Batch size is split evenly between replicas.
            train_dataset, eval_dataset = self.get_train_eval_datasets(
                num_replicas=self.strategy.num_replicas_in_sync
            )

            with self.strategy.scope():
                # Create distributed datasets.
                train_dist_dataset = (
                    self.strategy.experimental_distribute_dataset(
                        dataset=train_dataset
                    )
                )
                eval_dist_dataset = (
                    self.strategy.experimental_distribute_dataset(
                        dataset=eval_dataset
                    )
                )

                # Training block setups training, then loops through datasets.
                self.train_block(
                    train_dataset=train_dist_dataset,
                    eval_dataset=eval_dist_dataset
                )
        else:
            # Set global batch size for training.
            self.global_batch_size = self.params["train_batch_size"]

            # Get input datasets.
            train_dataset, eval_dataset = self.get_train_eval_datasets(
                num_replicas=1
            )

            # Training block setups training, then loops through datasets.
            self.train_block(
                train_dataset=train_dataset, eval_dataset=eval_dataset
            )
