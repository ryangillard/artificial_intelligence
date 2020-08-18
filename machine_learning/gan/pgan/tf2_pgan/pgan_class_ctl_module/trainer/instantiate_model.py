import tensorflow as tf

from . import discriminators
from . import generators


class InstantiateModel(object):
    """Class that contains methods used for instantiating model objects.
    """
    def __init__(self):
        """Instantiate instance of `InstantiateModel`.
        """
        pass

    def _instantiate_optimizer(self, scope):
        """Instantiates scoped optimizer with parameters.

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

    def _instantiate_optimizers(self):
        """Instantiates all network optimizers.
        """
        # Instantiate generator optimizer.
        self._instantiate_optimizer(scope="generator")

        # Instantiate discriminator optimizer.
        self._instantiate_optimizer(scope="discriminator")

    def _instantiate_network_objects(self):
        """Instantiates generator and discriminator objects with parameters.
        """
        # Instantiate generator.
        self.network_objects["generator"] = generators.Generator(
            kernel_regularizer=tf.keras.regularizers.l1_l2(
                l1=self.params["generator_l1_regularization_scale"],
                l2=self.params["generator_l2_regularization_scale"]
            ),
            bias_regularizer=None,
            name="generator",
            params=self.params,
            alpha_var=self.alpha_var,
            num_growths=self.num_growths
        )

        # Instantiate discriminator.
        self.network_objects["discriminator"] = discriminator.Discriminator(
            kernel_regularizer=tf.keras.regularizers.l1_l2(
                l1=self.params["discriminator_l1_regularization_scale"],
                l2=self.params["discriminator_l2_regularization_scale"]
            ),
            bias_regularizer=None,
            name="discriminator",
            params=self.params,
            alpha_var=self.alpha_var,
            num_growths=self.num_growths
        )

    def _get_unique_trainable_variables(self, scope):
        """Gets union of unique trainable variables within given scope.

        Args:
            scope: str, the name of the network of interest.
        """
        # All names of 0th model variables are already guaranteed unique.
        unique_names = set(
            [
                var.name
                for var in (
                    self.network_objects[scope].models[0].trainable_variables
                )
            ]
        )

        unique_trainable_variables = (
            self.network_objects[scope].models[0].trainable_variables
        )

        # Loop through future growth models to get trainable variables.
        for i in range(1, self.num_growths):
            trainable_variables = (
                self.network_objects[scope].models[i].trainable_variables
            )

            # Loop through variables and append any that are unique.
            for var in trainable_variables:
                if var.name not in unique_names:
                    unique_names.add(var.name)
                    unique_trainable_variables.append(var)

        self.unique_trainable_variables[scope] = unique_trainable_variables

    def _create_optimizer_variable_slots(self, scope):
        """Creates optimizer variable slots for given scoped model type.

        It is needed to build any optimizer variables within graph mode since
        variables cannot be created outside the first call of a tf.function.

        Args:
            scope: str, the name of the network of interest.
        """
        # Get the union of all trainable variables across all model growths.
        self._get_unique_trainable_variables(scope)

        # Create placeholder gradients that we can apply to model variables.
        # Note: normally some gradients (especially of future growth models)
        placeholder_gradients = [
            tf.zeros_like(input=var, dtype=tf.float32)
            for var in self.unique_trainable_variables[scope]
        ]

        # Apply gradients to create optimizer variable slots for each
        # trainable variable.
        self.optimizers[scope].apply_gradients(
            zip(
                placeholder_gradients, self.unique_trainable_variables[scope]
            )
        )

    @tf.function
    def _non_distributed_instantiate_optimizer_variables(self):
        """Instantiates optimizer variable slots for given scoped model type.

        It is needed to build any optimizer variables within graph mode since
        variables cannot be created outside the first call of a tf.function.
        This is the non-distributed version.

        Args:
            scope: str, the name of the network of interest.
        """
        self._create_optimizer_variable_slots(scope="generator")
        self._create_optimizer_variable_slots(scope="discriminator")

        return tf.zeros(shape=[], dtype=tf.float32)

    @tf.function
    def _distributed_instantiate_optimizer_variables(self):
        """Instantiates optimizer variable slots for given scoped model type.

        It is needed to build any optimizer variables within graph mode since
        variables cannot be created outside the first call of a tf.function.
        This is the distributed version.

        Args:
            scope: str, the name of the network of interest.
        """
        if self.params["tf_version"] > 2.1:
            run_function = self.strategy.run
        else:
            run_function = self.strategy.experimental_run_v2

        per_replica_losses = run_function(
            fn=self._non_distributed_instantiate_optimizer_variables
        )

        return self.strategy.reduce(
            reduce_op=tf.distribute.ReduceOp.SUM,
            value=per_replica_losses,
            axis=None
        )

    def instantiate_model_objects(self):
        """Instantiate model network objects, network models, and optimizers.
        """
        # Instantiate generator and discriminator optimizers.
        self._instantiate_optimizers()

        # Instantiate generator and discriminator objects.
        self._instantiate_network_objects()

        # Instantiate generator and discriminator optimizer variable slots.
        if self.params["use_graph_mode"]:
            if self.strategy:
                _ = self._distributed_instantiate_optimizer_variables()
            else:
                _ = self._non_distributed_instantiate_optimizer_variables()
