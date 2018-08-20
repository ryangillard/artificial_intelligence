# c_neural_network

This project is a neural network framework written in C that can be used to build custom neural network models for many applications.

## Inputs

Currently there are three input files to the program in the form of tab delimited text files that will create your custom neural network architecture and supply important hyperparameters.

### number_of_hidden_layers.txt

This input file contains just one value which is the number of hidden layers of the neural network which is chosen from the set of non-negative integers, i.e. 0 &le; n &lt; &infin;. The simplest neural network will have zero hidden layers and thus will just consist of an input and output layer. Adding hidden layers can help the network learn much more complicated functions but will also run slower and may require more generalization to counteract overfitting.

### training_validation_hyperparameters.txt

This input files contains the hyperparameters used for training and evaluation.

loss_function_type - Integer that selects from the given loss function types.

classification_threshold - Float that sets the classification threshold.

alpha_forgiveness_rate - Float that sets the forgiveness of misclassification for generalized multi-class, multi-label classification. &alpha; &ge; 0

beta_false_negatives_cost & gamma_false_positives_cost - Floats that set the false negative and false positive costs which penalize each error type differently which is useful for certain problems for generalized multi-class, multi-label classification. &beta; &ge; 0, &gamma; &le; 1, &beta; = 1 | &gamma; = 1 

train_percent & valid_percent - Floats that determine how much of the input data is used for training and validation respectively. 0 &lt; train_percent + valid_percent &le; 1

train_steps	& eval_steps - Positive integers used to set the number of training steps (mini-batches) to train on and how many training steps to perform before the next model evaluation using the validation set.

batch_size - Positive integer that determines how many examples to include in each mini-batch for training. A batch size of 1 will reduce to stochastic gradient descent and a batch size of the total number of training examples reduces to batch gradient descent.

learning_rate - Float that determines how fast the network parameters learn the input data function by multiplying itself with the gradient as part of the parameter update. Too low of a value and the network will train too slowly, too high the network might get stuck in a suboptimal solution or even diverge.

clip_norm - Float that ensures that the global norm of the parameter gradients doesn't go above this value. If set to 0, then there will be no gradient clipping.

optimizer - Integer that selects from the given gradient descent optimizers to use.

optimizer_parameter0 thru optimizer_parameter4 - Floats. These are parameters used for the gradient descent optimizers. Each optimizer has a different number of parameters therefore not all of these will be used based on the optimizer hyperparameter selected above.

### neural_network_layer_hyperparameters.txt

This input file contains the hyperparameters for each neural network layer. Each line is a layer and there should be number_of_hidden_layers + 2 layers (hidden layers plus an input layer and an output layer).

number_of_neurons - Positive integer that sets the number of neurons in this layer.

activation_type - Integer that selects from the given activation function types to use for all of the neurons in this layer.

activation_function_alpha - Float that sets the alpha parameter of certain activation functions such as leaky ReLU and ELU for this layer.

kernel_weight_initialization_type - Integer that selects from given initialization types for this layer's kernel weights.

kernel_weight_initialization_parameter0 thru kernel_weight_initialization_parameter2 - Floats. These are parameters used for this layer's kernel weight initialization. Each initialization type has a different number of parameters therefore not all of these will be used based on the kernel weight initialization hyperparameter selected above.

kernel_l1_regularization_strength - Float that sets the L1 regularization strength for this layer's kernel weights. If set to 0, then there is no L1 regularization for this layer's kernel weights.

kernel_l2_regularization_strength - Float that sets the L2 regularization strength for this layer's kernel weights. If set to 0, then there is no L2 regularization for this layer's kernel weights.

bias_weight_initialization_type - Integer that selects from given initialization types for this layer's bias weights.

bias_weight_initialization_parameter0 & bias_weight_initialization_parameter1 - Floats. These are parameters used for this layer's bias weight initialization. Each initialization type has a different number of parameters therefore not all of these will be used based on the bias weight initialization hyperparameter selected above.

bias_l1_regularization_strength - Float that sets the L1 regularization strength for this layer's bias weights. If set to 0, then there is no L1 regularization for this layer's bias weights.

bias_l2_regularization_strength - Float that sets the L2 regularization strength for this layer's bias weights. If set to 0, then there is no L2 regularization for this layer's bias weights.

dropout_probability - Float that sets this layer's dropout probability where 0 &le; p &le; 1; A probability of 0 means there is no dropout and the network functions as usual. A probability of 1 means that everything is dropped out which means this layer will not be able to learn. A good value is usually something between 0.1 and 0.5.

batch_norm_flag - Integer 0 or 1 that indicates if this layer will have a batch normalization wrapper or not.

batch_norm_after_activation_flag - Integer 0 or 1 that indicates if batch normalization (if it is on) for this layer will be applied after the activation function or not.

batch_norm_momentum - Float that determines how much of the moving batch normalization statistics are carried over to the next mini-batch during training for this layer.

batch_norm_moving_mean_initializer - Float that sets the initialization value of the batch normalization moving mean for this layer.

batch_norm_moving_variance_initializer - Float that sets the initialization value of the batch normalization moving variance for this layer.

batch_norm_beta_initializer - Float that sets the initialization value of the batch normalization beta for this layer.

batch_norm_gamma_initializer - Float that sets the initialization value of the batch normalization gamma for this layer.

## Outputs

The outputs save the final model parameters to disk so that they can be reloaded for inference at any time.

### kernel_weights.txt

This output file contains the kernel weight values for each layer. Each matrix is num_input_neurons x num_output_neurons for each layer.

### bias_weights.txt

This output file contains the bias weight values for each layer. Each vector is num_output_neurons for each layer.

### parametric_relu_alphas.txt

This output file contains the parametric ReLU alpha values for each layer (if layer uses parametric ReLU activation functions). Each vector is num_output_neurons for each layer.

### batch_norm_parameters.txt

This output file contains the batch normalization parameter values for each layer (if layer uses batch normalization). Each matrix is num_output_neurons by 4 (includes in order the moving mean, moving variance, beta, and gamma) for each layer.
