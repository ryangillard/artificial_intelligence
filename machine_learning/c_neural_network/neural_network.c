/*
 * neural_network.c
 *
 *      Author: Ryan S. Gillard
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// This is a sign function macro
#define sign(x) (x > 0) - (x < 0)

// This is our global epsilon to avoid division by zero
double epsilon = 0.00000001;

// Whether we want debug print statements or not
int debug_print = 0;

/**************************************************************************************************************/
/**************************************************************************************************************/
/************************************************ STRUCTURES **************************************************/
/**************************************************************************************************************/
/**************************************************************************************************************/

/* This structure stores data for each trainable parameter */
struct TrainableParameters
{
	double variable;
	double gradient;
	double aggregate0;
	double aggregate1;
	double aggregate2;
	double update;
};

/* This structure stores data for each neural network completely connected layer */
struct NeuralNetworkLayerHyperparameters
{
	unsigned int number_of_neurons;

	int activation_type;
	double activation_function_alpha_initializer;

	int kernel_weight_initialization_type;
	double kernel_weight_initialization_parameter0;
	double kernel_weight_initialization_parameter1;
	double kernel_weight_initialization_parameter2;

	double kernel_l1_regularization_strength;
	double kernel_l2_regularization_strength;

	int bias_weight_initialization_type;
	double bias_weight_initialization_parameter0;
	double bias_weight_initialization_parameter1;

	double bias_l1_regularization_strength;
	double bias_l2_regularization_strength;

	double dropout_probability;

	int batch_norm_flag;
	int batch_norm_after_activation_flag;
	double batch_norm_momentum;
	double batch_norm_moving_mean_initializer;
	double batch_norm_moving_variance_initializer;
	double batch_norm_beta_initializer;
	double batch_norm_gamma_initializer;
};

/* This structure stores data for the hyperparameters for training and validation */
struct TrainingValidationHyperparameters
{
	int loss_function_type;
	double classification_threshold;
	double alpha_forgiveness_rate;
	double beta_false_negative_cost;
	double gamma_false_positive_cost;

	double train_percent;
	double valid_percent;
	unsigned int train_steps;
	unsigned int eval_steps;

	unsigned int batch_size;
	double learning_rate;
	double clip_norm;

	int optimizer;
	double optimizer_parameter0;
	double optimizer_parameter1;
	double optimizer_parameter2;
	double optimizer_parameter3;
	double optimizer_parameter4;
};

/* This structure stores data for a neuron */
struct NeuralNetworkLayerNeuron
{
	double weighted_sum;
	double activation;
	double delta;
};

/* This structure stores data for each batch normalization layer */
struct BatchNormalizationLayerNeuronParameters
{
	double batch_norm_mean;
	double batch_norm_variance;
	double batch_norm_moving_mean;
	double batch_norm_moving_variance;

	struct TrainableParameters batch_norm_beta;
	struct TrainableParameters batch_norm_gamma;
};

/* This structure stores data for regression evaluation metrics */
struct RegressionEvaluationMetric
{
	unsigned int train_step;
	double loss;
	double mse;
	double rmse;
};

/* This structure stores data for classification evaluation metrics */
struct ClassificationEvaluationMetric
{
	unsigned int train_step;
	double loss;
	double exact_match_ratio;
	double accuracy;
	double avg_precision;
	double avg_recall;
	double f1_score;
};

/**************************************************************************************************************/
/**************************************************************************************************************/
/************************************************ PROTOTYPES **************************************************/
/**************************************************************************************************************/
/**************************************************************************************************************/

/******************************************************************************************/
/*********************************** OVERALL STRUCTURE ************************************/
/******************************************************************************************/

/* This function reads a 0d unsigned int from the given filepath */
void ReadFileTensor0DUnsignedInt(char *filepath, unsigned int *tensor);
/* This function reads the neural network layer hyperparameters from the given filepath */
void ReadFileNeuralNetworkLayerHyperparameters(char *filepath, unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters);
/* This function reads the training hyperparameters from the given filepath */
void ReadFileTrainingHyperparameters(char *filepath, struct TrainingValidationHyperparameters *training_validation_hyperparameters);

/******************************************************************************************/
/********************************** TRAINABLE PARAMETERS **********************************/
/******************************************************************************************/

/* This function initializes kernel weights for each layer depending on the kernel weight initialization type */
void InitializeKernelWeights(unsigned int number_of_layers, unsigned int total_kernel_weights, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, double initial_accumulator_value, struct TrainableParameters *neural_network_kernel_weights, unsigned int *kernel_weights_offset_indices);
/* This function initializes kernel weights in layer to a given constant number */
void KernelWeightInitializationConstant(unsigned int layer_index, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainableParameters *neural_network_kernel_weights, unsigned int offset, double constant);
/* This function initializes kernel weights in layer to a random uniform number within a given range */
void KernelWeightInitializationRandomUniform(unsigned int layer_index, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainableParameters *neural_network_kernel_weights, unsigned int offset, double range_min, double range_max);
/* This function initializes kernel weights in layer to a random normal number with a given mean and standard deviation */
void KernelWeightInitializationRandomNormal(unsigned int layer_index, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainableParameters *neural_network_kernel_weights, unsigned int offset, double mu, double sigma);
/* This function initializes kernel weights in layer to a random normal number with a given mean and standard deviation within two standard deviations */
void KernelWeightInitializationTruncatedNormal(unsigned int layer_index, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainableParameters *neural_network_kernel_weights, unsigned int offset, double mu, double sigma);
/* This function initializes kernel weights in layer to a random normal or uniform number with a given parameters using variance scaling */
void KernelWeightInitializationVarianceScaling(unsigned int layer_index, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainableParameters *neural_network_kernel_weights, unsigned int offset, double scale, double mode, double distribution);

/* This function initializes bias weights for each layer depending on the bias weight initialization type */
void InitializeBiasWeights(unsigned int number_of_layers, unsigned int total_bias_weights, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, double initial_accumulator_value, struct TrainableParameters *neural_network_bias_weights, unsigned int *bias_weights_offset_indices);
/* This function initializes bias weights in layer to a given constant number */
void BiasWeightInitializationConstant(unsigned int layer_index, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainableParameters *neural_network_bias_weights, unsigned int offset, double constant);
/* This function initializes bias weights in layer to a random uniform number within a given range */
void BiasWeightInitializationRandomUniform(unsigned int layer_index, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainableParameters *neural_network_bias_weights, unsigned int offset, double range_min, double range_max);
/* This function initializes bias weights in layer to a random normal number with a given mean and standard deviation */
void BiasWeightInitializationRandomNormal(unsigned int layer_index, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainableParameters *neural_network_bias_weights, unsigned int offset, double mu, double sigma);
/* This function initializes bias weights in layer to a random normal number with a given mean and standard deviation within two standard deviations */
void BiasWeightInitializationTruncatedNormal(unsigned int layer_index, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainableParameters *neural_network_bias_weights, unsigned int offset, double mu, double sigma);

/* This function returns a random uniform number within given range */
double RUnif(double range_min, double range_max);
/* This function returns a random normal number with given mean and standard deviation */
double RNorm(double mu, double sigma);
/* This function returns a random uniform number within range [0,1] */
double UnifRand(void);
/* This function returns a random normal number with zero mean and unit standard deviation */
double NormRand(void);

/* This function initializes parametric ReLU alphas for each layer depending on the activation function alpha initializer */
void InitializeParametricReLUAlphas(unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, double initial_accumulator_value, struct TrainableParameters *parametric_relu_alphas, unsigned int *parametric_relu_alpha_offset_indices);
/* This function initializes batch normalization layer parameters for each layer depending on several initializers */
void InitializeBatchNormalizationLayerParameters(unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, double initial_accumulator_value, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, unsigned int *batch_normalization_layer_offset_indices);

/* This function initializes trainable parameters gradient and aggregate information */
void InitializeTrainableParameterGradientsAndAggregates(double initial_accumulator_value, double *gradient, double *aggregate0, double *aggregate1, double *aggregate2, double *update);

/******************************************************************************************/
/*************************************** INPUT DATA ***************************************/
/******************************************************************************************/

/* This function allocates memory for features and labels arrays */
void AllocateFeaturesAndLabels(unsigned int rows, unsigned int feature_cols, unsigned int label_cols, double **features, double **labels);
/* This function reads the input data features and labels from the given filepath */
void InputDataFunction(char *filepath, unsigned int rows, unsigned int feature_cols, unsigned int label_cols, double **features_tensor, double **labels_tensor, unsigned int transpose);
/* This function initializes the features and labels arrays for each split (i.e. train, valid, test) */
void InitializeSplitFeaturesAndLabels(unsigned int rows, unsigned int feature_cols, unsigned int label_cols, unsigned int offset, double *features, double *labels, double *features_split, double *labels_split, int transpose);

/******************************************************************************************/
/********************************* TRAINING AND EVALUATION ********************************/
/******************************************************************************************/

/* This function performs mini batch gradient descent for both training (FeedForward and Backpropagation) and evaluation (FeedForward) */
void MiniBatchGradientDescent(unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int train_rows, double *features_train, double *labels_train, unsigned int valid_rows, double *features_valid, double *labels_valid, struct TrainableParameters *neural_network_kernel_weights, unsigned int *kernel_weights_offset_indices, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int *bias_weights_offset_indices, unsigned int total_bias_weights, struct TrainableParameters *parametric_relu_alphas, unsigned int *parametric_relu_alpha_offset_indices, unsigned int total_parametric_relu_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, unsigned int *batch_normalization_layer_offset_indices, unsigned int total_batch_normalization_neurons, int transpose);
/* This function creates a mini-batch from the training data */
unsigned int CreateTrainingDataMiniBatch(int transpose, unsigned int batch_size, unsigned int number_of_input_neurons, unsigned int number_of_output_neurons, unsigned int train_rows, unsigned int max_train_feature_elements, unsigned int max_train_label_elements, double *features_train, double *labels_train, double *batch_features_tensor, double *batch_labels_tensor, unsigned int current_training_record_index);

/******************************************************************************************/
/************************************** FEEDFORWARD ***************************************/
/******************************************************************************************/

/* This function feeds the input data forward through the neural network layers */
double FeedForward(unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int batch_size, double *batch_features_tensor, double *batch_labels_tensor, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int *neural_network_neurons_minus_inputs_offset_indices, struct TrainableParameters *neural_network_kernel_weights, unsigned int *kernel_weights_offset_indices, struct TrainableParameters *neural_network_bias_weights, unsigned int *bias_weights_offset_indices, struct TrainableParameters *parametric_relu_alphas, unsigned int *parametric_relu_alpha_offset_indices, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, double *batch_normalization_layer_xmu, unsigned int *batch_normalization_layer_offset_indices, int *dropout_layer_mask_neurons, unsigned int *dropout_layer_offset_indices, int training, int transpose);
/* This function performs matrix multiplication between two given matrices */
void MatrixMultiplication(unsigned int m, unsigned int n, unsigned int p, double *A, double *B, double *C, unsigned int A_offset, unsigned int B_offset, unsigned int C_offset, unsigned long A_size, unsigned long B_size, unsigned long C_size, int transpose_A, int transpose_B);
/* This function adds biases (if present) to the result of matmul(X, W) */
void AddBiases(unsigned int batch_size, unsigned int number_of_neurons, struct TrainableParameters *neural_network_bias_weights, unsigned int bias_weights_offset, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int neuron_offset);

/******************************************************************************************/
/********************************* ACTIVATION FUNCTIONS ***********************************/
/******************************************************************************************/

/* This function applies given activation function to given layers weighted sums and returns result as layer's activations */
void ApplyActivationFunction(int activation_type, unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int neuron_offset, double non_parametric_alpha, struct TrainableParameters *parametric_relu_alphas, unsigned int parametric_relu_offset);
/* This function applies linear activation function to given layer's neurons */
void ApplyLinearActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int );
/* This function returns the linear activation */
double ActivationFunctionLinear(double x);
/* This function applies sigmoid activation function to given layer's neurons */
void ApplySigmoidActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int );
/* This function returns the sigmoid activation */
double ActivationFunctionSigmoid(double x);
/* This function applies tanh activation function to given layer's neurons */
void ApplyTanhActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int );
/* This function returns the tanh activation */
double ActivationFunctionTanh(double x);
/* This function applies ReLU activation function to given layer's neurons */
void ApplyReLUActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int );
/* This function returns the ReLU activation */
double ActivationFunctionReLU(double x);
/* This function applies leaky ReLU activation function to given layer's neurons */
void ApplyLeakyReLUActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int , double alpha);
/* This function applies parametric ReLU activation function to given layer's neurons */
void ApplyParametricReLUActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int , struct TrainableParameters *parametric_relu_alphas, unsigned int parametric_relu_offset);
/* This function returns the parametric ReLU activation with alpha */
double ActivationFunctionParametricReLU(double x, double alpha);
/* This function applies elu activation function to given layer's neurons */
void ApplyEluActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int , double alpha);
/* This function returns the elu activation */
double ActivationFunctionElu(double x, double alpha);
/* This function applies softmax activation function to given layer's neurons */
void ApplySoftmaxActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int );

/******************************************************************************************/
/*********************************** ADVANCED METHODS *************************************/
/******************************************************************************************/

/* This function applies batch normalization to the given layer (note different behavior for training and inference) */
void ApplyBatchNormalization(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int activations_offset, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer, double *batch_normalization_layer_xmu, unsigned int batch_normalization_layer_offset, double batch_norm_momentum, int training);
/* This function applies dropout to the given layer (only during training) */
void ApplyDropout(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int activations_offset, int *dropout_layer_mask_neurons, unsigned int dropout_offset, double dropout_probability);

/******************************************************************************************/
/***************************************** LOSS *******************************************/
/******************************************************************************************/

/* This function calculates the given loss function at the end of the forward pass of the neural network layers */
double CalculateLoss(unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, int loss_function_type, unsigned int batch_size, unsigned int number_of_outputs, double *logits, unsigned int logits_offset, unsigned long logits_size, double *labels, int transpose, struct TrainableParameters *neural_network_kernel_weights, unsigned int *kernel_weights_offset_indices, struct TrainableParameters *neural_network_bias_weights, unsigned int *bias_weights_offset_indices);
/* This function calculates the mean squared error loss typically used for regression */
double MSELossFunction(unsigned int batch_size, unsigned int number_of_outputs, double *logits, unsigned int logits_offset, unsigned long logits_size, double *labels, int transpose);
/* This function calculates the softmax cross entropy loss using the logits of the final layer typically used for multi-class, single label classification */
double SoftmaxCrossEntropyWithLogitsLossFunction(unsigned int batch_size, unsigned int number_of_outputs, double *logits, unsigned int logits_offset, unsigned long logits_size, double *labels, int transpose);
/* This function calculates the sigmoid cross entropy loss using the logits of the final layer typically used for multi-class, multi-label classification */
double SigmoidCrossEntropyWithLogitsLossFunction(unsigned int batch_size, unsigned int number_of_outputs, double *logits, unsigned int logits_offset, unsigned long logits_size, double *labels, int transpose);

/******************************************************************************************/
/************************************ REGULARIZATION **************************************/
/******************************************************************************************/

/* This function calculates L1 regularization loss of given kernel weight layer */
double AddKernelL1RegularizationLoss(unsigned int current_layer_number_of_neurons, unsigned int next_layer_number_of_neurons, struct TrainableParameters *neural_network_kernel_weights, unsigned int kernel_weights_offset, double lambda1, unsigned int batch_size);
/* This function calculates L2 regularization loss of given kernel weight layer */
double AddKernelL2RegularizationLoss(unsigned int current_layer_number_of_neurons, unsigned int next_layer_number_of_neurons, struct TrainableParameters *neural_network_kernel_weights, unsigned int kernel_weights_offset, double lambda2, unsigned int batch_size);
/* This function calculates L1 regularization loss of given bias weight layer */
double AddBiasL1RegularizationLoss(unsigned int next_layer_number_of_neurons, struct TrainableParameters *neural_network_bias_weights, unsigned int bias_weights_offset, double lambda1, unsigned int batch_size);
/* This function calculates L2 regularization loss of given bias weight layer */
double AddBiasL2RegularizationLoss(unsigned int next_layer_number_of_neurons, struct TrainableParameters *neural_network_bias_weights, unsigned int bias_weights_offset, double lambda2, unsigned int batch_size);

/******************************************************************************************/
/************************************** EVALUATION ****************************************/
/******************************************************************************************/

/* This function evaluates the currently trained model on the valdiation dataset */
void ModelEvaluation(unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int batch_size, double *features, double *labels, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int *neural_network_neurons_minus_inputs_offset_indices, struct TrainableParameters *neural_network_kernel_weights, unsigned int *kernel_weights_offset_indices, struct TrainableParameters *neural_network_bias_weights, unsigned int *bias_weights_offset_indices, struct TrainableParameters *parametric_relu_alphas, unsigned int *parametric_relu_alpha_offset_indices, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, double *batch_normalization_layer_xmu, unsigned int *batch_normalization_layer_offset_indices, int *dropout_layer_mask_neurons, unsigned int *dropout_layer_offset_indices, int training, int transpose, struct RegressionEvaluationMetric *regression_evaluation_metrics, struct ClassificationEvaluationMetric *classification_evaluation_metrics, unsigned int *evaluation_count, unsigned int step);
/* This function calculates common regression evaluation metrics */
void CalculateRegressionEvaluationMetrics(unsigned int batch_size, unsigned int number_of_outputs, double *logits, unsigned int logits_offset, unsigned long logits_size, double *labels, double *mse, double *rmse);
/* This function calculates common classification evaluation metrics */
void CalculateClassificationEvaluationMetrics(unsigned int batch_size, unsigned int number_of_outputs, double *probabilities, unsigned int probabilities_offset, unsigned long probabilities_size, double *labels, double classification_threshold, int multilabel, double alpha_forgiveness_rate, double beta_false_negative_cost, double gamma_false_positive_cost, double *exact_match_ratio, double *accuracy, double *avg_precision, double *avg_recall, double *f1_score);

/******************************************************************************************/
/************************************ BACKPROPAGATION *************************************/
/******************************************************************************************/

/* This function backpropagates the error through each layer of the neural network to find the parameter gradients to update the parameters */
void Backpropagation(unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int batch_size, double *batch_features_tensor, double *batch_labels_tensor, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int *neural_network_neurons_minus_inputs_offset_indices, struct TrainableParameters *neural_network_kernel_weights, unsigned int *kernel_weights_offset_indices, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int *bias_weights_offset_indices, unsigned int total_bias_weights, struct TrainableParameters *parametric_relu_alphas, unsigned int *parametric_relu_alpha_offset_indices, unsigned int total_parametric_relu_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, double *batch_normalization_layer_xmu, unsigned int *batch_normalization_layer_offset_indices, unsigned int total_batch_normalization_neurons, int *dropout_layer_mask_neurons, unsigned int *dropout_layer_offset_indices, int transpose, unsigned int train_step);

/******************************************************************************************/
/************************************** DERIVATIVES ***************************************/
/******************************************************************************************/

/* This function applies derivative of given activation function to given layers weighted sums and returns result as layer's deltas */
void ApplyDerivativeActivationFunction(int activation_type, unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int neuron_offset, double non_parametric_alpha, struct TrainableParameters *parametric_relu_alphas, unsigned int parametric_relu_offset);
/* This function applies the derivative of linear activation function to given layer's neurons */
void ApplyDerivativeLinearActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int );
/* This function returns the derivative of linear activation */
double DerivativeActivationFunctionLinear(double x);
/* This function applies the derivative of sigmoid activation function to given layer's neurons */
void ApplyDerivativeSigmoidActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int );
/* This function returns the derivative of sigmoid activation */
double DerivativeActivationFunctionSigmoid(double x);
/* This function applies the derivative of tanh activation function to given layer's neurons */
void ApplyDerivativeTanhActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int );
/* This function returns the derivative of tanh activation */
double DerivativeActivationFunctionTanh(double x);
/* This function applies the derivative of ReLU activation function to given layer's neurons */
void ApplyDerivativeReLUActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int );
/* This function returns the derivative of ReLU activation */
double DerivativeActivationFunctionReLU(double x);
/* This function applies the derivative of leaky ReLU activation function to given layer's neurons */
void ApplyDerivativeLeakyReLUActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int , double alpha);
/* This function applies the derivative of parametric ReLU activation function to given layer's neurons */
void ApplyDerivativeParametricReLUActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int neuron_offset, struct TrainableParameters *parametric_relu_alphas, unsigned int parametric_relu_offset);
/* This function returns the derivative of parametric ReLU activation */
double DerivativeActivationFunctionParametricReLU(double x, double alpha);
/* This function applies the derivative of elu activation function to given layer's neurons */
void ApplyDerivativeEluActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int , double alpha);
/* This function returns the derivative of elu activation */
double DerivativeActivationFunctionElu(double x, double alpha);

/******************************************************************************************/
/*************************************** GRADIENTS ****************************************/
/******************************************************************************************/

/* This function applies the gradient of batch normalization to the given layer */
void ApplyBatchNormalizationGradient(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int activations_offset, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, double *batch_normalization_layer_xmu, unsigned int batch_normalization_layer_offset, double learning_rate);
/* This function applies the gradient of dropout to the given layer */
void ApplyDropoutGradient(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int activations_offset, int *dropout_layer_mask_neurons, unsigned int dropout_offset, double dropout_probability);

/* This function calculates the gradients of the neural network kernel weights */
void CalculateNeuralNetworkKernelWeightGradients(unsigned int batch_size, unsigned int number_of_neurons_in, unsigned int number_of_neurons_out, double *activations, double *deltas, struct TrainableParameters *neural_network_kernel_weights, unsigned long activation_size, unsigned long delta_size, unsigned int activation_offset, unsigned int delta_offset, unsigned int kernel_offset);
/* This function calculates the gradients of the neural network bias weights */
void CalculateNeuralNetworkBiasWeightGradients(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, struct TrainableParameters *neural_network_bias_weights, unsigned int delta_offset, unsigned int bias_offset);
/* This function calculates the gradients of the parametric ReLU alphas */
void CalculateParametricReLUGradients(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int neuron_offset, struct TrainableParameters *parametric_relu_alphas, unsigned int parametric_relu_offset);
/* This function calculates the gradients of the batch normalization layer parameters */
void CalculateBatchNormalizationParameterGradients(unsigned int batch_norm_index, unsigned int batch_size, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, double delta_beta, double delta_gamma);

/******************************************************************************************/
/******************************** REGULARIZATION GRADIENTS ********************************/
/******************************************************************************************/

/* This function calculates the gradient of L1 regularization loss of given kernel weight layer */
void AddKernelL1RegularizationGradient(unsigned int current_layer_number_of_neurons, unsigned int next_layer_number_of_neurons, struct TrainableParameters *neural_network_kernel_weights, unsigned int kernel_weights_offset, double lambda1, unsigned int batch_size);
/* This function calculates the gradient of L2 regularization loss of given kernel weight layer */
void AddKernelL2RegularizationGradient(unsigned int current_layer_number_of_neurons, unsigned int next_layer_number_of_neurons, struct TrainableParameters *neural_network_kernel_weights, unsigned int kernel_weights_offset, double lambda2, unsigned int batch_size);
/* This function calculates the gradient of L1 regularization loss of given bias weight layer */
void AddBiasL1RegularizationGradient(unsigned int next_layer_number_of_neurons, struct TrainableParameters *neural_network_bias_weights, unsigned int bias_weights_offset, double lambda1, unsigned int batch_size);
/* This function calculates the gradient of L2 regularization loss of given bias weight layer */
void AddBiasL2RegularizationGradient(unsigned int next_layer_number_of_neurons, struct TrainableParameters *neural_network_bias_weights, unsigned int bias_weights_offset, double lambda2, unsigned int batch_size);

/******************************************************************************************/
/*********************************** GRADIENT CLIPPING ************************************/
/******************************************************************************************/

/* This function clips gradients by their global norm */
void ApplyGradientClippingByGlobalNorm(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters);
/* This function calculates the global norm of all gradients */
double CalculateGradientGlobalNorm(unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters);
/* This function calculates the sum of a trainable parameter's squared gradients */
double CalculateSquaredGradientsSum(unsigned int total_gradients, double *gradient, unsigned long gradient_size);
/* This function applies the global norm scale factor to gradient */
void ApplyGradientGlobalNormScaleFactor(double *gradient, double scale_factor);

/******************************************************************************************/
/********************************** GRADIENT OPTIMIZERS ***********************************/
/******************************************************************************************/

/* This function applies gradient optimizers to the currently calculated gradients */
void ApplyGradientOptimizers(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, unsigned int train_step);
/* This function applies the vanilla gradient descent optimizer to the currently calculated gradients */
void ApplyOptimizerVanillaSGD(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters);
/* This function is the vanilla sgd optimizer */
void VanillaSGDOptimizer(double gradient, double *update, double learning_rate);
/* This function applies the momentum optimizer to the currently calculated gradients */
void ApplyOptimizerMomentum(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters);
/* This function is the momentum optimizer */
void MomentumOptimizer(double gradient, double *update, double learning_rate, double momentum);
/* This function applies the nesterov momentum optimizer to the currently calculated gradients */
void ApplyOptimizerNesterovMomentum(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, unsigned int train_step);
/* This function is the nesterov momentum optimizer */
void NesterovMomentumOptimizer(double variable, double gradient, double *m_moment_aggregate, double *update, double learning_rate, double momentum, unsigned int train_step);
/* This function applies the adagrad optimizer to the currently calculated gradients */
void ApplyOptimizerAdagrad(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters);
/* This function is the adagrad optimizer */
void AdagradOptimizer(double gradient, double *accumulator_aggregate, double *update, double learning_rate);
/* This function applies the adadelta optimizer to the currently calculated gradients */
void ApplyOptimizerAdadelta(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters);
/* This function is the adadelta optimizer */
void AdadeltaOptimizer(double gradient, double *accumulator_aggregate, double *delta_accumulator_aggregate, double *update, double rho);
/* This function applies the rmsprop optimizer to the currently calculated gradients */
void ApplyOptimizerRMSProp(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters);
/* This function is the rmsprop optimizer */
void RMSPropOptimizer(double gradient, double *m_moment_aggregate, double *accumulator_aggregate, double *update, double learning_rate, double decay, double momentum, int centered);
/* This function applies the adam optimizer to the currently calculated gradients */
void ApplyOptimizerAdam(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, unsigned int train_step);
/* This function is the adam optimizer */
void AdamOptimizer(double gradient, double *m_moment_aggregate, double *v_moment_aggregate, double *update, double effective_learning_rate, double beta1, double beta2);
/* This function applies the adamax optimizer to the currently calculated gradients */
void ApplyOptimizerAdamax(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, unsigned int train_step);
/* This function is the adamax optimizer */
void AdamaxOptimizer(double gradient, double *m_moment_aggregate, double *v_moment_aggregate, double *update, double effective_learning_rate, double beta1, double beta2);
/* This function applies the nadam optimizer to the currently calculated gradients */
void ApplyOptimizerNadam(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, unsigned int train_step);
/* This function is the nadam optimizer */
void NadamOptimizer(double gradient, double *m_moment_aggregate, double *v_moment_aggregate, double *update, double effective_learning_rate, double beta1, double beta2);
/* This function applies the amsgrad optimizer to the currently calculated gradients */
void ApplyOptimizerAMSGrad(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, unsigned int train_step);
/* This function is the amsgrad optimizer */
void AMSGradOptimizer(double gradient, double *m_moment_aggregate, double *v_moment_aggregate, double *v_moment_hat_aggregate, double *update, double effective_learning_rate, double beta1, double beta2);
/* This function applies the ftrl optimizer to the currently calculated gradients */
void ApplyOptimizerFTRL(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters);
/* This function is the ftrl optimizer */
void FTRLOptimizer(double variable, double gradient, double *accumulator_aggregate, double *accumulator_new_aggregate, double *linear_aggregate, double *update, double learning_rate, double learning_rate_power, double l1_regularization_strength, double l2_regularization_strength, double l2_shrinkage_regularization_strength);

/******************************************************************************************/
/****************************** UPDATE TRAINABLE PARAMETERS *******************************/
/******************************************************************************************/

/* This function updates the neural network's trainable parameters using this training batch's finalized gradients */
void UpdateNeuralNetworkTrainableParameters(unsigned int total_parameters, struct TrainableParameters *neural_network_parameters);
/* This function updates the batch normalization layer parameters using this training batch's finalized gradients */
void UpdateBatchNormalizationParameters(unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters);

/******************************************************************************************/
/*********************************** SAVE FINAL MODEL *************************************/
/******************************************************************************************/

/* This function saves the final model's trainable parameters */
void SaveFinalModelTrainableParameters(unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainableParameters *neural_network_kernel_weights, unsigned int *kernel_weights_offset_indices, struct TrainableParameters *neural_network_bias_weights, unsigned int *bias_weights_offset_indices, struct TrainableParameters *parametric_relu_alphas, unsigned int *parametric_relu_alpha_offset_indices, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, unsigned int *batch_normalization_layer_offset_indices);

/******************************************************************************************/
/************************************* DEBUG PRINT ****************************************/
/******************************************************************************************/
/* This function prints the FeedForward tensors if we are in debug print mode */
void DebugPrintFeedForward(unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, unsigned int batch_size, double *batch_features_tensor, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int *neural_network_neurons_minus_inputs_offset_indices, struct TrainableParameters *neural_network_kernel_weights, unsigned int *kernel_weights_offset_indices, struct TrainableParameters *neural_network_bias_weights, unsigned int *bias_weights_offset_indices, struct TrainableParameters *parametric_relu_alphas, unsigned int *parametric_relu_alpha_offset_indices, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, double *batch_normalization_layer_xmu, unsigned int *batch_normalization_layer_offset_indices);
/* This function prints the Backpropagation tensors if we are in debug print mode */
void DebugPrintBackpropagation(unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, unsigned int batch_size, double *batch_labels_tensor, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int *neural_network_neurons_minus_inputs_offset_indices, struct TrainableParameters *neural_network_kernel_weights, unsigned int *kernel_weights_offset_indices, struct TrainableParameters *neural_network_bias_weights, unsigned int *bias_weights_offset_indices, struct TrainableParameters *parametric_relu_alphas, unsigned int *parametric_relu_alpha_offset_indices, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, unsigned int *batch_normalization_layer_offset_indices);

/**************************************************************************************************************/
/**************************************************************************************************************/
/**************************************************** MAIN ****************************************************/
/**************************************************************************************************************/
/**************************************************************************************************************/

int main(int argc, char *argv[])
{
	/******************************************************************************************/
	/*********************************** OVERALL STRUCTURE ************************************/
	/******************************************************************************************/

	unsigned int i, j, k, array_index;

	/* Get number of hidden layers */
	unsigned int number_of_hidden_layers = 0;

	ReadFileTensor0DUnsignedInt("inputs/number_of_hidden_layers.txt", &number_of_hidden_layers);
	printf("main: number_of_hidden_layers = %u\n", number_of_hidden_layers);

	/* Get all neural network layer hyperparameters */
	unsigned int number_of_layers = number_of_hidden_layers + 2;

	struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters; // 0th layer is input layer, all others are dense layers
	neural_network_layers_hyperparameters = malloc(sizeof(struct NeuralNetworkLayerHyperparameters) * number_of_layers);

	ReadFileNeuralNetworkLayerHyperparameters("inputs/neural_network_layer_hyperparameters.txt", number_of_layers, neural_network_layers_hyperparameters);

	/* Get training hyperparameters */
	struct TrainingValidationHyperparameters *training_validation_hyperparameters;
	training_validation_hyperparameters = malloc(sizeof(struct TrainingValidationHyperparameters) * 1);

	ReadFileTrainingHyperparameters("inputs/training_validation_hyperparameters.txt", training_validation_hyperparameters);

	/******************************************************************************************/
	/********************************** TRAINABLE PARAMETERS **********************************/
	/******************************************************************************************/

	/* The initial accumulator value is different from zero for some gradient optimization algorithms */
	double initial_accumulator_value;
	if (training_validation_hyperparameters->optimizer == 3 || training_validation_hyperparameters->optimizer == 10) // adagrad or ftrl
	{
		initial_accumulator_value = training_validation_hyperparameters->optimizer_parameter0;
	}
	else
	{
		initial_accumulator_value = 0.0;
	}

	/* Declare inter-layer weights and biases */
	/* Kernels */
	unsigned int total_kernel_weights = 0;
	unsigned int *kernel_weights_offset_indices;
	kernel_weights_offset_indices = malloc(sizeof(unsigned int) * (number_of_layers - 1));
	for (i = 0; i < number_of_layers - 1; i++)
	{
		kernel_weights_offset_indices[i] = total_kernel_weights;
		printf("main: i = %u, kernel_weights_offset_indices[i] = %u\n", i, kernel_weights_offset_indices[i]);
		total_kernel_weights += neural_network_layers_hyperparameters[i].number_of_neurons * neural_network_layers_hyperparameters[i + 1].number_of_neurons;
	} // end of i loop

	struct TrainableParameters *neural_network_kernel_weights;
	neural_network_kernel_weights = malloc(sizeof(struct TrainableParameters) * total_kernel_weights);
	for (i = 0; i < number_of_layers - 1; i++)
	{
		printf("Weight matrix between layer %u and %u has shape [%u, %u]\n", i, i + 1, neural_network_layers_hyperparameters[i].number_of_neurons, neural_network_layers_hyperparameters[i + 1].number_of_neurons);
	}

	/* Biases */
	unsigned int total_bias_weights = 0;
	unsigned int *bias_weights_offset_indices;
	bias_weights_offset_indices = malloc(sizeof(unsigned int) * (number_of_layers - 1));
	for (i = 0; i < number_of_layers - 1; i++)
	{
		bias_weights_offset_indices[i] = total_bias_weights;
		printf("main: i = %u, bias_weights_offset_indices[i] = %u\n", i, bias_weights_offset_indices[i]);
		if (neural_network_layers_hyperparameters[i + 1].bias_weight_initialization_type >= 0)
		{
			total_bias_weights += neural_network_layers_hyperparameters[i + 1].number_of_neurons;
		}
	} // end of i loop

	struct TrainableParameters *neural_network_bias_weights;
	neural_network_bias_weights = malloc(sizeof(struct TrainableParameters) * total_bias_weights);

	/* Initialization of layer kernel and bias weights */
	/*
	name				initialization_type	initialization_parameter1	initialization_parameter2	initialization_parameter3
	zero				0					unused						unused						unused
	one					1					unused						unused						unused
	constant			2					constant					unused						unused
	random_uniform		3					range_min					range_max					unused
	random_normal		4					mu							sigma						unused
	truncated_normal	5					mu							sigma						unused
	variance_scaling	6					scale						mode						distribution
	 */

	InitializeKernelWeights(number_of_layers, total_kernel_weights, neural_network_layers_hyperparameters, initial_accumulator_value, neural_network_kernel_weights, kernel_weights_offset_indices);
	InitializeBiasWeights(number_of_layers, total_bias_weights, neural_network_layers_hyperparameters, initial_accumulator_value, neural_network_bias_weights, bias_weights_offset_indices);

	/* Create parametric ReLU alphas */
	unsigned int total_parametric_relu_neurons = 0;
	unsigned int *parametric_relu_alpha_offset_indices;
	parametric_relu_alpha_offset_indices = malloc(sizeof(unsigned int) * (number_of_layers - 1));
	for (i = 0; i < number_of_layers - 1; i++)
	{
		if (neural_network_layers_hyperparameters[i + 1].activation_type == 5) // parametric ReLU
		{
			parametric_relu_alpha_offset_indices[i] = total_parametric_relu_neurons;
			total_parametric_relu_neurons += neural_network_layers_hyperparameters[i + 1].number_of_neurons;
		}
	} // end of i loop

	struct TrainableParameters *parametric_relu_alphas;
	parametric_relu_alphas = malloc(sizeof(struct TrainableParameters) * total_parametric_relu_neurons);

	InitializeParametricReLUAlphas(number_of_layers, neural_network_layers_hyperparameters, initial_accumulator_value, parametric_relu_alphas, parametric_relu_alpha_offset_indices);

	/* Create batch normalization layers */
	unsigned int total_batch_normalization_neurons = 0;
	unsigned int *batch_normalization_layer_offset_indices;
	batch_normalization_layer_offset_indices = malloc(sizeof(unsigned int) * (number_of_layers - 1));
	for (i = 0; i < number_of_layers - 1; i++)
	{
		if (neural_network_layers_hyperparameters[i + 1].batch_norm_flag == 1)
		{
			batch_normalization_layer_offset_indices[i] = total_batch_normalization_neurons;
			total_batch_normalization_neurons += neural_network_layers_hyperparameters[i + 1].number_of_neurons;
		}
	} // end of i loop

	struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters;
	batch_normalization_layer_neurons_parameters = malloc(sizeof(struct BatchNormalizationLayerNeuronParameters) * total_batch_normalization_neurons);

	InitializeBatchNormalizationLayerParameters(number_of_layers, neural_network_layers_hyperparameters, initial_accumulator_value, batch_normalization_layer_neurons_parameters, batch_normalization_layer_offset_indices);

	/******************************************************************************************/
	/*************************************** INPUT DATA ***************************************/
	/******************************************************************************************/

	/* Get input data */
	unsigned int number_of_input_data_records = 0;
	ReadFileTensor0DUnsignedInt("inputs/number_of_input_data_records.txt", &number_of_input_data_records);

	printf("main: number_of_input_data_records = %u\n", number_of_input_data_records);

	double *features;
	double *labels;
	AllocateFeaturesAndLabels(number_of_input_data_records, neural_network_layers_hyperparameters[0].number_of_neurons, neural_network_layers_hyperparameters[number_of_layers - 1].number_of_neurons, &features, &labels);

	InputDataFunction("inputs/input_data.txt", number_of_input_data_records, neural_network_layers_hyperparameters[0].number_of_neurons, neural_network_layers_hyperparameters[number_of_layers - 1].number_of_neurons, &features, &labels, 0);

	/* Split data */
	unsigned int train_rows = (int)(number_of_input_data_records * training_validation_hyperparameters->train_percent), valid_rows = (int)(number_of_input_data_records * training_validation_hyperparameters->valid_percent), test_rows = number_of_input_data_records - train_rows - valid_rows;

	double *features_train;
	double *labels_train;
	AllocateFeaturesAndLabels(train_rows, neural_network_layers_hyperparameters[0].number_of_neurons, neural_network_layers_hyperparameters[number_of_layers - 1].number_of_neurons, &features_train, &labels_train);
	InitializeSplitFeaturesAndLabels(train_rows, neural_network_layers_hyperparameters[0].number_of_neurons, neural_network_layers_hyperparameters[number_of_layers - 1].number_of_neurons, 0, features, labels, features_train, labels_train, 0);

	double *features_valid;
	double *labels_valid;
	AllocateFeaturesAndLabels(valid_rows, neural_network_layers_hyperparameters[0].number_of_neurons, neural_network_layers_hyperparameters[number_of_layers - 1].number_of_neurons, &features_valid, &labels_valid);
	InitializeSplitFeaturesAndLabels(valid_rows, neural_network_layers_hyperparameters[0].number_of_neurons, neural_network_layers_hyperparameters[number_of_layers - 1].number_of_neurons, 0, features, labels, features_valid, labels_valid, 0);

	double *features_test;
	double *labels_test;
	AllocateFeaturesAndLabels(test_rows, neural_network_layers_hyperparameters[0].number_of_neurons, neural_network_layers_hyperparameters[number_of_layers - 1].number_of_neurons, &features_test, &labels_test);
	InitializeSplitFeaturesAndLabels(test_rows, neural_network_layers_hyperparameters[0].number_of_neurons, neural_network_layers_hyperparameters[number_of_layers - 1].number_of_neurons, 0, features, labels, features_test, labels_test, 0);

	/******************************************************************************************/
	/********************************* TRAINING AND EVALUATION ********************************/
	/******************************************************************************************/

	/* Train and evaluate neural network */
	MiniBatchGradientDescent(number_of_layers, neural_network_layers_hyperparameters, training_validation_hyperparameters, train_rows, features_train, labels_train, valid_rows, features_valid, labels_valid, neural_network_kernel_weights, kernel_weights_offset_indices, total_kernel_weights, neural_network_bias_weights, bias_weights_offset_indices, total_bias_weights, parametric_relu_alphas, parametric_relu_alpha_offset_indices, total_parametric_relu_neurons, batch_normalization_layer_neurons_parameters, batch_normalization_layer_offset_indices, total_batch_normalization_neurons, 0);

	/******************************************************************************************/
	/************************************ SAVE FINAL MODEL ************************************/
	/******************************************************************************************/

	SaveFinalModelTrainableParameters(number_of_layers, neural_network_layers_hyperparameters, neural_network_kernel_weights, kernel_weights_offset_indices, neural_network_bias_weights, bias_weights_offset_indices, parametric_relu_alphas, parametric_relu_alpha_offset_indices, batch_normalization_layer_neurons_parameters, batch_normalization_layer_offset_indices);

	/******************************************************************************************/
	/*************************** FREE DYNAMICALLY ALLOCATED MEMORY ****************************/
	/******************************************************************************************/

	/* Free dynamic array memory */
	free(labels_test);
	free(features_test);
	free(labels_valid);
	free(features_valid);
	free(labels_train);
	free(features_train);
	free(labels);
	free(features);

	free(batch_normalization_layer_neurons_parameters);
	free(batch_normalization_layer_offset_indices);

	free(parametric_relu_alphas);
	free(parametric_relu_alpha_offset_indices);

	free(neural_network_bias_weights);
	free(bias_weights_offset_indices);

	free(neural_network_kernel_weights);
	free(kernel_weights_offset_indices);

	free(training_validation_hyperparameters);
	free(neural_network_layers_hyperparameters);

	return 0;
} // end of main function

/**************************************************************************************************************/
/**************************************************************************************************************/
/************************************************* FUNCTIONS **************************************************/
/**************************************************************************************************************/
/**************************************************************************************************************/

/******************************************************************************************/
/*********************************** OVERALL STRUCTURE ************************************/
/******************************************************************************************/

/* This function reads a 0d unsigned int from the given filepath */
void ReadFileTensor0DUnsignedInt(char *filepath, unsigned int *tensor)
{
	printf("ReadFileTensor0DUnsignedInt: filepath = %s\n", filepath);

	FILE *infile = fopen(filepath, "r");
	if (infile == NULL)
	{
		printf("Failed to open file %s\n", filepath);
	}
	else
	{
		int systemreturn;
		systemreturn = fscanf(infile, "%u", tensor);
		if (systemreturn == -1)
		{
			printf("ReadFileTensor0DUnsignedInt: Failed reading file %s\n", filepath);
		}
		fclose(infile);
	}
} // end of ReadFileTensor0DUnsignedInt function

/* This function reads the neural network layer hyperparameters from the given filepath */
void ReadFileNeuralNetworkLayerHyperparameters(char *filepath, unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters)
{
	printf("ReadFileNeuralNetworkLayerHyperparameters: filepath = %s\n", filepath);

	FILE *infile = fopen(filepath, "r");
	if (infile == NULL)
	{
		printf("ReadFileNeuralNetworkLayerHyperparameters: Failed to open file %s\n", filepath);
	}
	else
	{
		int systemreturn;
		unsigned int i;

		printf("ReadFileNeuralNetworkLayerHyperparameters: NeuralNetworkLayerHyperparameters:\n");
		printf("layer\tnumber_of_neurons\tactivation_type\tactivation_function_alpha_initializer\t");
		printf("kernel_weight_initialization_type\tkernel_weight_initialization_parameter0\tkernel_weight_initialization_parameter1\tkernel_weight_initialization_parameter2\t");
		printf("kernel_l1_regularization_strength\tkernel_l2_regularization_strength\t");
		printf("bias_weight_initialization_type\tbias_weight_initialization_parameter0\tbias_weight_initialization_parameter1\t");
		printf("bias_l1_regularization_strength\tbias_l2_regularization_strength\t");
		printf("dropout_probability\tbatch_norm_flag\tbatch_norm_after_activation_flag\t");
		printf("batch_norm_momentum\tbatch_norm_moving_mean_initializer\tbatch_norm_moving_variance_initializer\tbatch_norm_beta_initializer\tbatch_norm_gamma_initializer\n");

		for (i = 0; i < number_of_layers; i++)
		{
			systemreturn = fscanf(infile, "%u\t%d\t%lf\t", &neural_network_layers_hyperparameters[i].number_of_neurons, &neural_network_layers_hyperparameters[i].activation_type, &neural_network_layers_hyperparameters[i].activation_function_alpha_initializer);
			if (systemreturn == -1)
			{
				printf("ReadFileNeuralNetworkLayerHyperparameters: Failed reading file %s\n", filepath);
			}
			printf("%u\t%u\t%d\t%.16f\t", i, neural_network_layers_hyperparameters[i].number_of_neurons, neural_network_layers_hyperparameters[i].activation_type, neural_network_layers_hyperparameters[i].activation_function_alpha_initializer);

			systemreturn = fscanf(infile, "%d\t%lf\t%lf\t%lf\t", &neural_network_layers_hyperparameters[i].kernel_weight_initialization_type, &neural_network_layers_hyperparameters[i].kernel_weight_initialization_parameter0, &neural_network_layers_hyperparameters[i].kernel_weight_initialization_parameter1, &neural_network_layers_hyperparameters[i].kernel_weight_initialization_parameter2);
			if (systemreturn == -1)
			{
				printf("ReadFileNeuralNetworkLayerHyperparameters: Failed reading file %s\n", filepath);
			}
			printf("%d\t%.16f\t%.16f\t%.16f\t", neural_network_layers_hyperparameters[i].kernel_weight_initialization_type, neural_network_layers_hyperparameters[i].kernel_weight_initialization_parameter0, neural_network_layers_hyperparameters[i].kernel_weight_initialization_parameter1, neural_network_layers_hyperparameters[i].kernel_weight_initialization_parameter2);

			systemreturn = fscanf(infile, "%lf\t%lf\t", &neural_network_layers_hyperparameters[i].kernel_l1_regularization_strength, &neural_network_layers_hyperparameters[i].kernel_l2_regularization_strength);
			if (systemreturn == -1)
			{
				printf("ReadFileNeuralNetworkLayerHyperparameters: Failed reading file %s\n", filepath);
			}
			printf("%.16f\t%.16f\t", neural_network_layers_hyperparameters[i].kernel_l1_regularization_strength, neural_network_layers_hyperparameters[i].kernel_l2_regularization_strength);

			systemreturn = fscanf(infile, "%d\t%lf\t%lf\t", &neural_network_layers_hyperparameters[i].bias_weight_initialization_type, &neural_network_layers_hyperparameters[i].bias_weight_initialization_parameter0, &neural_network_layers_hyperparameters[i].bias_weight_initialization_parameter1);
			if (systemreturn == -1)
			{
				printf("ReadFileNeuralNetworkLayerHyperparameters: Failed reading file %s\n", filepath);
			}
			printf("%d\t%.16f\t%.16f\t", neural_network_layers_hyperparameters[i].bias_weight_initialization_type, neural_network_layers_hyperparameters[i].bias_weight_initialization_parameter0, neural_network_layers_hyperparameters[i].bias_weight_initialization_parameter1);

			systemreturn = fscanf(infile, "%lf\t%lf\t", &neural_network_layers_hyperparameters[i].bias_l1_regularization_strength, &neural_network_layers_hyperparameters[i].bias_l2_regularization_strength);
			if (systemreturn == -1)
			{
				printf("ReadFileNeuralNetworkLayerHyperparameters: Failed reading file %s\n", filepath);
			}
			printf("%.16f\t%.16f\t", neural_network_layers_hyperparameters[i].bias_l1_regularization_strength, neural_network_layers_hyperparameters[i].bias_l2_regularization_strength);

			systemreturn = fscanf(infile, "%lf\t%d\t%d\t", &neural_network_layers_hyperparameters[i].dropout_probability, &neural_network_layers_hyperparameters[i].batch_norm_flag, &neural_network_layers_hyperparameters[i].batch_norm_after_activation_flag);
			if (systemreturn == -1)
			{
				printf("ReadFileNeuralNetworkLayerHyperparameters: Failed reading file %s\n", filepath);
			}
			printf("%.16f\t%d\t%d\t", neural_network_layers_hyperparameters[i].dropout_probability, neural_network_layers_hyperparameters[i].batch_norm_flag, neural_network_layers_hyperparameters[i].batch_norm_after_activation_flag);

			systemreturn = fscanf(infile, "%lf\t%lf\t%lf\t%lf\t%lf\n", &neural_network_layers_hyperparameters[i].batch_norm_momentum, &neural_network_layers_hyperparameters[i].batch_norm_moving_mean_initializer, &neural_network_layers_hyperparameters[i].batch_norm_moving_variance_initializer, &neural_network_layers_hyperparameters[i].batch_norm_beta_initializer, &neural_network_layers_hyperparameters[i].batch_norm_gamma_initializer);
			if (systemreturn == -1)
			{
				printf("ReadFileNeuralNetworkLayerHyperparameters: Failed reading file %s\n", filepath);
			}
			printf("%.16f\t%.16f\t%.16f\t%.16f\t%.16f\n", neural_network_layers_hyperparameters[i].batch_norm_momentum, neural_network_layers_hyperparameters[i].batch_norm_moving_mean_initializer, neural_network_layers_hyperparameters[i].batch_norm_moving_variance_initializer, neural_network_layers_hyperparameters[i].batch_norm_beta_initializer, neural_network_layers_hyperparameters[i].batch_norm_gamma_initializer);
		} // end of i loop
		fclose(infile);
	}
} // end of ReadFileNeuralNetworkLayerHyperparameters function

/* This function reads the training hyperparameters from the given filepath */
void ReadFileTrainingHyperparameters(char *filepath, struct TrainingValidationHyperparameters *training_validation_hyperparameters)
{
	printf("ReadFileTrainingHyperparameters: filepath = %s\n", filepath);

	FILE *infile = fopen(filepath, "r");
	if (infile == NULL)
	{
		printf("ReadFileTrainingHyperparameters: Failed to open file %s\n", filepath);
	}
	else
	{
		int systemreturn;

		printf("ReadFileTrainingHyperparameters: training_validation_hyperparameters:\n");
		printf("loss_function_type\talpha_forgiveness_rate\tbeta_false_negative_cost\tgamma_false_positive_cost\t");
		printf("train_percent\tvalid_percent\t");
		printf("train_steps\teval_steps\t");
		printf("batch_size\tlearning_rate\t");
		printf("clip_norm\t");
		printf("optimizer\toptimizer_parameter0\toptimizer_parameter1\n");

		systemreturn = fscanf(infile, "%u\t%lf\t%lf\t%lf\t%lf\t", &training_validation_hyperparameters->loss_function_type, &training_validation_hyperparameters->alpha_forgiveness_rate, &training_validation_hyperparameters->classification_threshold, &training_validation_hyperparameters->beta_false_negative_cost, &training_validation_hyperparameters->gamma_false_positive_cost);
		if (systemreturn == -1)
		{
			printf("ReadFileTrainingHyperparameters: Failed reading file %s\n", filepath);
		}
		printf("%u\t%.16f\t%.16f\t%.16f\t%.16f\t", training_validation_hyperparameters->loss_function_type, training_validation_hyperparameters->classification_threshold, training_validation_hyperparameters->alpha_forgiveness_rate, training_validation_hyperparameters->beta_false_negative_cost, training_validation_hyperparameters->gamma_false_positive_cost);

		systemreturn = fscanf(infile, "%lf\t%lf\t", &training_validation_hyperparameters->train_percent, &training_validation_hyperparameters->valid_percent);
		if (systemreturn == -1)
		{
			printf("ReadFileTrainingHyperparameters: Failed reading file %s\n", filepath);
		}
		printf("%.16f\t%.16f\t", training_validation_hyperparameters->train_percent, training_validation_hyperparameters->valid_percent);

		systemreturn = fscanf(infile, "%u\t%u\t", &training_validation_hyperparameters->train_steps, &training_validation_hyperparameters->eval_steps);
		if (systemreturn == -1)
		{
			printf("ReadFileTrainingHyperparameters: Failed reading file %s\n", filepath);
		}
		printf("%u\t%u\t", training_validation_hyperparameters->train_steps, training_validation_hyperparameters->eval_steps);

		systemreturn = fscanf(infile, "%u\t%lf\t", &training_validation_hyperparameters->batch_size, &training_validation_hyperparameters->learning_rate);
		if (systemreturn == -1)
		{
			printf("ReadFileTrainingHyperparameters: Failed reading file %s\n", filepath);
		}
		printf("%u\t%.16f\t", training_validation_hyperparameters->batch_size, training_validation_hyperparameters->learning_rate);

		systemreturn = fscanf(infile, "%lf\t", &training_validation_hyperparameters->clip_norm);
		if (systemreturn == -1)
		{
			printf("ReadFileTrainingHyperparameters: Failed reading file %s\n", filepath);
		}
		printf("%.16f\t", training_validation_hyperparameters->clip_norm);

		systemreturn = fscanf(infile, "%d\t%lf\t%lf\t%lf\t%lf\t%lf\n", &training_validation_hyperparameters->optimizer, &training_validation_hyperparameters->optimizer_parameter0, &training_validation_hyperparameters->optimizer_parameter1, &training_validation_hyperparameters->optimizer_parameter2, &training_validation_hyperparameters->optimizer_parameter3, &training_validation_hyperparameters->optimizer_parameter4);
		if (systemreturn == -1)
		{
			printf("ReadFileTrainingHyperparameters: Failed reading file %s\n", filepath);
		}
		printf("%d\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\n", training_validation_hyperparameters->optimizer, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1, training_validation_hyperparameters->optimizer_parameter2, training_validation_hyperparameters->optimizer_parameter3, training_validation_hyperparameters->optimizer_parameter4);
		fclose(infile);
	}
} // end of ReadFileTrainingHyperparameters function

/******************************************************************************************/
/********************************** TRAINABLE PARAMETERS **********************************/
/******************************************************************************************/

/* This function initializes kernel weights for each layer depending on the kernel weight initialization type */
void InitializeKernelWeights(unsigned int number_of_layers, unsigned int total_kernel_weights, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, double initial_accumulator_value, struct TrainableParameters *neural_network_kernel_weights, unsigned int *kernel_weights_offset_indices)
{
	/* Initialization of weights and biases*/
	/*
	name				initialization_type	initialization_parameter1	initialization_parameter2	initialization_parameter3
	zero				0					unused						unused						unused
	one					1					unused						unused						unused
	constant			2					constant					unused						unused
	random_uniform		3					range_min					range_max					unused
	random_normal		4					mu							sigma						unused
	truncated_normal	5					mu							sigma						unused
	variance_scaling	6					scale						mode						distribution
	 */

	unsigned int i;

	/* Initialize kernel weight variable values */
	for (i = 0; i < number_of_layers - 1; i++)
	{
		switch (neural_network_layers_hyperparameters[i + 1].kernel_weight_initialization_type)
		{
			case 0:
				KernelWeightInitializationConstant(i + 1, neural_network_layers_hyperparameters, neural_network_kernel_weights, kernel_weights_offset_indices[i], 0.0);
				break;
			case 1:
				KernelWeightInitializationConstant(i + 1, neural_network_layers_hyperparameters, neural_network_kernel_weights, kernel_weights_offset_indices[i], 1.0);
				break;
			case 2:
				KernelWeightInitializationConstant(i + 1, neural_network_layers_hyperparameters, neural_network_kernel_weights, kernel_weights_offset_indices[i], neural_network_layers_hyperparameters[i + 1].kernel_weight_initialization_parameter0);
				break;
			case 3:
				KernelWeightInitializationRandomUniform(i + 1, neural_network_layers_hyperparameters, neural_network_kernel_weights, kernel_weights_offset_indices[i], neural_network_layers_hyperparameters[i + 1].kernel_weight_initialization_parameter0, neural_network_layers_hyperparameters[i + 1].kernel_weight_initialization_parameter1);
				break;
			case 4:
				KernelWeightInitializationRandomNormal(i + 1, neural_network_layers_hyperparameters, neural_network_kernel_weights, kernel_weights_offset_indices[i], neural_network_layers_hyperparameters[i + 1].kernel_weight_initialization_parameter0, neural_network_layers_hyperparameters[i + 1].kernel_weight_initialization_parameter1);
				break;
			case 5:
				KernelWeightInitializationTruncatedNormal(i + 1, neural_network_layers_hyperparameters, neural_network_kernel_weights, kernel_weights_offset_indices[i], neural_network_layers_hyperparameters[i + 1].kernel_weight_initialization_parameter0, neural_network_layers_hyperparameters[i + 1].kernel_weight_initialization_parameter1);
				break;
			case 6:
				KernelWeightInitializationVarianceScaling(i + 1, neural_network_layers_hyperparameters, neural_network_kernel_weights, kernel_weights_offset_indices[i], neural_network_layers_hyperparameters[i + 1].kernel_weight_initialization_parameter0, neural_network_layers_hyperparameters[i + 1].kernel_weight_initialization_parameter1, neural_network_layers_hyperparameters[i + 1].kernel_weight_initialization_parameter2);
				break;
			default:
				KernelWeightInitializationConstant(i + 1, neural_network_layers_hyperparameters, neural_network_kernel_weights, kernel_weights_offset_indices[i], 0.0); // initialize to zero since no valid type given
				printf("InitializeKernelWeights: No weight initialization for layer %u\n", i + 1);
		}
	} // end of i loop

	/* Initialize other kernel weight values */
	for (i = 0; i < total_kernel_weights; i++)
	{
		InitializeTrainableParameterGradientsAndAggregates(initial_accumulator_value, &neural_network_kernel_weights[i].gradient, &neural_network_kernel_weights[i].aggregate0, &neural_network_kernel_weights[i].aggregate1, &neural_network_kernel_weights[i].aggregate2, &neural_network_kernel_weights[i].update);
	} // end of i loop
} // end of InitializeKernelWeights function

/* This function initializes kernel weights in layer to a given constant number */
void KernelWeightInitializationConstant(unsigned int layer_index, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainableParameters *neural_network_kernel_weights, unsigned int offset, double constant)
{
	printf("KernelWeightInitializationConstant: Initializing layer %u\n", layer_index - 1);
	unsigned int i, j, weight_index;

	for (i = 0; i < neural_network_layers_hyperparameters[layer_index - 1].number_of_neurons; i++)
	{
		for (j = 0; j < neural_network_layers_hyperparameters[layer_index].number_of_neurons; j++)
		{
			weight_index = i * neural_network_layers_hyperparameters[layer_index].number_of_neurons + j + offset;

			neural_network_kernel_weights[weight_index].variable = constant;
		} // end of j loop
	} // end of i loop
} // end of KernelWeightInitializationConstant function

/* This function initializes kernel weights in layer to a random uniform number within a given range */
void KernelWeightInitializationRandomUniform(unsigned int layer_index, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainableParameters *neural_network_kernel_weights, unsigned int offset, double range_min, double range_max)
{
	printf("KernelWeightInitializationRandomUniform: Initializing layer %u\n", layer_index - 1);
	unsigned int i, j, weight_index;

	for (i = 0; i < neural_network_layers_hyperparameters[layer_index - 1].number_of_neurons; i++)
	{
		for (j = 0; j < neural_network_layers_hyperparameters[layer_index].number_of_neurons; j++)
		{
			weight_index = i * neural_network_layers_hyperparameters[layer_index].number_of_neurons + j + offset;

			neural_network_kernel_weights[weight_index].variable = RUnif(range_min, range_max);
		} // end of j loop
	} // end of i loop
} // end of KernelWeightInitializationRandomUniform function

/* This function initializes kernel weights in layer to a random normal number with a given mean and standard deviation */
void KernelWeightInitializationRandomNormal(unsigned int layer_index, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainableParameters *neural_network_kernel_weights, unsigned int offset, double mu, double sigma)
{
	printf("KernelWeightInitializationRandomNormal: Initializing layer %u\n", layer_index - 1);
	unsigned int i, j, weight_index;

	for (i = 0; i < neural_network_layers_hyperparameters[layer_index - 1].number_of_neurons; i++)
	{
		for (j = 0; j < neural_network_layers_hyperparameters[layer_index].number_of_neurons; j++)
		{
			weight_index = i * neural_network_layers_hyperparameters[layer_index].number_of_neurons + j + offset;

			neural_network_kernel_weights[weight_index].variable = RNorm(mu, sigma);
		} // end of j loop
	} // end of i loop
} // end of KernelWeightInitializationRandomNormal function

/* This function initializes kernel weights in layer to a random normal number with a given mean and standard deviation within two standard deviations */
void KernelWeightInitializationTruncatedNormal(unsigned int layer_index, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainableParameters *neural_network_kernel_weights, unsigned int offset, double mu, double sigma)
{
	printf("KernelWeightInitializationTruncatedNormal: Initializing layer %u\n", layer_index - 1);
	unsigned int i, j, weight_index;

	for (i = 0; i < neural_network_layers_hyperparameters[layer_index - 1].number_of_neurons; i++)
	{
		for (j = 0; j < neural_network_layers_hyperparameters[layer_index].number_of_neurons; j++)
		{
			weight_index = i * neural_network_layers_hyperparameters[layer_index].number_of_neurons + j + offset;

			do
			{
				neural_network_kernel_weights[weight_index].variable = RNorm(mu, sigma);
			} while (fabs((neural_network_kernel_weights[weight_index].variable - mu) / sigma) > 2.0);
		} // end of j loop
	} // end of i loop
} // end of KernelWeightInitializationTruncatedNormal function

/* This function initializes kernel weights in layer to a random normal or uniform number with a given parameters using variance scaling */
void KernelWeightInitializationVarianceScaling(unsigned int layer_index, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainableParameters *neural_network_kernel_weights, unsigned int offset, double scale, double mode, double distribution)
{
	printf("KernelWeightInitializationVarianceScaling: Initializing layer %u\n", layer_index - 1);
	unsigned int i, j, weight_index;
	double n = 1.0, sigma = 1.0, range_min = 0.0, range_max = 1.0;

	switch ((int)distribution)
	{
		case 0: // normal
			switch ((int)mode)
			{
				case 0: // fan-in, n = number of input units, same as lecun_normal with a scale of 1. or he_normal with a scale of 2.
					n = neural_network_layers_hyperparameters[layer_index - 1].number_of_neurons;
					break;
				case 1: // fan-out, n = number of output units
					n = neural_network_layers_hyperparameters[layer_index].number_of_neurons;
					break;
				case 2: // average, n = average number of input and output units, same as glorot_normal and xavier_normal with scale of 1.
					n = (neural_network_layers_hyperparameters[layer_index - 1].number_of_neurons + neural_network_layers_hyperparameters[layer_index].number_of_neurons) * 0.5;
					break;
			}
			sigma = sqrt(scale / n);

			for (i = 0; i < neural_network_layers_hyperparameters[layer_index - 1].number_of_neurons; i++)
			{
				for (j = 0; j < neural_network_layers_hyperparameters[layer_index].number_of_neurons; j++)
				{
					weight_index = i * neural_network_layers_hyperparameters[layer_index].number_of_neurons + j + offset;

					do
					{
						neural_network_kernel_weights[weight_index].variable = RNorm(0., sigma);
					} while (fabs(neural_network_kernel_weights[weight_index].variable / sigma) > 2.0);
				} // end of j loop
			} // end of i loop
			break;
		case 1: // uniform
			switch ((int)mode)
			{
				case 0: // fan-in, n = number of input units, same as lecun_uniform with scale of 1. or he_uniform with a scale of 2.
					n = neural_network_layers_hyperparameters[layer_index - 1].number_of_neurons;
					break;
				case 1: // fan-out, n = number of output units
					n = neural_network_layers_hyperparameters[layer_index].number_of_neurons;
					break;
				case 2: // average, n = average number of input and output units, same as glorot_uniform with scale of 1.
					n = (neural_network_layers_hyperparameters[layer_index - 1].number_of_neurons + neural_network_layers_hyperparameters[layer_index].number_of_neurons) * 0.5;
					break;
			}
			range_min = -sqrt(3.0 * scale / n);
			range_max = sqrt(3.0 * scale / n);

			for (i = 0; i < neural_network_layers_hyperparameters[layer_index - 1].number_of_neurons; i++)
			{
				for (j = 0; j < neural_network_layers_hyperparameters[layer_index].number_of_neurons; j++)
				{
					weight_index = i * neural_network_layers_hyperparameters[layer_index].number_of_neurons + j + offset;

					neural_network_kernel_weights[weight_index].variable = RUnif(range_min, range_max);
				} // end of j loop
			} // end of i loop
			break;
	}
} // end of KernelWeightInitializationVarianceScaling function

/* This function initializes bias weights for each layer depending on the bias weight initialization type */
void InitializeBiasWeights(unsigned int number_of_layers, unsigned int total_bias_weights, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, double initial_accumulator_value, struct TrainableParameters *neural_network_bias_weights, unsigned int *bias_weights_offset_indices)
{
	/* Initialization of weights and biases*/
	/*
	name				initialization_type	initialization_parameter1	initialization_parameter2	initialization_parameter3
	zero				0					unused						unused						unused
	one					1					unused						unused						unused
	constant			2					constant					unused						unused
	random_uniform		3					range_min					range_max					unused
	random_normal		4					mu							sigma						unused
	truncated_normal	5					mu							sigma						unused
	variance_scaling	6					scale						mode						distribution
	 */

	unsigned int i;

	/* Initialize bias weight variable values */
	for (i = 0; i < number_of_layers - 1; i++)
	{
		switch (neural_network_layers_hyperparameters[i + 1].bias_weight_initialization_type)
		{
			case 0:
				BiasWeightInitializationConstant(i + 1, neural_network_layers_hyperparameters, neural_network_bias_weights, bias_weights_offset_indices[i], 0.0);
				break;
			case 1:
				BiasWeightInitializationConstant(i + 1, neural_network_layers_hyperparameters, neural_network_bias_weights, bias_weights_offset_indices[i], 1.0);
				break;
			case 2:
				BiasWeightInitializationConstant(i + 1, neural_network_layers_hyperparameters, neural_network_bias_weights, bias_weights_offset_indices[i], neural_network_layers_hyperparameters[i + 1].bias_weight_initialization_parameter0);
				break;
			case 3:
				BiasWeightInitializationRandomUniform(i + 1, neural_network_layers_hyperparameters, neural_network_bias_weights, bias_weights_offset_indices[i], neural_network_layers_hyperparameters[i + 1].bias_weight_initialization_parameter0, neural_network_layers_hyperparameters[i + 1].bias_weight_initialization_parameter1);
				break;
			case 4:
				BiasWeightInitializationRandomNormal(i + 1, neural_network_layers_hyperparameters, neural_network_bias_weights, bias_weights_offset_indices[i], neural_network_layers_hyperparameters[i + 1].bias_weight_initialization_parameter0, neural_network_layers_hyperparameters[i + 1].bias_weight_initialization_parameter1);
				break;
			case 5:
				BiasWeightInitializationTruncatedNormal(i + 1, neural_network_layers_hyperparameters, neural_network_bias_weights, bias_weights_offset_indices[i], neural_network_layers_hyperparameters[i + 1].bias_weight_initialization_parameter0, neural_network_layers_hyperparameters[i + 1].bias_weight_initialization_parameter1);
				break;
			default:
				BiasWeightInitializationConstant(i + 1, neural_network_layers_hyperparameters, neural_network_bias_weights, bias_weights_offset_indices[i], 0.0); // initialize to zero, even though it will be unused
				printf("InitializeBiasWeights: No bias initialization for layer %u\n", i + 1);
		}
	}

	/* Initialize other bias weight values */
	for (i = 0; i < total_bias_weights; i++)
	{
		InitializeTrainableParameterGradientsAndAggregates(initial_accumulator_value, &neural_network_bias_weights[i].gradient, &neural_network_bias_weights[i].aggregate0, &neural_network_bias_weights[i].aggregate1, &neural_network_bias_weights[i].aggregate2, &neural_network_bias_weights[i].update);
	} // end of i loop
} // end of InitializeBiasWeights function

/* This function initializes bias weights in layer to a given constant number */
void BiasWeightInitializationConstant(unsigned int layer_index, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainableParameters *neural_network_bias_weights, unsigned int offset, double constant)
{
	printf("BiasWeightInitializationConstant: Initializing layer %u\n", layer_index - 1);
	unsigned int i, weight_index;

	for (i = 0; i < neural_network_layers_hyperparameters[layer_index].number_of_neurons; i++)
	{
		weight_index = i + offset;

		neural_network_bias_weights[weight_index].variable = constant;
	} // end of i loop
} // end of BiasWeightInitializationConstant function

/* This function initializes bias weights in layer to a random uniform number within a given range */
void BiasWeightInitializationRandomUniform(unsigned int layer_index, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainableParameters *neural_network_bias_weights, unsigned int offset, double range_min, double range_max)
{
	printf("BiasWeightInitializationRandomUniform: Initializing layer %u\n", layer_index - 1);
	unsigned int i, weight_index;

	for (i = 0; i < neural_network_layers_hyperparameters[layer_index].number_of_neurons; i++)
	{
		weight_index = i + offset;

		neural_network_bias_weights[weight_index].variable = RUnif(range_min, range_max);
	} // end of i loop
} // end of BiasWeightInitializationRandomUniform function

/* This function initializes bias weights in layer to a random normal number with a given mean and standard deviation */
void BiasWeightInitializationRandomNormal(unsigned int layer_index, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainableParameters *neural_network_bias_weights, unsigned int offset, double mu, double sigma)
{
	printf("BiasWeightInitializationRandomNormal: Initializing layer %u\n", layer_index - 1);
	unsigned int i, weight_index;

	for (i = 0; i < neural_network_layers_hyperparameters[layer_index].number_of_neurons; i++)
	{
		weight_index = i + offset;

		neural_network_bias_weights[weight_index].variable = RNorm(mu, sigma);
	} // end of i loop
} // end of BiasWeightInitializationRandomNormal function

/* This function initializes bias weights in layer to a random normal number with a given mean and standard deviation within two standard deviations */
void BiasWeightInitializationTruncatedNormal(unsigned int layer_index, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainableParameters *neural_network_bias_weights, unsigned int offset, double mu, double sigma)
{
	printf("BiasWeightInitializationTruncatedNormal: Initializing layer %u\n", layer_index - 1);
	unsigned int i, weight_index;

	for (i = 0; i < neural_network_layers_hyperparameters[layer_index].number_of_neurons; i++)
	{
		weight_index = i + offset;

		do
		{
			neural_network_bias_weights[weight_index].variable = RNorm(mu, sigma);
		} while (fabs((neural_network_bias_weights[weight_index].variable - mu) / sigma) > 2.0);
	} // end of i loop
} // end of BiasWeightInitializationTruncatedNormal function

/* This function returns a random uniform number within given range */
double RUnif(double range_min, double range_max)
{
	return range_min + (range_max - range_min) * UnifRand();
} // end of RUnif function

/* This function returns a random normal number with given mean and standard deviation */
double RNorm(double mu, double sigma)
{
	if (sigma == 0.0)
	{
		return mu;
	}
	else
	{
		return mu + sigma * NormRand();
	}
} // end of RNorm function

/* This function returns a random uniform number within range [0,1] */
double UnifRand(void)
{
	return (double)rand() / (double)RAND_MAX;
}	// end of UnifRand function

/* This function returns a random normal number with zero mean and unit standard deviation */
double NormRand(void)
{
	const static double a[32] =
	{
		0.0000000, 0.03917609, 0.07841241, 0.1177699,
		0.1573107, 0.19709910, 0.23720210, 0.2776904,
		0.3186394, 0.36012990, 0.40225010, 0.4450965,
		0.4887764, 0.53340970, 0.57913220, 0.6260990,
		0.6744898, 0.72451440, 0.77642180, 0.8305109,
		0.8871466, 0.94678180, 1.00999000, 1.0775160,
		1.1503490, 1.22985900, 1.31801100, 1.4177970,
		1.5341210, 1.67594000, 1.86273200, 2.1538750
	};

	const static double d[31] =
	{
		0.0000000, 0.0000000, 0.0000000, 0.0000000,
		0.0000000, 0.2636843, 0.2425085, 0.2255674,
		0.2116342, 0.1999243, 0.1899108, 0.1812252,
		0.1736014, 0.1668419, 0.1607967, 0.1553497,
		0.1504094, 0.1459026, 0.1417700, 0.1379632,
		0.1344418, 0.1311722, 0.1281260, 0.1252791,
		0.1226109, 0.1201036, 0.1177417, 0.1155119,
		0.1134023, 0.1114027, 0.1095039
	};

	const static double t[31] =
	{
		7.673828e-4, 0.002306870, 0.003860618, 0.005438454,
		0.007050699, 0.008708396, 0.010423570, 0.012209530,
		0.014081250, 0.016055790, 0.018152900, 0.020395730,
		0.022811770, 0.025434070, 0.028302960, 0.031468220,
		0.034992330, 0.038954830, 0.043458780, 0.048640350,
		0.054683340, 0.061842220, 0.070479830, 0.081131950,
		0.094624440, 0.112300100, 0.136498000, 0.171688600,
		0.227624100, 0.330498000, 0.584703100
	};

	const static double h[31] =
	{
		0.03920617, 0.03932705, 0.03950999, 0.03975703,
		0.04007093, 0.04045533, 0.04091481, 0.04145507,
		0.04208311, 0.04280748, 0.04363863, 0.04458932,
		0.04567523, 0.04691571, 0.04833487, 0.04996298,
		0.05183859, 0.05401138, 0.05654656, 0.05953130,
		0.06308489, 0.06737503, 0.07264544, 0.07926471,
		0.08781922, 0.09930398, 0.11555990, 0.14043440,
		0.18361420, 0.27900160, 0.70104740
	};

	double s, u1, w, y, u2, aa, tt;
	int i;

	u1 = UnifRand();
	s = 0.0;
	if (u1 > 0.5)
	{
		s = 1.0;
	}
	u1 = u1 + u1 - s;
	u1 *= 32.0;
	i = (int) u1;
	if (i == 32)
	{
		i = 31;
	}

	if (i != 0)
	{
		u2 = u1 - i;
		aa = a[i - 1];
		while (u2 <= t[i - 1])
		{
			u1 = UnifRand();
			w = u1 * (a[i] - aa);
			tt = (w * 0.5 + aa) * w;
			for(;;)
			{
				if (u2 > tt)
				{
					goto deliver;
				}
				u1 = UnifRand();
				if (u2 < u1)
				{
					break;
				}
				tt = u1;
				u2 = UnifRand();
			}
			u2 = UnifRand();
		}
		w = (u2 - t[i - 1]) * h[i - 1];
	}
	else
	{
		i = 6;
		aa = a[31];
		for(;;)
		{
			u1 = u1 + u1;
			if (u1 >= 1.0)
			{
				break;
			}
			aa = aa + d[i - 1];
			i = i + 1;
		}
		u1 = u1 - 1.0;
		for(;;)
		{
			w = u1 * d[i - 1];
			tt = (w * 0.5 + aa) * w;
			for(;;)
			{
				u2 = UnifRand();
				if (u2 > tt)
				{
					goto jump;
				}
				u1 = UnifRand();
				if (u2 < u1)
				{
					break;
				}
				tt = u1;
			}
			u1 = UnifRand();
		}
		jump:;
	}

	deliver:
	y = aa + w;
	return (s == 1.0) ? -y : y;
}	// end of NormRand function

/* This function initializes parametric ReLU alphas for each layer depending on the activation function alpha initializer */
void InitializeParametricReLUAlphas(unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, double initial_accumulator_value, struct TrainableParameters *parametric_relu_alphas, unsigned int *parametric_relu_alpha_offset_indices)
{
	unsigned int i, j, array_index;

	for (i = 0; i < number_of_layers - 1; i++)
	{
		if (neural_network_layers_hyperparameters[i + 1].activation_type == 5) // parametric ReLU
		{
			for (j = 0; j < neural_network_layers_hyperparameters[i + 1].number_of_neurons; j++)
			{
				array_index = j + parametric_relu_alpha_offset_indices[i];

				parametric_relu_alphas[array_index].variable = neural_network_layers_hyperparameters[i + 1].activation_function_alpha_initializer;
				InitializeTrainableParameterGradientsAndAggregates(initial_accumulator_value, &parametric_relu_alphas[array_index].gradient, &parametric_relu_alphas[array_index].aggregate0, &parametric_relu_alphas[array_index].aggregate1, &parametric_relu_alphas[array_index].aggregate2, &parametric_relu_alphas[array_index].update);
			} // end of j loop
		}
	} // end of i loop
} // end of InitializeParametricReLUAlphas function

/* This function initializes batch normalization layer parameters for each layer depending on several initializers */
void InitializeBatchNormalizationLayerParameters(unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, double initial_accumulator_value, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, unsigned int *batch_normalization_layer_offset_indices)
{
	unsigned int i, j, array_index;

	for (i = 0; i < number_of_layers - 1; i++)
	{
		if (neural_network_layers_hyperparameters[i + 1].batch_norm_flag == 1)
		{
			for (j = 0; j < neural_network_layers_hyperparameters[i + 1].number_of_neurons; j++)
			{
				array_index = j + batch_normalization_layer_offset_indices[i];

				/* Moving mean */
				batch_normalization_layer_neurons_parameters[array_index].batch_norm_moving_mean = neural_network_layers_hyperparameters[i + 1].batch_norm_moving_mean_initializer;

				/* Moving variance */
				batch_normalization_layer_neurons_parameters[array_index].batch_norm_moving_variance = neural_network_layers_hyperparameters[i + 1].batch_norm_moving_variance_initializer;

				/* Beta */
				batch_normalization_layer_neurons_parameters[array_index].batch_norm_beta.variable = neural_network_layers_hyperparameters[i + 1].batch_norm_beta_initializer;
				InitializeTrainableParameterGradientsAndAggregates(initial_accumulator_value, &batch_normalization_layer_neurons_parameters[array_index].batch_norm_beta.gradient, &batch_normalization_layer_neurons_parameters[array_index].batch_norm_beta.aggregate0, &batch_normalization_layer_neurons_parameters[array_index].batch_norm_beta.aggregate1, &batch_normalization_layer_neurons_parameters[array_index].batch_norm_beta.aggregate2, &batch_normalization_layer_neurons_parameters[array_index].batch_norm_beta.update);

				/* Gamma */
				batch_normalization_layer_neurons_parameters[array_index].batch_norm_gamma.variable = neural_network_layers_hyperparameters[i + 1].batch_norm_gamma_initializer;
				InitializeTrainableParameterGradientsAndAggregates(initial_accumulator_value, &batch_normalization_layer_neurons_parameters[array_index].batch_norm_gamma.gradient, &batch_normalization_layer_neurons_parameters[array_index].batch_norm_gamma.aggregate0, &batch_normalization_layer_neurons_parameters[array_index].batch_norm_gamma.aggregate1, &batch_normalization_layer_neurons_parameters[array_index].batch_norm_gamma.aggregate2, &batch_normalization_layer_neurons_parameters[array_index].batch_norm_gamma.update);
			} // end of j loop
		}
	} // end of i loop
} // end of InitializeBatchNormalizationLayerParameters function

/* This function initializes trainable parameters gradient and aggregate information */
void InitializeTrainableParameterGradientsAndAggregates(double initial_accumulator_value, double *gradient, double *aggregate0, double *aggregate1, double *aggregate2, double *update)
{
	(*gradient) = 0.0;
	(*aggregate0) = 0.0;
	(*aggregate1) = initial_accumulator_value;
	(*aggregate2) = 0.0;
	(*update) = 0.0;
} // end of InitializeTrainableParameterGradientsAndAggregates function

/******************************************************************************************/
/*************************************** INPUT DATA ***************************************/
/******************************************************************************************/

/* This function allocates memory for features and labels arrays */
void AllocateFeaturesAndLabels(unsigned int rows, unsigned int feature_cols, unsigned int label_cols, double **features, double **labels)
{
	(*features) = malloc(sizeof(double) * rows * feature_cols);
	(*labels) = malloc(sizeof(double) * rows * label_cols);
} // end of AllocateFeaturesAndLabels function

/* This function reads the input data features and labels from the given filepath */
void InputDataFunction(char *filepath, unsigned int rows, unsigned int feature_cols, unsigned int label_cols, double **features_tensor, double **labels_tensor, unsigned int transpose)
{
	printf("InputDataFunction: filepath = %s\n", filepath);
	printf("InputDataFunction: rows = %u, feature_cols = %u, label_cols = %u\n", rows, feature_cols, label_cols);

	FILE *infile = fopen(filepath, "r");
	if (infile == NULL)
	{
		printf("InputDataFunction: Failed to open file %s\n", filepath);
	}
	else
	{
		int systemreturn;
		unsigned int i, j;

		if (transpose == 0)
		{
			for (i = 0; i < rows; i++)
			{
				for (j = 0; j < label_cols; j++)
				{
					systemreturn = fscanf(infile, "%lf\t", &(*labels_tensor)[i * label_cols + j]);
					if (systemreturn == -1)
					{
						printf("InputDataFunction: Failed reading file %s at label indices %u, %u\n", filepath, i, j);
					}
				} // end of j loop

				for (j = 0; j < feature_cols; j++)
				{
					systemreturn = fscanf(infile, "%lf\t", &(*features_tensor)[i * feature_cols + j]);
					if (systemreturn == -1)
					{
						printf("InputDataFunction: Failed reading file %s at feature indices %u, %u\n", filepath, i, j);
					}
				} // end of j loop
			} // end of i loop
		}
		else
		{
			for (i = 0; i < rows; i++)
			{
				for (j = 0; j < label_cols; j++)
				{
					systemreturn = fscanf(infile, "%lf\t", &(*labels_tensor)[j * rows + i]);
					if (systemreturn == -1)
					{
						printf("InputDataFunction: Failed reading file %s at label indices %u, %u\n", filepath, i, j);
					}
				} // end of j loop

				for (j = 0; j < feature_cols; j++)
				{
					systemreturn = fscanf(infile, "%lf\t", &(*features_tensor)[j * rows + i]);
					if (systemreturn == -1)
					{
						printf("InputDataFunction: Failed reading file %s at feature indices %u, %u\n", filepath, i, j);
					}
				} // end of j loop
			} // end of i loop
		}
		fclose(infile);
	}
} // end of InputDataFunction function

/* This function initializes the features and labels arrays for each split (i.e. train, valid, test) */
void InitializeSplitFeaturesAndLabels(unsigned int rows, unsigned int feature_cols, unsigned int label_cols, unsigned int offset, double *features, double *labels, double *features_split, double *labels_split, int transpose)
{
	unsigned int i, j;

	if (transpose == 0)
	{
		for (i = 0; i < rows; i++)
		{
			for (j = 0; j < feature_cols; j++)
			{
				features_split[i * feature_cols + j] = features[i * feature_cols + j + offset];
			} // end of j loop

			for (j = 0; j < label_cols; j++)
			{
				labels_split[i * label_cols + j] = labels[i * label_cols + j + offset];
			} // end of j loop
		} // end of i loop
	}
	else
	{
		for (i = 0; i < rows; i++)
		{
			for (j = 0; j < feature_cols; j++)
			{
				features_split[j * rows + i] = features[j * rows + i + offset];
			} // end of j loop

			for (j = 0; j < label_cols; j++)
			{
				labels_split[j * rows + i] = labels[j * rows + i + offset];
			} // end of j loop
		} // end of i loop
	}
} // end of InitializeSplitFeaturesAndLabels function

/******************************************************************************************/
/********************************* TRAINING AND EVALUATION ********************************/
/******************************************************************************************/

/* This function performs mini batch gradient descent for both training (FeedForward and Backpropagation) and evaluation (FeedForward) */
void MiniBatchGradientDescent(unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int train_rows, double *features_train, double *labels_train, unsigned int valid_rows, double *features_valid, double *labels_valid, struct TrainableParameters *neural_network_kernel_weights, unsigned int *kernel_weights_offset_indices, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int *bias_weights_offset_indices, unsigned int total_bias_weights, struct TrainableParameters *parametric_relu_alphas, unsigned int *parametric_relu_alpha_offset_indices, unsigned int total_parametric_relu_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, unsigned int *batch_normalization_layer_offset_indices, unsigned int total_batch_normalization_neurons, int transpose)
{
	printf("MiniBatchGradientDescent: Starting mini-batch gradient descent!\n");

	unsigned int step, i, j;
	unsigned int current_training_record_index = 0, max_train_feature_elements = train_rows * neural_network_layers_hyperparameters[0].number_of_neurons, max_train_label_elements = train_rows * neural_network_layers_hyperparameters[number_of_layers - 1].number_of_neurons;
	double loss = 0.0;

	/* Declare and initialize neural network for each non-input neuron */
	unsigned int total_neural_network_neurons = 0;
	unsigned int *neural_network_neurons_minus_inputs_offset_indices;
	neural_network_neurons_minus_inputs_offset_indices = malloc(sizeof(unsigned int) * number_of_layers);
	for (i = 0; i < number_of_layers - 1; i++)
	{
		neural_network_neurons_minus_inputs_offset_indices[i] = total_neural_network_neurons;
		total_neural_network_neurons += neural_network_layers_hyperparameters[i + 1].number_of_neurons;
	} // end of i loop

	/* Training batch-neurons */
	struct NeuralNetworkLayerNeuron *neural_network_layer_neurons_train;
	neural_network_layer_neurons_train = malloc(sizeof(struct NeuralNetworkLayerNeuron) * (training_validation_hyperparameters->batch_size * total_neural_network_neurons));

	/* Validation batch-neurons */
	struct NeuralNetworkLayerNeuron *neural_network_layer_neurons_valid;
	neural_network_layer_neurons_valid = malloc(sizeof(struct NeuralNetworkLayerNeuron) * (valid_rows * total_neural_network_neurons));

	/* Batch normalization neurons */
	double *batch_normalization_layer_xmu_train;
	batch_normalization_layer_xmu_train = malloc(sizeof(double) * (training_validation_hyperparameters->batch_size * total_batch_normalization_neurons));

	double batch_normalization_layer_xmu_valid; // don't need to store all values since there is no Backpropagation outside of training

	/* Create dropout mask layers */
	unsigned int total_dropout_neurons = 0;
	unsigned int *dropout_layer_offset_indices;
	dropout_layer_offset_indices = malloc(sizeof(unsigned int) * (number_of_layers - 1));
	for (i = 0; i < number_of_layers - 1; i++)
	{
		if (neural_network_layers_hyperparameters[i + 1].dropout_probability > 0.0)
		{
			dropout_layer_offset_indices[i] = total_dropout_neurons;
			total_dropout_neurons += neural_network_layers_hyperparameters[i + 1].number_of_neurons;
		}
	} // end of i loop

	int *dropout_layer_mask_neurons;
	dropout_layer_mask_neurons = malloc(sizeof(int) * total_dropout_neurons);
	for (i = 0; i < total_batch_normalization_neurons; i++)
	{
		dropout_layer_mask_neurons[i] = 1; // neurons all start on
	} // end of i loop

	/* Create training batch feature and label tensors */
	double *batch_features_tensor;
	batch_features_tensor = malloc(sizeof(double *) * training_validation_hyperparameters->batch_size * neural_network_layers_hyperparameters[0].number_of_neurons);

	double *batch_labels_tensor;
	batch_labels_tensor = malloc(sizeof(double *) * training_validation_hyperparameters->batch_size * neural_network_layers_hyperparameters[number_of_layers - 1].number_of_neurons);

	/* Keep track of loss */
	double *loss_train;
	loss_train = malloc(sizeof(double) * training_validation_hyperparameters->train_steps);

	/* This conditional block is to ensure that during training the first step does not evaluate the validation set,
	 but ensures that there is at least one evaluation by the end without redoing any */
	unsigned int train_count = 0, evaluation_count = 0, number_of_evaluations = 1;
	if (training_validation_hyperparameters->train_steps <= training_validation_hyperparameters->eval_steps)
	{
		number_of_evaluations = 1;
	}
	else
	{
		if (training_validation_hyperparameters->train_steps == 1)
		{
			number_of_evaluations = 1;
		}
		else
		{
			if (training_validation_hyperparameters->eval_steps == 1)
			{
				number_of_evaluations = training_validation_hyperparameters->train_steps / training_validation_hyperparameters->eval_steps - 1;
			}
			else
			{
				number_of_evaluations = training_validation_hyperparameters->train_steps / training_validation_hyperparameters->eval_steps;
			}
		}
	}
	printf("number_of_evaluations = %u\n", number_of_evaluations);

	/* Create evaluation metric structures to store each evaluation step's metrics */
	struct RegressionEvaluationMetric *regression_evaluation_metrics;
	struct ClassificationEvaluationMetric *classification_evaluation_metrics;

	/* Only allocate the memory to the given model type */
	if (training_validation_hyperparameters->loss_function_type == 0) // regression
	{
		regression_evaluation_metrics = malloc(sizeof(struct RegressionEvaluationMetric) * number_of_evaluations);
	}
	else // classification
	{
		classification_evaluation_metrics = malloc(sizeof(struct ClassificationEvaluationMetric) * number_of_evaluations);
	}

	/* Proceed with mini-batch gradient descent training and evaluation loop */
	printf("MiniBatchGradientDescent: training_validation_hyperparameters->train_steps = %u\n\n", training_validation_hyperparameters->train_steps);
	for (step = 0; step < training_validation_hyperparameters->train_steps; step++)
	{
		printf("*****************************************************************************************\n");
		printf("**************************************** STEP %u ****************************************\n", step);
		printf("*****************************************************************************************\n");

		/* Create a mini-batch of features and labels to use for training from the training set */
		current_training_record_index = CreateTrainingDataMiniBatch(transpose, training_validation_hyperparameters->batch_size, neural_network_layers_hyperparameters[0].number_of_neurons, neural_network_layers_hyperparameters[number_of_layers - 1].number_of_neurons, train_rows, max_train_feature_elements, max_train_label_elements, features_train, labels_train, batch_features_tensor, batch_labels_tensor, current_training_record_index);

		/* Calculate training loss from forward pass */
		loss = FeedForward(number_of_layers, neural_network_layers_hyperparameters, training_validation_hyperparameters, training_validation_hyperparameters->batch_size, batch_features_tensor, batch_labels_tensor, neural_network_layer_neurons_train, neural_network_neurons_minus_inputs_offset_indices, neural_network_kernel_weights, kernel_weights_offset_indices, neural_network_bias_weights, bias_weights_offset_indices, parametric_relu_alphas, parametric_relu_alpha_offset_indices, batch_normalization_layer_neurons_parameters, batch_normalization_layer_xmu_train, batch_normalization_layer_offset_indices, dropout_layer_mask_neurons, dropout_layer_offset_indices, 1, transpose);
		if (loss != loss) // if NaN loss
		{
			printf("\nMiniBatchGradientDescent: Error: NaN loss!\n");
			break;
		}
		loss_train[train_count++] = loss;

		/* Now that training forward pass is done, do the backward pass to update the parameters */
		Backpropagation(number_of_layers, neural_network_layers_hyperparameters, training_validation_hyperparameters, training_validation_hyperparameters->batch_size, batch_features_tensor, batch_labels_tensor, neural_network_layer_neurons_train, neural_network_neurons_minus_inputs_offset_indices, neural_network_kernel_weights, kernel_weights_offset_indices, total_kernel_weights, neural_network_bias_weights, bias_weights_offset_indices, total_bias_weights, parametric_relu_alphas, parametric_relu_alpha_offset_indices, total_parametric_relu_neurons, batch_normalization_layer_neurons_parameters, batch_normalization_layer_xmu_train, batch_normalization_layer_offset_indices, total_batch_normalization_neurons, dropout_layer_mask_neurons, dropout_layer_offset_indices, transpose, step);

		/* Print evaluation metrics */
		if (step > 0 && step % training_validation_hyperparameters->eval_steps == 0)
		{
			printf("\nMiniBatchGradientDescent: Evaluating at step %u\n", step);
			ModelEvaluation(number_of_layers, neural_network_layers_hyperparameters, training_validation_hyperparameters, valid_rows, features_valid, labels_valid, neural_network_layer_neurons_valid, neural_network_neurons_minus_inputs_offset_indices, neural_network_kernel_weights, kernel_weights_offset_indices, neural_network_bias_weights, bias_weights_offset_indices, parametric_relu_alphas, parametric_relu_alpha_offset_indices, batch_normalization_layer_neurons_parameters, &batch_normalization_layer_xmu_valid, batch_normalization_layer_offset_indices, dropout_layer_mask_neurons, dropout_layer_offset_indices, 0, transpose, regression_evaluation_metrics, classification_evaluation_metrics, &evaluation_count, step);
		}
	} // end of step loop

	/* Print final evaluation metrics */
	/* This conditional block is to ensure that during training the first step does not evaluate the validation set,
	 but ensures that there is at least one evaluation by the end without redoing any */
	if (training_validation_hyperparameters->train_steps <= training_validation_hyperparameters->eval_steps)
	{
		printf("\n\nMiniBatchGradientDescent: Final evaluation\n");
		loss = FeedForward(number_of_layers, neural_network_layers_hyperparameters, training_validation_hyperparameters, valid_rows, features_valid, labels_valid, neural_network_layer_neurons_valid, neural_network_neurons_minus_inputs_offset_indices, neural_network_kernel_weights, kernel_weights_offset_indices, neural_network_bias_weights, bias_weights_offset_indices, parametric_relu_alphas, parametric_relu_alpha_offset_indices, batch_normalization_layer_neurons_parameters, &batch_normalization_layer_xmu_valid, batch_normalization_layer_offset_indices, dropout_layer_mask_neurons, dropout_layer_offset_indices, 0, transpose);
		ModelEvaluation(number_of_layers, neural_network_layers_hyperparameters, training_validation_hyperparameters, valid_rows, features_valid, labels_valid, neural_network_layer_neurons_valid, neural_network_neurons_minus_inputs_offset_indices, neural_network_kernel_weights, kernel_weights_offset_indices, neural_network_bias_weights, bias_weights_offset_indices, parametric_relu_alphas, parametric_relu_alpha_offset_indices, batch_normalization_layer_neurons_parameters, &batch_normalization_layer_xmu_valid, batch_normalization_layer_offset_indices, dropout_layer_mask_neurons, dropout_layer_offset_indices, 0, transpose, regression_evaluation_metrics, classification_evaluation_metrics, &evaluation_count, training_validation_hyperparameters->train_steps - 1);
	}
	else
	{
		if (training_validation_hyperparameters->train_steps == 1)
		{
			printf("\n\nMiniBatchGradientDescent: Final evaluation\n");
			ModelEvaluation(number_of_layers, neural_network_layers_hyperparameters, training_validation_hyperparameters, valid_rows, features_valid, labels_valid, neural_network_layer_neurons_valid, neural_network_neurons_minus_inputs_offset_indices, neural_network_kernel_weights, kernel_weights_offset_indices, neural_network_bias_weights, bias_weights_offset_indices, parametric_relu_alphas, parametric_relu_alpha_offset_indices, batch_normalization_layer_neurons_parameters, &batch_normalization_layer_xmu_valid, batch_normalization_layer_offset_indices, dropout_layer_mask_neurons, dropout_layer_offset_indices, 0, transpose, regression_evaluation_metrics, classification_evaluation_metrics, &evaluation_count, training_validation_hyperparameters->train_steps - 1);
		}
		else
		{
			if (training_validation_hyperparameters->eval_steps != 1)
			{
				if (training_validation_hyperparameters->train_steps % training_validation_hyperparameters->eval_steps == 0)
				{
					printf("\n\nMiniBatchGradientDescent: Final evaluation\n");
					ModelEvaluation(number_of_layers, neural_network_layers_hyperparameters, training_validation_hyperparameters, valid_rows, features_valid, labels_valid, neural_network_layer_neurons_valid, neural_network_neurons_minus_inputs_offset_indices, neural_network_kernel_weights, kernel_weights_offset_indices, neural_network_bias_weights, bias_weights_offset_indices, parametric_relu_alphas, parametric_relu_alpha_offset_indices, batch_normalization_layer_neurons_parameters, &batch_normalization_layer_xmu_valid, batch_normalization_layer_offset_indices, dropout_layer_mask_neurons, dropout_layer_offset_indices, 0, transpose, regression_evaluation_metrics, classification_evaluation_metrics, &evaluation_count, training_validation_hyperparameters->train_steps - 1);
				}
			}
		}
	}

	/* Print training loss */
	printf("\nloss_train:\n");
	for (step = 0; step < train_count; step++)
	{
		printf("%u\t%.16f\n", step, loss_train[step]);
	} // end of step loop

	/* Print evaluation metrics */
	if (training_validation_hyperparameters->loss_function_type == 0) // regression
	{
		printf("train_step\tloss\tmse\trmse\n");
		for (step = 0; step < evaluation_count; step++)
		{
			printf("%u\t%.16f\t%.16f\t%.16f\n", regression_evaluation_metrics[step].train_step, regression_evaluation_metrics[step].loss, regression_evaluation_metrics[step].mse, regression_evaluation_metrics[step].rmse);
		} // end of step loop
		printf("\n");
	}
	else // classification
	{
		printf("train_step\tloss\texact_match_ratio\taccuracy\tavg_precision\tavg_recall\tf1_score\n");
		for (step = 0; step < evaluation_count; step++)
		{
			printf("%u\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\t%.16f\n", classification_evaluation_metrics[step].train_step, classification_evaluation_metrics[step].loss, classification_evaluation_metrics[step].exact_match_ratio, classification_evaluation_metrics[step].accuracy, classification_evaluation_metrics[step].avg_precision, classification_evaluation_metrics[step].avg_recall, classification_evaluation_metrics[step].f1_score);
		} // end of step loop
		printf("\n");
	}

	/* Free dynamic array memory */
	if (training_validation_hyperparameters->loss_function_type == 0) // regression
	{
		free(regression_evaluation_metrics);
	}
	else // classification
	{
		free(classification_evaluation_metrics);
	}

	free(loss_train);

	free(batch_labels_tensor);
	free(batch_features_tensor);

	free(dropout_layer_mask_neurons);
	free(dropout_layer_offset_indices);

	free(batch_normalization_layer_xmu_train);

	free(neural_network_layer_neurons_valid);
	free(neural_network_layer_neurons_train);

	free(neural_network_neurons_minus_inputs_offset_indices);
} // end of MiniBatchGradientDescent function

/* This function creates a mini-batch from the training data */
unsigned int CreateTrainingDataMiniBatch(int transpose, unsigned int batch_size, unsigned int number_of_input_neurons, unsigned int number_of_output_neurons, unsigned int train_rows, unsigned int max_train_feature_elements, unsigned int max_train_label_elements, double *features_train, double *labels_train, double *batch_features_tensor, double *batch_labels_tensor, unsigned int current_training_record_index)
{
	unsigned int i, j;

	if (transpose == 0)
	{
		for (i = 0; i < batch_size; i++)
		{
			for (j = 0; j < number_of_input_neurons; j++)
			{
				batch_features_tensor[i * number_of_input_neurons + j] = features_train[((i + current_training_record_index) * number_of_input_neurons + j) % max_train_feature_elements];
			} // end of j loop

			for (j = 0; j < number_of_output_neurons; j++)
			{
				batch_labels_tensor[i * number_of_output_neurons + j] = labels_train[((i + current_training_record_index) * number_of_output_neurons + j) % max_train_label_elements];
			} // end of j loop
		} // end of i loop
	}
	else
	{
		for (i = 0; i < batch_size; i++)
		{
			for (j = 0; j < number_of_input_neurons; j++)
			{
				batch_features_tensor[j * batch_size + i] = features_train[((j + current_training_record_index) * batch_size + i) % max_train_feature_elements];
			} // end of j loop

			for (j = 0; j < number_of_output_neurons; j++)
			{
				batch_labels_tensor[j * batch_size + i] = labels_train[((j + current_training_record_index) * batch_size + i) % max_train_label_elements];
			} // end of j loop
		} // end of i loop
	}

	/* Update current training record index to keep track of where we are in the data tensor for each batch */
	current_training_record_index = (current_training_record_index + batch_size) % train_rows;

	return current_training_record_index;
} // end of CreateTrainingDataBatch function

/******************************************************************************************/
/************************************** FEEDFORWARD ***************************************/
/******************************************************************************************/

/* This function feeds the input data forward through the neural network layers */
double FeedForward(unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int batch_size, double *batch_features_tensor, double *batch_labels_tensor, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int *neural_network_neurons_minus_inputs_offset_indices, struct TrainableParameters *neural_network_kernel_weights, unsigned int *kernel_weights_offset_indices, struct TrainableParameters *neural_network_bias_weights, unsigned int *bias_weights_offset_indices, struct TrainableParameters *parametric_relu_alphas, unsigned int *parametric_relu_alpha_offset_indices, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, double *batch_normalization_layer_xmu, unsigned int *batch_normalization_layer_offset_indices, int *dropout_layer_mask_neurons, unsigned int *dropout_layer_offset_indices, int training, int transpose)
{
	if (training == 1)
	{
		printf("FeedForward: Feeding forward mini-batch for training\n");
	}
	else
	{
		printf("FeedForward: Feeding forward mini-batch for validation\n");
	}

	unsigned int i, j, k;
	double loss = 0.0;

	/* First layer using the inputs */
	MatrixMultiplication(batch_size, neural_network_layers_hyperparameters[1].number_of_neurons, neural_network_layers_hyperparameters[0].number_of_neurons, batch_features_tensor, &neural_network_kernel_weights[0].variable, &neural_network_layer_neurons[0].weighted_sum, 0, 0, 0, 1, sizeof(struct TrainableParameters) / sizeof(void*), sizeof(struct NeuralNetworkLayerNeuron) / sizeof(void*), transpose, 0);

	if (neural_network_layers_hyperparameters[1].bias_weight_initialization_type >= 0)
	{
		AddBiases(batch_size, neural_network_layers_hyperparameters[1].number_of_neurons, neural_network_bias_weights, 0, neural_network_layer_neurons, 0);
	}

	if (neural_network_layers_hyperparameters[1].batch_norm_flag == 1)
	{
		if (neural_network_layers_hyperparameters[1].batch_norm_after_activation_flag == 0)
		{
			ApplyBatchNormalization(batch_size, neural_network_layers_hyperparameters[1].number_of_neurons, neural_network_layer_neurons, 0, batch_normalization_layer_neurons_parameters, batch_normalization_layer_xmu, 0, neural_network_layers_hyperparameters[1].batch_norm_momentum, training);

			ApplyActivationFunction(neural_network_layers_hyperparameters[1].activation_type, batch_size, neural_network_layers_hyperparameters[1].number_of_neurons, neural_network_layer_neurons, 0, neural_network_layers_hyperparameters[1].activation_function_alpha_initializer, parametric_relu_alphas, 0);
		}
		else
		{
			ApplyActivationFunction(neural_network_layers_hyperparameters[1].activation_type, batch_size, neural_network_layers_hyperparameters[1].number_of_neurons, neural_network_layer_neurons, 0, neural_network_layers_hyperparameters[1].activation_function_alpha_initializer, parametric_relu_alphas, 0);

			ApplyBatchNormalization(batch_size, neural_network_layers_hyperparameters[1].number_of_neurons, neural_network_layer_neurons, 0, batch_normalization_layer_neurons_parameters, batch_normalization_layer_xmu, 0, neural_network_layers_hyperparameters[1].batch_norm_momentum, training);
		}
	}
	else
	{
		ApplyActivationFunction(neural_network_layers_hyperparameters[1].activation_type, batch_size, neural_network_layers_hyperparameters[1].number_of_neurons, neural_network_layer_neurons, 0, neural_network_layers_hyperparameters[1].activation_function_alpha_initializer, parametric_relu_alphas, 0);
	}

	if (training == 1)
	{
		if (neural_network_layers_hyperparameters[1].dropout_probability > 0.0)
		{
			ApplyDropout(batch_size, neural_network_layers_hyperparameters[1].number_of_neurons, neural_network_layer_neurons, 0, dropout_layer_mask_neurons, 0, neural_network_layers_hyperparameters[1].dropout_probability);
		}
	}

	/* Now do the rest of the layers */
	for (i = 1; i < number_of_layers - 1; i++)
	{
		MatrixMultiplication(batch_size, neural_network_layers_hyperparameters[i + 1].number_of_neurons, neural_network_layers_hyperparameters[i].number_of_neurons, &neural_network_layer_neurons[0].activation, &neural_network_kernel_weights[0].variable, &neural_network_layer_neurons[0].weighted_sum, batch_size * neural_network_neurons_minus_inputs_offset_indices[i - 1], kernel_weights_offset_indices[i], batch_size * neural_network_neurons_minus_inputs_offset_indices[i], sizeof(struct NeuralNetworkLayerNeuron) / sizeof(void*), sizeof(struct TrainableParameters) / sizeof(void*), sizeof(struct NeuralNetworkLayerNeuron) / sizeof(void*), 0, 0);
		if (neural_network_layers_hyperparameters[i + 1].bias_weight_initialization_type >= 0)
		{
			AddBiases(batch_size, neural_network_layers_hyperparameters[i + 1].number_of_neurons, neural_network_bias_weights, bias_weights_offset_indices[i], neural_network_layer_neurons, batch_size * neural_network_neurons_minus_inputs_offset_indices[i]);
		}

		if (neural_network_layers_hyperparameters[i + 1].batch_norm_flag == 1)
		{
			if (neural_network_layers_hyperparameters[i + 1].batch_norm_after_activation_flag == 0)
			{
				ApplyBatchNormalization(batch_size, neural_network_layers_hyperparameters[i + 1].number_of_neurons, neural_network_layer_neurons, batch_size * neural_network_neurons_minus_inputs_offset_indices[i], batch_normalization_layer_neurons_parameters, batch_normalization_layer_xmu, batch_normalization_layer_offset_indices[i], neural_network_layers_hyperparameters[i + 1].batch_norm_momentum, training);

				ApplyActivationFunction(neural_network_layers_hyperparameters[i + 1].activation_type, batch_size, neural_network_layers_hyperparameters[i + 1].number_of_neurons, neural_network_layer_neurons, batch_size * neural_network_neurons_minus_inputs_offset_indices[i], neural_network_layers_hyperparameters[i + 1].activation_function_alpha_initializer, parametric_relu_alphas, parametric_relu_alpha_offset_indices[i]);
			}
			else
			{
				ApplyActivationFunction(neural_network_layers_hyperparameters[i + 1].activation_type, batch_size, neural_network_layers_hyperparameters[i + 1].number_of_neurons, neural_network_layer_neurons, batch_size * neural_network_neurons_minus_inputs_offset_indices[i], neural_network_layers_hyperparameters[i + 1].activation_function_alpha_initializer, parametric_relu_alphas, parametric_relu_alpha_offset_indices[i]);

				ApplyBatchNormalization(batch_size, neural_network_layers_hyperparameters[i + 1].number_of_neurons, neural_network_layer_neurons, batch_size * neural_network_neurons_minus_inputs_offset_indices[i], batch_normalization_layer_neurons_parameters, batch_normalization_layer_xmu, batch_normalization_layer_offset_indices[i], neural_network_layers_hyperparameters[i + 1].batch_norm_momentum, training);
			}
		}
		else
		{
			ApplyActivationFunction(neural_network_layers_hyperparameters[i + 1].activation_type, batch_size, neural_network_layers_hyperparameters[i + 1].number_of_neurons, neural_network_layer_neurons, batch_size * neural_network_neurons_minus_inputs_offset_indices[i], neural_network_layers_hyperparameters[i + 1].activation_function_alpha_initializer, parametric_relu_alphas, parametric_relu_alpha_offset_indices[i]);
		}

		if (training == 1)
		{
			if (neural_network_layers_hyperparameters[i + 1].dropout_probability > 0.0)
			{
				ApplyDropout(batch_size, neural_network_layers_hyperparameters[i + 1].number_of_neurons, neural_network_layer_neurons, batch_size * neural_network_neurons_minus_inputs_offset_indices[i], dropout_layer_mask_neurons, dropout_layer_offset_indices[i], neural_network_layers_hyperparameters[i + 1].dropout_probability);
			}
		}
	} // end of i loop

	if (debug_print == 1 && training == 1)
	{
		DebugPrintFeedForward(number_of_layers, neural_network_layers_hyperparameters, batch_size, batch_features_tensor, neural_network_layer_neurons, neural_network_neurons_minus_inputs_offset_indices, neural_network_kernel_weights, kernel_weights_offset_indices, neural_network_bias_weights, bias_weights_offset_indices, parametric_relu_alphas, parametric_relu_alpha_offset_indices, batch_normalization_layer_neurons_parameters, batch_normalization_layer_xmu, batch_normalization_layer_offset_indices);
	}

	loss = CalculateLoss(number_of_layers,neural_network_layers_hyperparameters, training_validation_hyperparameters->loss_function_type, batch_size, neural_network_layers_hyperparameters[number_of_layers - 1].number_of_neurons, &neural_network_layer_neurons[0].activation, training_validation_hyperparameters->batch_size * neural_network_neurons_minus_inputs_offset_indices[number_of_layers - 2], sizeof(struct NeuralNetworkLayerNeuron) / sizeof(void*), batch_labels_tensor, transpose, neural_network_kernel_weights, kernel_weights_offset_indices, neural_network_bias_weights, bias_weights_offset_indices);

	if (training == 1)
	{
		printf("FeedForward: training loss = %.16f\n", loss);
	}
	else
	{
		printf("FeedForward: validation loss = %.16f\n", loss);
	}
	printf("\n");

	return loss;
} // end of FeedForward function

/* This function performs matrix multiplication between two given matrices */
void MatrixMultiplication(unsigned int m, unsigned int n, unsigned int p, double *A, double *B, double *C, unsigned int A_offset, unsigned int B_offset, unsigned int C_offset, unsigned long A_size, unsigned long B_size, unsigned long C_size, int transpose_A, int transpose_B)
{
	/* C = [m, n] */
	/* A = [m, p] */
	/* B = [p, n] */

	unsigned int i, j, k, A_index, B_index, C_index;

	printf("MatrixMultiplication: m = %u, n = %u, p = %u, A_offset = %u, B_offset = %u, C_offset = %u, A_size = %zu, B_size = %zu, C_size = %zu\n", m, n, p, A_offset, B_offset, C_offset, A_size, B_size, C_size);

	if (transpose_B == 0) // if B is NOT transposed
	{
		if (transpose_A == 0) // if A is NOT transposed
		{
			for (i = 0; i < m; i++)
			{
				for (j = 0; j < n; j++)
				{
					C_index = (i * n + j + C_offset) * C_size;

					C[C_index] = 0.0;
					for (k = 0; k < p; k++)
					{
						A_index = (i * p + k + A_offset) * A_size;
						B_index = (k * n + j + B_offset) * B_size;

						C[C_index] += A[A_index] * B[B_index];
					} // end of k loop
				} // end of j loop
			} // end of i loop
		} // end of if a is NOT transposed
		else // if a is transposed
		{
			for (i = 0; i < m; i++)
			{
				for (j = 0; j < n; j++)
				{
					C_index = (i * n + j + C_offset) * C_size;

					C[C_index] = 0.0;
					for (k = 0; k < p; k++)
					{
						A_index = (k * m + i + A_offset) * A_size;
						B_index = (k * n + j + B_offset) * B_size;

						C[C_index] += A[A_index] * B[B_index];
					} // end of k loop
				} // end of j loop
			} // end of i loop
		} // end of if a is  transposed
	} // end of if B is NOT transposed
	else // if B is transposed
	{
		if (transpose_A == 0)
		{
			for (i = 0; i < m; i++)
			{
				for (j = 0; j < n; j++)
				{
					C_index = (i * n + j + C_offset) * C_size;

					C[C_index] = 0.0;
					for (k = 0; k < p; k++)
					{
						A_index = (i * p + k + A_offset) * A_size;
						B_index = (j * p + k + B_offset) * B_size;

						C[C_index] += A[A_index] * B[B_index];
					} // end of k loop
				} // end of j loop
			} // end of i loop
		} // end of if a is NOT transposed
		else
		{
			for (i = 0; i < m; i++)
			{
				for (j = 0; j < n; j++)
				{
					C_index = (i * n + j + C_offset) * C_size;

					C[C_index] = 0.0;
					for (k = 0; k < p; k++)
					{
						A_index = (k * m + i + A_offset) * A_size;
						B_index = (j * p + k + B_offset) * B_size;

						C[C_index] += A[A_index] * B[B_index];
					} // end of k loop
				} // end of j loop
			} // end of i loop
		} // end of if a is transposed
	} // end of if b is transposed
} // end of MatrixMultiplication function

/* This function adds biases (if present) to the result of matmul(X, W) */
void AddBiases(unsigned int batch_size, unsigned int number_of_neurons, struct TrainableParameters *neural_network_bias_weights, unsigned int bias_weights_offset, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int neuron_offset)
{
	unsigned int i, j, neuron_index, bias_index;

	for (i = 0; i < batch_size; i++)
	{
		for (j = 0; j < number_of_neurons; j++)
		{
			neuron_index = i * number_of_neurons + j + neuron_offset;
			bias_index = j + bias_weights_offset;

			neural_network_layer_neurons[neuron_index].weighted_sum += neural_network_bias_weights[bias_index].variable;
		} // end of j loop
	} // end of i loop
} // end of AddBiases function

/******************************************************************************************/
/********************************* ACTIVATION FUNCTIONS ***********************************/
/******************************************************************************************/

/* This function applies given activation function to given layers weighted sums and returns result as layer's activations */
void ApplyActivationFunction(int activation_type, unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int neuron_offset, double non_parametric_alpha, struct TrainableParameters *parametric_relu_alphas, unsigned int parametric_relu_offset)
{
	/* Layer activation types */
	/*
	name		activation_type		f(x)								f'(x)
	linear		0					x									1
	sigmoid		1					1 / (1 + exp(-x))					f(x) * (1 - f(x))
	tanh		2					2 / (1 + exp(-2 * x)) - 1			1 - f(x)^2
	relu		3					max(0, x)							x < 0 ? 0 : 1
	leaky_relu	4					max(alpha * x, x)					x < 0 ? alpha : 1
	prelu		5					max(alpha * x, x)					x < 0 ? alpha : 1
	elu			6					x < 0 ? alpha * (exp(x) - 1) : x	x < 0 ? f(alpha, x) + alpha : 1
	softmax		7					ei^(x) / sum(ei^(x), i, 1, n)		fi(x) * (kronecker_deltaij - fj(x))
	 */

	printf("ApplyActivationFunction: activation_type = %d\n", activation_type);

	switch (activation_type)
	{
		case 0:
			ApplyLinearActivationFunction(batch_size, number_of_neurons, neural_network_layer_neurons, neuron_offset);
			break;
		case 1:
			ApplySigmoidActivationFunction(batch_size, number_of_neurons, neural_network_layer_neurons, neuron_offset);
			break;
		case 2:
			ApplyTanhActivationFunction(batch_size, number_of_neurons, neural_network_layer_neurons, neuron_offset);
			break;
		case 3:
			ApplyReLUActivationFunction(batch_size, number_of_neurons, neural_network_layer_neurons, neuron_offset);
			break;
		case 4:
			ApplyLeakyReLUActivationFunction(batch_size, number_of_neurons, neural_network_layer_neurons, neuron_offset, non_parametric_alpha);
			break;
		case 5:
			ApplyParametricReLUActivationFunction(batch_size, number_of_neurons, neural_network_layer_neurons, neuron_offset, parametric_relu_alphas, parametric_relu_offset);
			break;
		case 6:
			ApplyEluActivationFunction(batch_size, number_of_neurons, neural_network_layer_neurons, neuron_offset, non_parametric_alpha);
			break;
		case 7:
			ApplySoftmaxActivationFunction(batch_size, number_of_neurons, neural_network_layer_neurons, neuron_offset);
			break;
		default:
			ApplyLinearActivationFunction(batch_size, number_of_neurons, neural_network_layer_neurons, neuron_offset);
			printf("ApplyActivationFunction: No activation function to apply\n");
	}
} // end of ApplyActivationFunction function

/* This function applies linear activation function to given layer's neurons */
void ApplyLinearActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int offset)
{
	unsigned int i, j, neuron_index;

	for (i = 0; i < batch_size; i++)
	{
		for (j = 0; j < number_of_neurons; j++)
		{
			neuron_index = i * number_of_neurons + j + offset;

			neural_network_layer_neurons[neuron_index].activation = ActivationFunctionLinear(neural_network_layer_neurons[neuron_index].weighted_sum);
		} // end of j loop
	} // end of i loop
} // end of ApplyLinearActivationFunction function

/* This function returns the linear activation */
double ActivationFunctionLinear(double x)
{
	/* f(x) = x */

	return x;
} // end of ActivationFunctionLinear function

/* This function applies linear activation function to given layer's neurons */
void ApplySigmoidActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int offset)
{
	unsigned int i, j, neuron_index;

	for (i = 0; i < batch_size; i++)
	{
		for (j = 0; j < number_of_neurons; j++)
		{
			neuron_index = i * number_of_neurons + j + offset;

			neural_network_layer_neurons[neuron_index].activation = ActivationFunctionSigmoid(neural_network_layer_neurons[neuron_index].weighted_sum);
		} // end of j loop
	} // end of i loop
} // end of ApplySigmoidActivationFunction function

/* This function returns the sigmoid activation */
double ActivationFunctionSigmoid(double x)
{
	/* f(x) = 1 / (1 + e^(-x)) */

	return 1.0 / (1.0 + exp(-x));
} // end of ActivationFunctionSigmoid function

/* This function applies tanh activation function to given layer's neurons */
void ApplyTanhActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int offset)
{
	unsigned int i, j, neuron_index;

	for (i = 0; i < batch_size; i++)
	{
		for (j = 0; j < number_of_neurons; j++)
		{
			neuron_index = i * number_of_neurons + j + offset;

			neural_network_layer_neurons[neuron_index].activation = ActivationFunctionTanh(neural_network_layer_neurons[neuron_index].weighted_sum);
		} // end of j loop
	} // end of i loop
} // end of ApplyTanhActivationFunction function

/* This function returns the tanh activation */
double ActivationFunctionTanh(double x)
{
	/* f(x) = 2 * sigmoid(2x) - 1 */

	return 2.0 * ActivationFunctionSigmoid(2.0 * x) - 1.0;
} // end of ActivationFunctionTanh function

/* This function applies relu activation function to given layer's neurons */
void ApplyReLUActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int offset)
{
	unsigned int i, j, neuron_index;

	for (i = 0; i < batch_size; i++)
	{
		for (j = 0; j < number_of_neurons; j++)
		{
			neuron_index = i * number_of_neurons + j + offset;

			neural_network_layer_neurons[neuron_index].activation = ActivationFunctionReLU(neural_network_layer_neurons[neuron_index].weighted_sum);
		} // end of j loop
	} // end of i loop
} // end of ApplyReLUActivationFunction function

/* This function returns the ReLU activation */
double ActivationFunctionReLU(double x)
{
	/* 	if x < 0
			f(x) = 0
		else
			f(x) = x
	*/

	return x < 0.0 ? 0.0 : x;
} // end of ActivationFunctionReLU function

/* This function applies leaky ReLU activation function to given layer's neurons */
void ApplyLeakyReLUActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int offset, double alpha)
{
	unsigned int i, j, neuron_index;

	for (i = 0; i < batch_size; i++)
	{
		for (j = 0; j < number_of_neurons; j++)
		{
			neuron_index = i * number_of_neurons + j + offset;

			neural_network_layer_neurons[neuron_index].activation = ActivationFunctionParametricReLU(neural_network_layer_neurons[neuron_index].weighted_sum, alpha);
		} // end of j loop
	} // end of i loop
} // end of ApplyLeakyReLUActivationFunction function

/* This function applies parametric ReLU activation function to given layer's neurons */
void ApplyParametricReLUActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int offset, struct TrainableParameters *parametric_relu_alphas, unsigned int parametric_relu_offset)
{
	unsigned int i, j, neuron_index, parametric_relu_index;

	for (i = 0; i < batch_size; i++)
	{
		for (j = 0; j < number_of_neurons; j++)
		{
			neuron_index = i * number_of_neurons + j + offset;

			parametric_relu_index = j + parametric_relu_offset;

			neural_network_layer_neurons[neuron_index].activation = ActivationFunctionParametricReLU(neural_network_layer_neurons[neuron_index].weighted_sum, parametric_relu_alphas[parametric_relu_index].variable);
		} // end of j loop
	} // end of i loop
} // end of ApplyParametricReluActivationFunction function

/* This function returns the parametric ReLU activation with alpha */
double ActivationFunctionParametricReLU(double x, double alpha)
{
	/* 	if x < 0
			f(x) = alpha * x
		else
			f(x) = x
	*/

	return x < 0.0 ? alpha * x : x;
} // end of ActivationFunctionParametricRelu function

/* This function applies elu activation function to given layer's neurons */
void ApplyEluActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int offset, double alpha)
{
	unsigned int i, j, neuron_index;

	for (i = 0; i < batch_size; i++)
	{
		for (j = 0; j < number_of_neurons; j++)
		{
			neuron_index = i * number_of_neurons + j + offset;

			neural_network_layer_neurons[neuron_index].activation = ActivationFunctionElu(neural_network_layer_neurons[neuron_index].weighted_sum, alpha);
		} // end of j loop
	} // end of i loop
} // end of ApplyEluActivationFunction function

/* This function returns the elu activation */
double ActivationFunctionElu(double x, double alpha)
{
	/* 	if x < 0
			f(x) = alpha * (e^(x) - 1)
		else
			f(x) = x
	*/

	return x < 0.0 ? alpha * (exp(x) - 1.0) : x;
} // end of ActivationFunctionElu function

/* This function applies softmax activation function to given layer's neurons */
void ApplySoftmaxActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int offset)
{
	/* f(xi) = e^(xi - max(x)) / sum(e^(xj - max(x)), j, 0, n - 1) */

	unsigned int i, j, neuron_index;
	double max_logit, denominator_sum;

	for (i = 0; i < batch_size; i++)
	{
		max_logit = -DBL_MAX;
		for (j = 0; j < number_of_neurons; j++)
		{
			neuron_index = i * number_of_neurons + j + offset;

			if (neural_network_layer_neurons[neuron_index].weighted_sum > max_logit)
			{
				max_logit = neural_network_layer_neurons[neuron_index].weighted_sum;
			}
		} // end of j loop

		denominator_sum = 0.0;
		for (j = 0; j < number_of_neurons; j++)
		{
			neuron_index = i * number_of_neurons + j + offset;

			/* Shift logits by the max logit to make numerically stable */
			neural_network_layer_neurons[neuron_index].activation = exp(neural_network_layer_neurons[neuron_index].weighted_sum - max_logit);
			denominator_sum += neural_network_layer_neurons[neuron_index].activation;
		} // end of j loop

		for (j = 0; j < number_of_neurons; j++)
		{
			neuron_index = i * number_of_neurons + j + offset;

			neural_network_layer_neurons[neuron_index].activation /= denominator_sum;
		} // end of j loop
	} // end of i loop
} // end of ApplySoftmaxActivationFunction function

/******************************************************************************************/
/*********************************** ADVANCED METHODS *************************************/
/******************************************************************************************/

/* This function applies batch normalization to the given layer (note different behavior for training and inference) */
void ApplyBatchNormalization(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int activations_offset, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, double *batch_normalization_layer_xmu, unsigned int batch_normalization_layer_offset, double batch_norm_momentum, int training)
{
	unsigned int i, j, batch_norm_index, neuron_index, xmu_index;

	if (training == 1)
	{
		for (j = 0; j < number_of_neurons; j++)
		{
			batch_norm_index = j + batch_normalization_layer_offset;

			/* Calculate the mean of the batch */
			batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_mean = 0.0;
			for (i = 0; i < batch_size; i++)
			{
				neuron_index = i * number_of_neurons + j + activations_offset;

				batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_mean += neural_network_layer_neurons[neuron_index].activation;
			} // end of i loop
			batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_mean /= batch_size;

			/* Update moving mean for use in inference */
			batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_moving_mean = (batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_moving_mean - batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_mean) * batch_norm_momentum + batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_mean;

			/* Calculate the variance of the batch */
			batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_variance = 0.0;
			for (i = 0; i < batch_size; i++)
			{
				neuron_index = i * number_of_neurons + j + activations_offset;
				xmu_index = i * number_of_neurons + batch_size * batch_normalization_layer_offset + j;

				/* Calculate the x - mu using the mean of the batch */
				batch_normalization_layer_xmu[xmu_index] = (neural_network_layer_neurons[neuron_index].activation - batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_mean);

				/* Calculate the variance of the batch */
				batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_variance += batch_normalization_layer_xmu[xmu_index] * batch_normalization_layer_xmu[xmu_index];
			} // end of i loop
			batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_variance /= batch_size;

			/* Update moving variance for use in inference */
			batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_moving_variance = (batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_moving_variance - batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_variance) * batch_norm_momentum + batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_variance;

			for (i = 0; i < batch_size; i++)
			{
				neuron_index = i * number_of_neurons + j + activations_offset;
				xmu_index = i * number_of_neurons + batch_size * batch_normalization_layer_offset + j;

				/* Standardize using batch mean and batch variance */
				neural_network_layer_neurons[neuron_index].activation = batch_normalization_layer_xmu[xmu_index] / sqrt(batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_variance + epsilon);

				/* Denormalize using beta and gamma */
				neural_network_layer_neurons[neuron_index].activation = neural_network_layer_neurons[neuron_index].activation * batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_gamma.variable + batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_beta.variable;
			} // end of i loop
		} // end of j loop
	}
	else
	{
		for (j = 0; j < number_of_neurons; j++)
		{
			batch_norm_index = j + batch_normalization_layer_offset;

			for (i = 0; i < batch_size; i++)
			{
				neuron_index = i * number_of_neurons + j + activations_offset;

				/* Calculate the x - mu using the moving mean */
				(*batch_normalization_layer_xmu) = (neural_network_layer_neurons[neuron_index].activation - batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_moving_mean);

				/* Standardize using moving mean and moving variance */
				neural_network_layer_neurons[neuron_index].activation = (*batch_normalization_layer_xmu) / sqrt(batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_moving_variance + epsilon);

				/* Denormalize using beta and gamma */
				neural_network_layer_neurons[neuron_index].activation = neural_network_layer_neurons[neuron_index].activation * batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_gamma.variable + batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_beta.variable;
			} // end of i loop
		} // end of j loop
	}
} // end of ApplyBatchNormalization function

/* This function applies dropout to the given layer (only during training) */
void ApplyDropout(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int activations_offset, int *dropout_layer_mask_neurons, unsigned int dropout_offset, double dropout_probability)
{
	unsigned int i, j, dropout_index, neuron_index;
	double keep_probability = 1.0 - dropout_probability;

	for (j = 0; j < number_of_neurons; j++)
	{
		dropout_index = j + dropout_offset;

		if (UnifRand() <= dropout_probability)
		{
			dropout_layer_mask_neurons[dropout_index] = 0; // turn neuron off

			for (i = 0; i < batch_size; i++)
			{
				neuron_index = i * number_of_neurons + j + activations_offset;

				neural_network_layer_neurons[neuron_index].activation = 0.0;
			} // end of i loop
		}
		else
		{
			dropout_layer_mask_neurons[dropout_index] = 1; // turn neuron on

			for (i = 0; i < batch_size; i++)
			{
				neuron_index = i * number_of_neurons + j + activations_offset;

				neural_network_layer_neurons[neuron_index].activation /= keep_probability;
			} // end of i loop
		}
	} // end of j loop
} // end of ApplyDropout function

/******************************************************************************************/
/***************************************** LOSS *******************************************/
/******************************************************************************************/

/* This function calculates the given loss function at the end of the forward pass of the neural network layers */
double CalculateLoss(unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, int loss_function_type, unsigned int batch_size, unsigned int number_of_outputs, double *logits, unsigned int logits_offset, unsigned long logits_size, double *labels, int transpose, struct TrainableParameters *neural_network_kernel_weights, unsigned int *kernel_weights_offset_indices, struct TrainableParameters *neural_network_bias_weights, unsigned int *bias_weights_offset_indices)
{
	unsigned int i;
	double loss = 0.0;

	/* Calculate base loss */
	switch(loss_function_type)
	{
		case 0:
			loss = MSELossFunction(batch_size, number_of_outputs, logits, logits_offset, logits_size, labels, transpose);
			break;
		case 1:
			loss = SoftmaxCrossEntropyWithLogitsLossFunction(batch_size, number_of_outputs, logits, logits_offset, logits_size, labels, transpose);
			break;
		case 2:
			loss = SigmoidCrossEntropyWithLogitsLossFunction(batch_size, number_of_outputs, logits, logits_offset, logits_size, labels, transpose);
			break;
		default:
			printf("CalculateLoss: No loss function selected!\n");
	}

	/* Add regularization to loss */
	double regularization_loss = 0.0;
	for (i = 0; i < number_of_layers - 1; i++)
	{
		if (neural_network_layers_hyperparameters[i + 1].kernel_l1_regularization_strength > 0.0)
		{
			printf("CalculateLoss: L1 regularization on weight matrix %u\n", i);
			regularization_loss += AddKernelL1RegularizationLoss(neural_network_layers_hyperparameters[i].number_of_neurons, neural_network_layers_hyperparameters[i + 1].number_of_neurons, neural_network_kernel_weights, kernel_weights_offset_indices[i], neural_network_layers_hyperparameters[i + 1].kernel_l1_regularization_strength, batch_size);
		}

		if (neural_network_layers_hyperparameters[i + 1].kernel_l2_regularization_strength > 0.0)
		{
			printf("CalculateLoss: L2 regularization on weight matrix %u\n", i);
			regularization_loss += AddKernelL2RegularizationLoss(neural_network_layers_hyperparameters[i].number_of_neurons, neural_network_layers_hyperparameters[i + 1].number_of_neurons, neural_network_kernel_weights, kernel_weights_offset_indices[i], neural_network_layers_hyperparameters[i + 1].kernel_l2_regularization_strength, batch_size);
		}

		if (neural_network_layers_hyperparameters[i + 1].bias_weight_initialization_type >= 0)
		{
			if (neural_network_layers_hyperparameters[i + 1].bias_l1_regularization_strength > 0.0)
			{
				printf("CalculateLoss: L1 regularization on bias vector %u\n", i);
				regularization_loss += AddBiasL1RegularizationLoss(neural_network_layers_hyperparameters[i + 1].number_of_neurons, neural_network_bias_weights, bias_weights_offset_indices[i], neural_network_layers_hyperparameters[i + 1].bias_l1_regularization_strength, batch_size);
			}

			if (neural_network_layers_hyperparameters[i + 1].bias_l2_regularization_strength > 0.0)
			{
				printf("CalculateLoss: L2 regularization on bias vector %u\n", i);
				regularization_loss += AddBiasL1RegularizationLoss(neural_network_layers_hyperparameters[i + 1].number_of_neurons, neural_network_bias_weights, bias_weights_offset_indices[i], neural_network_layers_hyperparameters[i + 1].bias_l1_regularization_strength, batch_size);
			}
		}
	} // end of i loop

	printf("CalculateLoss: loss = %.16f, regularization_loss = %.16f, total_loss = %.16f\n", loss, regularization_loss, loss + regularization_loss);

	return loss + regularization_loss;
} // end of CalculateLoss_function function

/* This function calculates the mean squared error loss typically used for regression */
double MSELossFunction(unsigned int batch_size, unsigned int number_of_outputs, double *logits, unsigned int logits_offset, unsigned long logits_size, double *labels, int transpose)
{
	/* loss = 1 / (2 * m) * sum((zi-yi)^2, i, 0, m - 1) */

	unsigned int i, j, logits_index, labels_index;
	double error, loss = 0.0;

	if (transpose == 0)
	{
		for (i = 0; i < batch_size; i++)
		{
			for (j = 0; j < number_of_outputs; j++)
			{
				logits_index = (i * number_of_outputs + j + logits_offset) * logits_size;
				labels_index = i * number_of_outputs + j;

				error = logits[logits_index] - labels[labels_index];

				/* Add example loss to batch loss */
				loss += error * error;
			} // end of j loop
		} // end of i loop
	}
	else
	{
		for (i = 0; i < batch_size; i++)
		{
			for (j = 0; j < number_of_outputs; j++)
			{
				logits_index = (j * batch_size + i + logits_offset) * logits_size;
				labels_index = j * batch_size + i;

				error = logits[logits_index] - labels[labels_index];

				/* Add example loss to batch loss */
				loss += error * error;
			} // end of j loop
		} // end of i loop
	}

	/* Convert to average loss across the batch */
	loss /= (2.0 * batch_size);

//	printf("MSELossFunction: loss = %.16f\n", loss);

	return loss;
} // end of MSELossFunction function

/* This function calculates the softmax cross entropy loss using the logits of the final layer typically used for multi-class, single label classification */
double SoftmaxCrossEntropyWithLogitsLossFunction(unsigned int batch_size, unsigned int number_of_outputs, double *logits, unsigned int logits_offset, unsigned long logits_size, double *labels, int transpose)
{
	/* loss = 1 / m * sum(e^(zi - max(z)) / sum(e^(zj - max(z)), j, 0, n - 1), i, 0, m - 1) */

	unsigned int i, j, logits_index, labels_index;
	double max_logit, denominator_sum, error, loss = 0.0;

	for (i = 0; i < batch_size; i++)
	{
		max_logit = -DBL_MAX;
		for (j = 0; j < number_of_outputs; j++)
		{
			logits_index = (i * number_of_outputs + j + logits_offset) * logits_size;

			if (logits[logits_index] > max_logit)
			{
				max_logit = logits[logits_index];
			}
		} // end of j loop

		denominator_sum = 0.0;
		for (j = 0; j < number_of_outputs; j++)
		{
			logits_index = (i * number_of_outputs + j + logits_offset) * logits_size;

			/* Shift logits by the max logit to make numerically stable */
			denominator_sum += exp(logits[logits_index] - max_logit);
		} // end of j loop
		denominator_sum = log(denominator_sum);

		/* Calculate per example loss */
		error = 0;
		for (j = 0; j < number_of_outputs; j++)
		{
			labels_index = i * number_of_outputs + j;
			logits_index = (labels_index + logits_offset) * logits_size;

			error += -labels[labels_index] * (logits[logits_index] - max_logit - denominator_sum);
		} // end of j loop

		/* Add example loss to batch loss */
		loss += error;
	} // end of i loop

	/* Convert to average loss across the batch */
	loss /= batch_size;

//	printf("SoftmaxCrossEntropyWithLogitsLossFunction: loss = %.16f\n", loss);

	return loss;
} // end of SoftmaxCrossEntropyWithLogitsLossFunction function

/* This function calculates the sigmoid cross entropy loss using the logits of the final layer typically used for multi-class, multi-label classification */
double SigmoidCrossEntropyWithLogitsLossFunction(unsigned int batch_size, unsigned int number_of_outputs, double *logits, unsigned int logits_offset, unsigned long logits_size, double *labels, int transpose)
{
	/* loss = 1 / m * sum(sum(max(zij, 0) - zij * yij + ln(1 + e^(-|zij|)), j, 0, n - 1), i, 0, m - 1) */

	unsigned int i, j, logits_index, labels_index;
	double error, loss = 0.0;

	// max(logits, 0) - logits * labels + log(1 + exp(-abs(logits)))
	for (i = 0; i < batch_size; i++)
	{
		/* Calculate per example loss */
		error = 0.0;
		for (j = 0; j < number_of_outputs; j++)
		{
			logits_index = (i * number_of_outputs + j + logits_offset) * logits_size;
			labels_index = i * number_of_outputs + j;

			error += fmax(logits[logits_index], 0.0) - logits[logits_index] * labels[labels_index] + log(1.0 + exp(-fabs(logits[logits_index])));
		} // end of j loop

		/* Add example loss to batch loss */
		loss += error;
	} // end of i loop

	/* Convert to average loss across the batch */
	loss /= batch_size;

//	printf("SigmoidCrossEntropyWithLogitsLossFunction: loss = %.16f\n", loss);

	return loss;
} // end of SigmoidCrossEntropyWithLogitsLossFunction function

/******************************************************************************************/
/************************************ REGULARIZATION **************************************/
/******************************************************************************************/

/* This function calculates L1 regularization loss of given kernel weight layer */
double AddKernelL1RegularizationLoss(unsigned int current_layer_number_of_neurons, unsigned int next_layer_number_of_neurons, struct TrainableParameters *neural_network_kernel_weights, unsigned int kernel_weights_offset, double lambda1, unsigned int batch_size)
{
	/* L = lambda1 / batch_size * sum(abs(Wi), i, 0, n - 1) */

	unsigned int i, j, kernel_index;
	double regularization_loss = 0.0;

	for (i = 0; i < current_layer_number_of_neurons; i++)
	{
		for (j = 0; j < next_layer_number_of_neurons; j++)
		{
			kernel_index = i * next_layer_number_of_neurons + j + kernel_weights_offset;

			regularization_loss += fabs(neural_network_kernel_weights[kernel_index].variable);
		} // end of j loop
	} // end of i loop
	regularization_loss *= lambda1 / batch_size;

	return regularization_loss;
} // end of AddKernelL1RegularizationLoss function

/* This function calculates L2 regularization loss of given kernel weight layer */
double AddKernelL2RegularizationLoss(unsigned int current_layer_number_of_neurons, unsigned int next_layer_number_of_neurons, struct TrainableParameters *neural_network_kernel_weights, unsigned int kernel_weights_offset, double lambda2, unsigned int batch_size)
{
	/* L = lambda2 / batch_size * sum(Wi^2, i, 0, n - 1) */

	unsigned int i, j, kernel_index;
	double regularization_loss = 0.0;

	for (i = 0; i < current_layer_number_of_neurons; i++)
	{
		for (j = 0; j < next_layer_number_of_neurons; j++)
		{
			kernel_index = i * next_layer_number_of_neurons + j + kernel_weights_offset;

			regularization_loss += neural_network_kernel_weights[kernel_index].variable * neural_network_kernel_weights[kernel_index].variable;
		} // end of j loop
	} // end of i loop
	regularization_loss *= lambda2 / batch_size;

	return regularization_loss;
} // end of AddKernelL2RegularizationLoss function

/* This function calculates L1 regularization loss of given bias weight layer */
double AddBiasL1RegularizationLoss(unsigned int next_layer_number_of_neurons, struct TrainableParameters *neural_network_bias_weights, unsigned int bias_weights_offset, double lambda1, unsigned int batch_size)
{
	/* L = lambda1 / batch_size * sum(abs(Wi), i, 0, n - 1) */

	unsigned int i, bias_index;
	double regularization_loss = 0.0;

	for (i = 0; i < next_layer_number_of_neurons; i++)
	{
		bias_index = i  + bias_weights_offset;

		regularization_loss += fabs(neural_network_bias_weights[bias_index].variable);
	} // end of i loop
	regularization_loss *= lambda1 / batch_size;

	return regularization_loss;
} // end of AddBiasL1RegularizationLoss function

/* This function calculates L2 regularization loss of given bias weight layer */
double AddBiasL2RegularizationLoss(unsigned int next_layer_number_of_neurons, struct TrainableParameters *neural_network_bias_weights, unsigned int bias_weights_offset, double lambda2, unsigned int batch_size)
{
	/* L = lambda2 / batch_size * sum(Wi^2, i, 0, n - 1) */

	unsigned int i, bias_index;
	double regularization_loss = 0.0;

	for (i = 0; i < next_layer_number_of_neurons; i++)
	{
		bias_index = i  + bias_weights_offset;

		regularization_loss += neural_network_bias_weights[bias_index].variable * neural_network_bias_weights[bias_index].variable;
	} // end of i loop
	regularization_loss *= lambda2 / batch_size;

	return regularization_loss;
} // end of AddBiasL2RegularizationLoss function

/******************************************************************************************/
/************************************** EVALUATION ****************************************/
/******************************************************************************************/

/* This function evaluates the currently trained model on the valdiation dataset */
void ModelEvaluation(unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int batch_size, double *features, double *labels, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int *neural_network_neurons_minus_inputs_offset_indices, struct TrainableParameters *neural_network_kernel_weights, unsigned int *kernel_weights_offset_indices, struct TrainableParameters *neural_network_bias_weights, unsigned int *bias_weights_offset_indices, struct TrainableParameters *parametric_relu_alphas, unsigned int *parametric_relu_alpha_offset_indices, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, double *batch_normalization_layer_xmu, unsigned int *batch_normalization_layer_offset_indices, int *dropout_layer_mask_neurons, unsigned int *dropout_layer_offset_indices, int training, int transpose, struct RegressionEvaluationMetric *regression_evaluation_metrics, struct ClassificationEvaluationMetric *classification_evaluation_metrics, unsigned int *evaluation_count, unsigned int step)
{
	double loss = 0.0;

	loss = FeedForward(number_of_layers, neural_network_layers_hyperparameters, training_validation_hyperparameters, batch_size, features, labels, neural_network_layer_neurons, neural_network_neurons_minus_inputs_offset_indices, neural_network_kernel_weights, kernel_weights_offset_indices, neural_network_bias_weights, bias_weights_offset_indices, parametric_relu_alphas, parametric_relu_alpha_offset_indices, batch_normalization_layer_neurons_parameters, batch_normalization_layer_xmu, batch_normalization_layer_offset_indices, dropout_layer_mask_neurons, dropout_layer_offset_indices, 0, transpose);

	if (training_validation_hyperparameters->loss_function_type == 0) // mse/regression
	{
		CalculateRegressionEvaluationMetrics(batch_size, neural_network_layers_hyperparameters[number_of_layers - 1].number_of_neurons, &neural_network_layer_neurons[0].activation, batch_size * neural_network_neurons_minus_inputs_offset_indices[number_of_layers - 2], sizeof(struct NeuralNetworkLayerNeuron) / sizeof(void*), labels, &regression_evaluation_metrics[(*evaluation_count)].mse, &regression_evaluation_metrics[(*evaluation_count)].rmse);

		regression_evaluation_metrics[(*evaluation_count)].train_step = step;
		regression_evaluation_metrics[(*evaluation_count)].loss = loss;
	}
	else // classification
	{
		if (training_validation_hyperparameters->loss_function_type == 1) // softmax cross entropy/multi-class, single-label classification
		{

			CalculateClassificationEvaluationMetrics(batch_size, neural_network_layers_hyperparameters[number_of_layers - 1].number_of_neurons, &neural_network_layer_neurons[0].activation, batch_size * neural_network_neurons_minus_inputs_offset_indices[number_of_layers - 2], sizeof(struct NeuralNetworkLayerNeuron) / sizeof(void*), labels, training_validation_hyperparameters->classification_threshold, 0, training_validation_hyperparameters->alpha_forgiveness_rate, training_validation_hyperparameters->beta_false_negative_cost, training_validation_hyperparameters->gamma_false_positive_cost, &classification_evaluation_metrics[(*evaluation_count)].exact_match_ratio, &classification_evaluation_metrics[(*evaluation_count)].accuracy, &classification_evaluation_metrics[(*evaluation_count)].avg_precision, &classification_evaluation_metrics[(*evaluation_count)].avg_recall, &classification_evaluation_metrics[(*evaluation_count)].f1_score);
		}
		else if (training_validation_hyperparameters->loss_function_type == 2) // sigmoid cross entropy/multi-class, multi-label classification
		{
			CalculateClassificationEvaluationMetrics(batch_size, neural_network_layers_hyperparameters[number_of_layers - 1].number_of_neurons, &neural_network_layer_neurons[0].activation, batch_size * neural_network_neurons_minus_inputs_offset_indices[number_of_layers - 2], sizeof(struct NeuralNetworkLayerNeuron) / sizeof(void*), labels, training_validation_hyperparameters->classification_threshold, 1, training_validation_hyperparameters->alpha_forgiveness_rate, training_validation_hyperparameters->beta_false_negative_cost, training_validation_hyperparameters->gamma_false_positive_cost, &classification_evaluation_metrics[(*evaluation_count)].exact_match_ratio, &classification_evaluation_metrics[(*evaluation_count)].accuracy, &classification_evaluation_metrics[(*evaluation_count)].avg_precision, &classification_evaluation_metrics[(*evaluation_count)].avg_recall, &classification_evaluation_metrics[(*evaluation_count)].f1_score);
		}

		classification_evaluation_metrics[(*evaluation_count)].train_step = step;
		classification_evaluation_metrics[(*evaluation_count)].loss = loss;
	}

	/* Increment evaluation count now that it is complete */
	(*evaluation_count)++;
} // end of ModelEvaluation function

/* This function calculates common regression evaluation metrics */
void CalculateRegressionEvaluationMetrics(unsigned int batch_size, unsigned int number_of_outputs, double *logits, unsigned int logits_offset, unsigned long logits_size, double *labels, double *mse, double *rmse)
{
	unsigned int i, j, logits_index, labels_index;
	double error;

	// Zero mse first
	(*mse) = 0.0;

	for (i = 0; i < batch_size; i++)
	{
		for (j = 0; j < number_of_outputs; j++)
		{
			labels_index = i * number_of_outputs + j;
			logits_index = (labels_index + logits_offset) * logits_size;

			error = labels[labels_index] - logits[logits_index];
			(*mse) += error * error;
		} // end of j loop
	} // end of i loop

	/* Take the sqrt of mse to obtain rmse */
	(*rmse) = sqrt((*mse));

	printf("CalculateRegressionEvaluationMetrics: mean_squared_error = %.16f, root_mean_squared_error = %.16f\n", (*mse), (*rmse));
} // end of CalculateRegressionEvaluationMetrics function

/* This function calculates common classification evaluation metrics */
void CalculateClassificationEvaluationMetrics(unsigned int batch_size, unsigned int number_of_outputs, double *probabilities, unsigned int probabilities_offset, unsigned long probabilities_size, double *labels, double classification_threshold, int multilabel, double alpha_forgiveness_rate, double beta_false_negative_cost, double gamma_false_positive_cost, double *exact_match_ratio, double *accuracy, double *avg_precision, double *avg_recall, double *f1_score)
{
	unsigned int i, j, probabilities_index, labels_index;
	int prediction, label, true_positive;

	/* Zero metrics first */
	(*exact_match_ratio) = 0.0;
	(*accuracy) = 0.0;
	(*avg_precision) = 0.0;
	(*avg_recall) = 0.0;
	(*f1_score) = 0.0;

	if (multilabel == 0)
	{
		/* Create structure to capture metrics for each class */
		struct classification_metric
		{
			unsigned int correct;
			unsigned int precision;
			unsigned int recall;
		};

		struct classification_metric *classification_metrics;
		classification_metrics = malloc(sizeof(struct classification_metric) * number_of_outputs);
		for (j = 0; j < number_of_outputs; j++)
		{
			classification_metrics[j].correct = 0;
			classification_metrics[j].precision = 0;
			classification_metrics[j].recall = 0;
		} // end of j loop

		unsigned int max_probability_index;
		double max_probability_value;

		for (i = 0; i < batch_size; i++)
		{
			max_probability_index = 0;
			max_probability_value = 0.0;

			for (j = 0; j < number_of_outputs; j++)
			{
				labels_index = i * number_of_outputs + j;
				probabilities_index = (labels_index + probabilities_offset) * probabilities_size;

				if (probabilities[probabilities_index] > max_probability_value)
				{
					max_probability_index = j;
					max_probability_value = probabilities[probabilities_index];
				}

				classification_metrics[j].precision += (int)labels[labels_index];
			} // end of j loop

			prediction  = max_probability_value >= classification_threshold;
			label = (int)labels[i * number_of_outputs + max_probability_index];

			true_positive = (prediction == label);
			classification_metrics[max_probability_index].correct += true_positive;
			classification_metrics[max_probability_index].recall += 1;
		} // end of i loop

		for (j = 0; j < number_of_outputs; j++)
		{
			(*accuracy) += classification_metrics[j].correct;
			(*avg_precision) += classification_metrics[j].precision == 0 ? 0 : (double)classification_metrics[j].correct / classification_metrics[j].precision;
			(*avg_recall) += classification_metrics[j].recall == 0 ? 0 : (double)classification_metrics[j].correct / classification_metrics[j].recall;
		} // end of j loop

		/* Accuracy */
		(*accuracy) /= batch_size;

		/* Precision */
		(*avg_precision) /= number_of_outputs;

		/* Recall */
		(*avg_recall) /= number_of_outputs;

		/* F1 Score */
		(*f1_score) = 2.0 * ((*avg_precision) * (*avg_recall)) / ((*avg_precision) + (*avg_recall));

		/* Exact match */
		(*exact_match_ratio) = (*accuracy);

		/* Free dynamically allocated memory */
		free(classification_metrics);
	}
	else // multi-class, multi-label
	{
		unsigned int exact_match, true_intersections, true_unions, true_actuals, true_predictions, false_negatives, false_positives;
		unsigned int exact_matches = 0;
		double acc;

		for (i = 0; i < batch_size; i++)
		{
			exact_match = 0;
			true_intersections = 0;
			true_unions = 0;
			true_actuals = 0;
			true_predictions = 0;
			false_negatives = 0;
			false_positives = 0;
			for (j = 0; j < number_of_outputs; j++)
			{
				labels_index = i * number_of_outputs + j;
				probabilities_index = (labels_index + probabilities_offset) * probabilities_size;

				prediction  = probabilities[probabilities_index] >= classification_threshold;
				label = (int)labels[labels_index];

				/* Exact match */
				if (prediction == label) // exact match between prediction and label
				{
					exact_match++;
				}

				/* Accuracy, precision, recall */
				if (label == 1)
				{
					if (prediction == 1)
					{
						true_intersections++;
						true_unions++;
						true_predictions++;
					}
					else // prediction == 0
					{
						true_unions++;
						false_negatives++;
					}
					true_actuals++;
				}
				else // label == 0
				{
					if (prediction == 1)
					{
						true_unions++;
						true_predictions++;
						false_positives++;
					}
				}
			} // end of j loop

			/* Exact match */
			if (exact_match == number_of_outputs)
			{
				exact_matches++;
			}

			/* Accuracy */
			// acc += pow((double)true_intersections / true_unions, alpha_forgiveness_rate); // where beta_false_negative_cost == gamma_false_positive_cost == 1
			acc = pow(1.0 - (beta_false_negative_cost * false_negatives + gamma_false_positive_cost * false_positives) / true_unions, alpha_forgiveness_rate);
			(*accuracy) += acc < 0 ? 0 : acc; // some choices of beta and gamma could make it negative so we will bound it by zero

			/* Precision */
			(*avg_precision) += (double)true_intersections / true_predictions;

			/* Recall */
			(*avg_recall) += (double)true_intersections / true_actuals;

			/* F1 Score */
			(*f1_score) += 2.0 * (double)true_intersections / (true_actuals + true_predictions);
		} // end of i loop
		(*exact_match_ratio) = (double)exact_matches / batch_size;
		(*accuracy) /= batch_size;
		(*avg_precision) /= batch_size;
		(*avg_recall) /= batch_size;
		(*f1_score) /= batch_size;
	}

	printf("CalculateClassificationEvaluationMetrics: exact_match_ratio = %.16f, accuracy = %.16f, avg_precision = %.16f, avg_recall = %.16f, f1_score = %.16f\n", (*exact_match_ratio), (*accuracy), (*avg_precision), (*avg_recall), (*f1_score));
} // end of CalculateClassificationEvaluationMetrics function

/******************************************************************************************/
/************************************ BACKPROPAGATION *************************************/
/******************************************************************************************/

/* This function backpropagates the error through each layer of the neural network to find the parameter gradients to update the parameters */
void Backpropagation(unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int batch_size, double *batch_features_tensor, double *batch_labels_tensor, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int *neural_network_neurons_minus_inputs_offset_indices, struct TrainableParameters *neural_network_kernel_weights, unsigned int *kernel_weights_offset_indices, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int *bias_weights_offset_indices, unsigned int total_bias_weights, struct TrainableParameters *parametric_relu_alphas, unsigned int *parametric_relu_alpha_offset_indices, unsigned int total_parametric_relu_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, double *batch_normalization_layer_xmu, unsigned int *batch_normalization_layer_offset_indices, unsigned int total_batch_normalization_neurons, int *dropout_layer_mask_neurons, unsigned int *dropout_layer_offset_indices, int transpose, unsigned int train_step)
{
	printf("Backpropagation: Entering function\n");
	unsigned int i, j, k, labels_index, neuron_index, parametric_relu_index, batch_norm_index;

	/* Calculate neuron deltas */
	if (transpose == 0)
	{
		// Just output layer
		printf("Backpropagation: Output layer\n");
		for (i = 0; i < batch_size; i++)
		{
			for (j = 0; j < neural_network_layers_hyperparameters[number_of_layers - 1].number_of_neurons; j++)
			{
				labels_index = i * neural_network_layers_hyperparameters[number_of_layers - 1].number_of_neurons + j;
				neuron_index = i * neural_network_layers_hyperparameters[number_of_layers - 1].number_of_neurons + j + batch_size * neural_network_neurons_minus_inputs_offset_indices[number_of_layers - 2];

				neural_network_layer_neurons[neuron_index].delta = neural_network_layer_neurons[neuron_index].activation - batch_labels_tensor[labels_index];
			} // end of j loop
		} // end of i loop

		/* Rest of the layers working backwards */
		for (i = number_of_layers - 2; i >= 1; i--)
		{
			printf("Backpropagation: layer = %u\n", i);
			/* For just the neural network weights */
			MatrixMultiplication(batch_size, neural_network_layers_hyperparameters[i].number_of_neurons, neural_network_layers_hyperparameters[i + 1].number_of_neurons, &neural_network_layer_neurons[0].delta, &neural_network_kernel_weights[0].variable, &neural_network_layer_neurons[0].delta, batch_size * neural_network_neurons_minus_inputs_offset_indices[i], kernel_weights_offset_indices[i], batch_size * neural_network_neurons_minus_inputs_offset_indices[i - 1], sizeof(struct NeuralNetworkLayerNeuron) / sizeof(void*), sizeof(struct TrainableParameters) / sizeof(void*), sizeof(struct NeuralNetworkLayerNeuron) / sizeof(void*), 0, 1);

			if (neural_network_layers_hyperparameters[i].dropout_probability > 0.0)
			{
				ApplyDropoutGradient(batch_size, neural_network_layers_hyperparameters[i].number_of_neurons, neural_network_layer_neurons, batch_size * neural_network_neurons_minus_inputs_offset_indices[i - 1], dropout_layer_mask_neurons, dropout_layer_offset_indices[i - 1], neural_network_layers_hyperparameters[i].dropout_probability);
			}

			if (neural_network_layers_hyperparameters[i].batch_norm_flag == 1)
			{
				if (neural_network_layers_hyperparameters[i].batch_norm_after_activation_flag == 0)
				{
					/* Element-wise multiply with derivative of activation function */
					ApplyDerivativeActivationFunction(neural_network_layers_hyperparameters[i].activation_type, batch_size, neural_network_layers_hyperparameters[i].number_of_neurons, neural_network_layer_neurons, batch_size * neural_network_neurons_minus_inputs_offset_indices[i - 1], neural_network_layers_hyperparameters[i].activation_function_alpha_initializer, parametric_relu_alphas, parametric_relu_alpha_offset_indices[i - 1]);

					ApplyBatchNormalizationGradient(batch_size, neural_network_layers_hyperparameters[i].number_of_neurons, neural_network_layer_neurons, batch_size * neural_network_neurons_minus_inputs_offset_indices[i - 1], batch_normalization_layer_neurons_parameters, batch_normalization_layer_xmu, batch_normalization_layer_offset_indices[i - 1], training_validation_hyperparameters->learning_rate);
				}
				else
				{
					ApplyBatchNormalizationGradient(batch_size, neural_network_layers_hyperparameters[i].number_of_neurons, neural_network_layer_neurons, batch_size * neural_network_neurons_minus_inputs_offset_indices[i - 1], batch_normalization_layer_neurons_parameters, batch_normalization_layer_xmu, batch_normalization_layer_offset_indices[i - 1], training_validation_hyperparameters->learning_rate);

					/* Element-wise multiply with derivative of activation function */
					ApplyDerivativeActivationFunction(neural_network_layers_hyperparameters[i].activation_type, batch_size, neural_network_layers_hyperparameters[i].number_of_neurons, neural_network_layer_neurons, batch_size * neural_network_neurons_minus_inputs_offset_indices[i - 1], neural_network_layers_hyperparameters[i].activation_function_alpha_initializer, parametric_relu_alphas, parametric_relu_alpha_offset_indices[i - 1]);
				}
			}
			else
			{
				/* Element-wise multiply with derivative of activation function */
				ApplyDerivativeActivationFunction(neural_network_layers_hyperparameters[i].activation_type, batch_size, neural_network_layers_hyperparameters[i].number_of_neurons, neural_network_layer_neurons, batch_size * neural_network_neurons_minus_inputs_offset_indices[i - 1], neural_network_layers_hyperparameters[i].activation_function_alpha_initializer, parametric_relu_alphas, parametric_relu_alpha_offset_indices[i - 1]);
			}

			/* Calculate parametric relu alphas */
			if (neural_network_layers_hyperparameters[i].activation_type == 5)
			{
				CalculateParametricReLUGradients(batch_size, neural_network_layers_hyperparameters[i].number_of_neurons, neural_network_layer_neurons, batch_size * neural_network_neurons_minus_inputs_offset_indices[i - 1], parametric_relu_alphas, parametric_relu_alpha_offset_indices[i - 1]);
			}
		} // end of i loop

		/* Update weights using activations and deltas */
		printf("Backpropagation: Updating weight layer %u\n", 0);

		CalculateNeuralNetworkKernelWeightGradients(batch_size, neural_network_layers_hyperparameters[0].number_of_neurons, neural_network_layers_hyperparameters[1].number_of_neurons, batch_features_tensor, &neural_network_layer_neurons[0].delta, neural_network_kernel_weights, 1, sizeof(struct NeuralNetworkLayerNeuron) / sizeof(void*), 0, 0, 0);

		if (neural_network_layers_hyperparameters[1].bias_weight_initialization_type >= 0)
		{
			CalculateNeuralNetworkBiasWeightGradients(batch_size, neural_network_layers_hyperparameters[1].number_of_neurons, neural_network_layer_neurons, neural_network_bias_weights, batch_size * neural_network_neurons_minus_inputs_offset_indices[0], bias_weights_offset_indices[0]);
		}

		for (i = 1; i < number_of_layers - 1; i++)
		{
			printf("Backpropagation: Updating weight layer %u\n", i);

			CalculateNeuralNetworkKernelWeightGradients(batch_size, neural_network_layers_hyperparameters[i].number_of_neurons, neural_network_layers_hyperparameters[i + 1].number_of_neurons, &neural_network_layer_neurons[0].activation, &neural_network_layer_neurons[0].delta, neural_network_kernel_weights, sizeof(struct NeuralNetworkLayerNeuron) / sizeof(void*), sizeof(struct NeuralNetworkLayerNeuron) / sizeof(void*), batch_size * neural_network_neurons_minus_inputs_offset_indices[i - 1], batch_size * neural_network_neurons_minus_inputs_offset_indices[i], kernel_weights_offset_indices[i]);

			if (neural_network_layers_hyperparameters[i + 1].bias_weight_initialization_type >= 0)
			{
				CalculateNeuralNetworkBiasWeightGradients(batch_size, neural_network_layers_hyperparameters[i + 1].number_of_neurons, neural_network_layer_neurons, neural_network_bias_weights, batch_size * neural_network_neurons_minus_inputs_offset_indices[i], bias_weights_offset_indices[i]);
			}
		} // end of i loop

		/* Add regularization to gradients */
		for (i = 0; i < number_of_layers - 1; i++)
		{
			if (neural_network_layers_hyperparameters[i + 1].kernel_l1_regularization_strength > 0.0)
			{
				printf("Backpropagation: L1 regularization on weight matrix %u\n", i);
				AddKernelL1RegularizationGradient(neural_network_layers_hyperparameters[i].number_of_neurons, neural_network_layers_hyperparameters[i + 1].number_of_neurons, neural_network_kernel_weights, kernel_weights_offset_indices[i], neural_network_layers_hyperparameters[i + 1].kernel_l1_regularization_strength, batch_size);
			}

			if (neural_network_layers_hyperparameters[i + 1].kernel_l2_regularization_strength > 0.0)
			{
				printf("Backpropagation: L2 regularization on weight matrix %u\n", i);
				AddKernelL2RegularizationGradient(neural_network_layers_hyperparameters[i].number_of_neurons, neural_network_layers_hyperparameters[i + 1].number_of_neurons, neural_network_kernel_weights, kernel_weights_offset_indices[i], neural_network_layers_hyperparameters[i + 1].kernel_l2_regularization_strength, batch_size);
			}

			if (neural_network_layers_hyperparameters[i + 1].bias_weight_initialization_type >= 0)
			{
				if (neural_network_layers_hyperparameters[i + 1].bias_l1_regularization_strength > 0.0)
				{
					printf("Backpropagation: L1 regularization on bias vector %u\n", i);
					AddBiasL1RegularizationGradient(neural_network_layers_hyperparameters[i + 1].number_of_neurons, neural_network_bias_weights, bias_weights_offset_indices[i], neural_network_layers_hyperparameters[i + 1].bias_l1_regularization_strength, batch_size);
				}

				if (neural_network_layers_hyperparameters[i + 1].bias_l2_regularization_strength > 0.0)
				{
					printf("Backpropagation: L2 regularization on bias vector %u\n", i);
					AddBiasL1RegularizationGradient(neural_network_layers_hyperparameters[i + 1].number_of_neurons, neural_network_bias_weights, bias_weights_offset_indices[i], neural_network_layers_hyperparameters[i + 1].bias_l1_regularization_strength, batch_size);
				}
			}
		} // end of i loop

		/* Apply gradient clipping */
		if (training_validation_hyperparameters->clip_norm > 0.0)
		{
			ApplyGradientClippingByGlobalNorm(training_validation_hyperparameters, total_kernel_weights, neural_network_kernel_weights, total_bias_weights, neural_network_bias_weights, total_parametric_relu_neurons, parametric_relu_alphas, total_batch_normalization_neurons, batch_normalization_layer_neurons_parameters);
		}

		/* Now that the gradients are calculated, apply optimizers */
		ApplyGradientOptimizers(training_validation_hyperparameters, total_kernel_weights, neural_network_kernel_weights, total_bias_weights, neural_network_bias_weights, total_parametric_relu_neurons, parametric_relu_alphas, total_batch_normalization_neurons, batch_normalization_layer_neurons_parameters, train_step);

		/* Now that gradients are finalized and optimized, update all of the trainable parameters */
		UpdateNeuralNetworkTrainableParameters(total_kernel_weights, neural_network_kernel_weights);
		UpdateNeuralNetworkTrainableParameters(total_bias_weights, neural_network_bias_weights);
		UpdateNeuralNetworkTrainableParameters(total_parametric_relu_neurons, parametric_relu_alphas);
		UpdateBatchNormalizationParameters(total_batch_normalization_neurons, batch_normalization_layer_neurons_parameters);
	}

	if (debug_print == 1)
	{
		DebugPrintBackpropagation(number_of_layers, neural_network_layers_hyperparameters, batch_size, batch_labels_tensor, neural_network_layer_neurons, neural_network_neurons_minus_inputs_offset_indices, neural_network_kernel_weights, kernel_weights_offset_indices, neural_network_bias_weights, bias_weights_offset_indices, parametric_relu_alphas, parametric_relu_alpha_offset_indices, batch_normalization_layer_neurons_parameters, batch_normalization_layer_offset_indices);
	}
} // end of Backpropagation function

/******************************************************************************************/
/************************************** DERIVATIVES ***************************************/
/******************************************************************************************/

/* This function applies derivative of given activation function to given layers weighted sums and returns result as layer's deltas */
void ApplyDerivativeActivationFunction(int activation_type, unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int neuron_offset, double non_parametric_alpha, struct TrainableParameters *parametric_relu_alphas, unsigned int parametric_relu_offset)
{
	/* Layer activation types */
	/*
	name		activation_type		f(x)								f'(x)
	linear		0					x									1
	sigmoid		1					1 / (1 + exp(-x))					f(x) * (1 - f(x))
	tanh		2					2 / (1 + exp(-2 * x)) - 1			1 - f(x)^2
	relu		3					max(0, x)							x < 0 ? 0 : 1
	leaky_relu	4					max(0.01 * x, x)					x < 0 ? 0.01 : 1
	prelu		5					max(alpha * x, x)					x < 0 ? alpha : 1
	elu			6					x < 0 ? alpha * (exp(x) - 1) : x	x < 0 ? f(alpha, x) + alpha : 1
	 */

	printf("ApplyDerivativeActivationFunction: activation_type = %d, batch_size = %u, number_of_neurons = %u, offset = %u\n", activation_type, batch_size, number_of_neurons, neuron_offset);

	switch (activation_type)
	{
		case 0:
			ApplyDerivativeLinearActivationFunction(batch_size, number_of_neurons, neural_network_layer_neurons, neuron_offset);
			break;
		case 1:
			ApplyDerivativeSigmoidActivationFunction(batch_size, number_of_neurons, neural_network_layer_neurons, neuron_offset);
			break;
		case 2:
			ApplyDerivativeTanhActivationFunction(batch_size, number_of_neurons, neural_network_layer_neurons, neuron_offset);
			break;
		case 3:
			ApplyDerivativeReLUActivationFunction(batch_size, number_of_neurons, neural_network_layer_neurons, neuron_offset);
			break;
		case 4:
			ApplyDerivativeLeakyReLUActivationFunction(batch_size, number_of_neurons, neural_network_layer_neurons, neuron_offset, non_parametric_alpha);
			break;
		case 5:
			ApplyDerivativeParametricReLUActivationFunction(batch_size, number_of_neurons, neural_network_layer_neurons, neuron_offset, parametric_relu_alphas, parametric_relu_offset);
			break;
		case 6:
			ApplyDerivativeEluActivationFunction(batch_size, number_of_neurons, neural_network_layer_neurons, neuron_offset, non_parametric_alpha);
			break;
		default:
			ApplyDerivativeLinearActivationFunction(batch_size, number_of_neurons, neural_network_layer_neurons, neuron_offset);
			printf("ApplyDerivativeActivationFunction: No activation function to apply\n");
	}
} // end of ApplyDerivativeActivationFunction function

/* This function applies the derivative of linear activation function to given layer's neurons */
void ApplyDerivativeLinearActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int neuron_offset)
{
	unsigned int i, j, neuron_index;

	for (i = 0; i < batch_size; i++)
	{
		for (j = 0; j < number_of_neurons; j++)
		{
			neuron_index = i * number_of_neurons + j + neuron_offset;

			neural_network_layer_neurons[neuron_index].delta *= DerivativeActivationFunctionLinear(neural_network_layer_neurons[neuron_index].weighted_sum);
		} // end of j loop
	} // end of i loop
} // end of ApplyDerivativeLinearActivationFunction function

/* This function returns the derivative of linear activation */
double DerivativeActivationFunctionLinear(double x)
{
	/* f'(x) = 1 */

	return 1.0;
} // end of DerivativeActivationFunctionLinear function

/* This function applies the derivative of sigmoid activation function to given layer's neurons */
void ApplyDerivativeSigmoidActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int neuron_offset)
{
	unsigned int i, j, neuron_index;

	for (i = 0; i < batch_size; i++)
	{
		for (j = 0; j < number_of_neurons; j++)
		{
			neuron_index = i * number_of_neurons + j + neuron_offset;

			neural_network_layer_neurons[neuron_index].delta *= DerivativeActivationFunctionSigmoid(neural_network_layer_neurons[neuron_index].weighted_sum);
		} // end of j loop
	} // end of i loop
} // end of ApplyDerivativeSigmoidActivationFunction function

/* This function returns the derivative of sigmoid activation */
double DerivativeActivationFunctionSigmoid(double x)
{
	/* f'(x) = sigmoid(x) * (1 - sigmoid(x)) */

	return ActivationFunctionSigmoid(x) * (1.0 - ActivationFunctionSigmoid(x));
} // end of DerivativeActivationFunctionSigmoid function

/* This function applies the derivative of tanh activation function to given layer's neurons */
void ApplyDerivativeTanhActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int neuron_offset)
{
	unsigned int i, j, neuron_index;

	for (i = 0; i < batch_size; i++)
	{
		for (j = 0; j < number_of_neurons; j++)
		{
			neuron_index = i * number_of_neurons + j + neuron_offset;

			neural_network_layer_neurons[neuron_index].delta *= DerivativeActivationFunctionTanh(neural_network_layer_neurons[neuron_index].weighted_sum);
		} // end of j loop
	} // end of i loop
} // end of ApplyDerivativeTanhActivationFunction function

/* This function returns the derivative of tanh activation */
double DerivativeActivationFunctionTanh(double x)
{
	/* f'(x) = 1 - tanh(x)^2 */

	double tanh = ActivationFunctionTanh(x);
	return 1.0 - tanh * tanh;
} // end of DerivativeActivationFunctionTanh function

/* This function applies the derivative of ReLU activation function to given layer's neurons */
void ApplyDerivativeReLUActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int neuron_offset)
{
	unsigned int i, j, neuron_index;

	for (i = 0; i < batch_size; i++)
	{
		for (j = 0; j < number_of_neurons; j++)
		{
			neuron_index = i * number_of_neurons + j + neuron_offset;

			if (fabs(neural_network_layer_neurons[neuron_index].weighted_sum) < DBL_EPSILON)
			{
				neural_network_layer_neurons[neuron_index].weighted_sum = 0.0;
			}

			neural_network_layer_neurons[neuron_index].delta *= DerivativeActivationFunctionReLU(neural_network_layer_neurons[neuron_index].weighted_sum);
		} // end of j loop
	} // end of i loop
} // end of ApplyDerivativeReLUActivationFunction function

/* This function returns the derivative of ReLU activation */
double DerivativeActivationFunctionReLU(double x)
{
	/* 	if x < 0
			f'(x) = 0
		else
			f'(x) = 1
	*/

	return x < 0.0 ? 0.0 : 1.0;
} // end of DerivativeActivationFunctionReLU function

/* This function applies the derivative of leaky ReLU activation function to given layer's neurons */
void ApplyDerivativeLeakyReLUActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int neuron_offset, double alpha)
{
	unsigned int i, j, neuron_index;

	for (i = 0; i < batch_size; i++)
	{
		for (j = 0; j < number_of_neurons; j++)
		{
			neuron_index = i * number_of_neurons + j + neuron_offset;

			if (fabs(neural_network_layer_neurons[neuron_index].weighted_sum) < DBL_EPSILON)
			{
				neural_network_layer_neurons[neuron_index].weighted_sum = 0.0;
			}

			neural_network_layer_neurons[neuron_index].delta *= DerivativeActivationFunctionParametricReLU(neural_network_layer_neurons[neuron_index].weighted_sum, alpha);
		} // end of j loop
	} // end of i loop
} // end of ApplyDerivativeLeakyReLUActivationFunction function

/* This function applies the derivative of parametric ReLU activation function to given layer's neurons */
void ApplyDerivativeParametricReLUActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int neuron_offset, struct TrainableParameters *parametric_relu_alphas, unsigned int parametric_relu_offset)
{
	unsigned int i, j, neuron_index, parametric_relu_index;

	for (i = 0; i < batch_size; i++)
	{
		for (j = 0; j < number_of_neurons; j++)
		{
			neuron_index = i * number_of_neurons + j + neuron_offset;
			parametric_relu_index = j + parametric_relu_offset;

			if (fabs(neural_network_layer_neurons[neuron_index].weighted_sum) < DBL_EPSILON)
			{
				neural_network_layer_neurons[neuron_index].weighted_sum = 0.0;
			}

			neural_network_layer_neurons[neuron_index].delta *= DerivativeActivationFunctionParametricReLU(neural_network_layer_neurons[neuron_index].weighted_sum, parametric_relu_alphas[parametric_relu_index].variable);
		} // end of j loop
	} // end of i loop
} // end of ApplyDerivativeParametricReLUActivationFunction function

/* This function returns the derivative of parametric ReLU activation */
double DerivativeActivationFunctionParametricReLU(double x, double alpha)
{
	/* 	if x < 0
			f'(x) = alpha
		else
			f'(x) = 1
	*/

	return x < 0.0 ? alpha : 1.0;
} // end of DerivativeActivationFunctionParametricReLU function

/* This function applies the derivative of elu activation function to given layer's neurons */
void ApplyDerivativeEluActivationFunction(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int neuron_offset, double alpha)
{
	unsigned int i, j, neuron_index;

	for (i = 0; i < batch_size; i++)
	{
		for (j = 0; j < number_of_neurons; j++)
		{
			neuron_index = i * number_of_neurons + j + neuron_offset;

			if (fabs(neural_network_layer_neurons[neuron_index].weighted_sum) < DBL_EPSILON)
			{
				neural_network_layer_neurons[neuron_index].weighted_sum = 0;
			}

			neural_network_layer_neurons[neuron_index].delta *= DerivativeActivationFunctionElu(neural_network_layer_neurons[neuron_index].weighted_sum, alpha);
		} // end of j loop
	} // end of i loop
} // end of ApplyDerivativeEluActivationFunction function

/* This function returns the derivative of elu activation */
double DerivativeActivationFunctionElu(double x, double alpha)
{
	/* 	if x < 0
			f'(x) = elu(x, alpha) + alpha
		else
			f'(x) = 1
	*/

	return x < 0.0 ? ActivationFunctionElu(x, alpha) + alpha : 1.0;
} // end of DerivativeActivationFunctionElu function

/******************************************************************************************/
/*************************************** GRADIENTS ****************************************/
/******************************************************************************************/

/* This function applies the gradient of batch normalization to the given layer */
void ApplyBatchNormalizationGradient(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int activations_offset, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, double *batch_normalization_layer_xmu, unsigned int batch_normalization_layer_offset, double learning_rate)
{
	unsigned int i, j, batch_norm_index, neuron_index, xmu_index;
	double delta_sum, delta_xmu_sum, delta_gamma;

	for (j = 0; j < number_of_neurons; j++)
	{
		delta_sum = 0.0;
		delta_xmu_sum = 0.0;
		delta_gamma = 0.0;

		batch_norm_index = j + batch_normalization_layer_offset;

		for (i = 0; i < batch_size; i++)
		{
			neuron_index = i * number_of_neurons + j + activations_offset;
			xmu_index = i * number_of_neurons + batch_size * batch_normalization_layer_offset + j;

			delta_sum += neural_network_layer_neurons[neuron_index].delta;
			delta_xmu_sum += neural_network_layer_neurons[neuron_index].delta * batch_normalization_layer_xmu[xmu_index];
			delta_gamma += neural_network_layer_neurons[neuron_index].delta * (neural_network_layer_neurons[neuron_index].activation - batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_beta.variable) / batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_gamma.variable;
		} // end of i loop

		for (i = 0; i < batch_size; i++)
		{
			neuron_index = i * number_of_neurons + j + activations_offset;
			xmu_index = i * number_of_neurons + batch_size * batch_normalization_layer_offset + j;

			neural_network_layer_neurons[neuron_index].delta = batch_size * neural_network_layer_neurons[neuron_index].delta - delta_sum - batch_normalization_layer_xmu[xmu_index] / (batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_variance + epsilon) * delta_xmu_sum;

			neural_network_layer_neurons[neuron_index].delta *= batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_gamma.variable / (batch_size * sqrt(batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_variance + epsilon));
		} // end of i loop

		/* Since we've already calculated the deltas we can use them to calculate the batch normalization parameter gradients right now rather than later */
		CalculateBatchNormalizationParameterGradients(batch_norm_index, batch_size, batch_normalization_layer_neurons_parameters, delta_sum, delta_gamma);
	} // end of j loop
} // end of ApplyBatchNormalizationGradient function

/* This function applies the gradient of dropout to the given layer */
void ApplyDropoutGradient(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int activations_offset, int *dropout_layer_mask_neurons, unsigned int dropout_offset, double dropout_probability)
{
	unsigned int i, j, dropout_index, neuron_index;
	double keep_probability = 1.0 - dropout_probability;

	for (j = 0; j < number_of_neurons; j++)
	{
		dropout_index = j + dropout_offset;

		if (dropout_layer_mask_neurons[dropout_index] == 0)
		{
			for (i = 0; i < batch_size; i++)
			{
				neuron_index = i * number_of_neurons + j + activations_offset;

				neural_network_layer_neurons[neuron_index].delta = 0.0;
			} // end of i loop
		}
		else
		{
			for (i = 0; i < batch_size; i++)
			{
				neuron_index = i * number_of_neurons + j + activations_offset;

				neural_network_layer_neurons[neuron_index].delta /= keep_probability;
			} // end of i loop
		}
	} // end of j loop
} // end of ApplyDropoutGradient function

/* This function calculates the gradients of the neural network kernel weights */
void CalculateNeuralNetworkKernelWeightGradients(unsigned int batch_size, unsigned int number_of_neurons_in, unsigned int number_of_neurons_out, double *activations, double *deltas, struct TrainableParameters *neural_network_kernel_weights, unsigned long activation_size, unsigned long delta_size, unsigned int activation_offset, unsigned int delta_offset, unsigned int kernel_offset)
{
	/* grad = matmul(aT, d) / m */

	unsigned int i, j, kernel_index;
	MatrixMultiplication(number_of_neurons_in, number_of_neurons_out, batch_size, activations, deltas, &neural_network_kernel_weights[0].gradient, activation_offset, delta_offset, kernel_offset, activation_size, delta_size, sizeof(struct TrainableParameters) / sizeof(void*), 1, 0);

	for (i = 0; i < number_of_neurons_in; i++)
	{
		for (j = 0; j < number_of_neurons_out; j++)
		{
			kernel_index = i * number_of_neurons_out + j + kernel_offset;

			neural_network_kernel_weights[kernel_index].gradient /= batch_size;
		} // end of j loop
	} // end of i loop
} // end of CalculateNeuralNetworkKernelWeightGradients

/* This function calculates the gradients of the neural network bias weights */
void CalculateNeuralNetworkBiasWeightGradients(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, struct TrainableParameters *neural_network_bias_weights, unsigned int delta_offset, unsigned int bias_offset)
{
	/* grad = sum(di, i, 0, n - 1) / m */

	unsigned int i, j, neuron_index, bias_index;

	for (i = 0; i < number_of_neurons; i++)
	{
		bias_index = i + bias_offset;
		neural_network_bias_weights[bias_index].gradient = 0.0;
		for (j = 0; j < batch_size; j++)
		{
			neuron_index = j * number_of_neurons + i + delta_offset;

			neural_network_bias_weights[bias_index].gradient += neural_network_layer_neurons[neuron_index].delta;
		} // end of j loop
		neural_network_bias_weights[bias_index].gradient /= batch_size;
	} // end of i loop
} // end of CalculateNeuralNetworkBiasWeightGradients function

/* This function calculates the gradients of the parametric ReLU alphas */
void CalculateParametricReLUGradients(unsigned int batch_size, unsigned int number_of_neurons, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int neuron_offset, struct TrainableParameters *parametric_relu_alphas, unsigned int parametric_relu_offset)
{
	/* grad = sum(zi, i, 0, n - 1) / m */

	unsigned int i, j, neuron_index, parametric_relu;

	for (i = 0; i < number_of_neurons; i++)
	{
		parametric_relu = i + parametric_relu_offset;
		parametric_relu_alphas[parametric_relu].gradient = 0.0;
		for (j = 0; j < batch_size; j++)
		{
			neuron_index = j * number_of_neurons + i + neuron_offset;
			parametric_relu_alphas[parametric_relu].gradient += neural_network_layer_neurons[neuron_index].weighted_sum;
		} // end of j loop
		parametric_relu_alphas[parametric_relu].gradient /= batch_size;
	} // end of i loop
} // end of CalculateParametricReLUGradients function

/* This function calculates the gradients of the batch normalization layer parameters */
void CalculateBatchNormalizationParameterGradients(unsigned int batch_norm_index, unsigned int batch_size, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, double delta_beta, double delta_gamma)
{
		batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_beta.gradient = delta_beta / batch_size;
		batch_normalization_layer_neurons_parameters[batch_norm_index].batch_norm_gamma.gradient = delta_gamma / batch_size;
} // end of CalculateBatchNormalizationParameterGradients function

/******************************************************************************************/
/******************************** REGULARIZATION GRADIENTS ********************************/
/******************************************************************************************/

/* This function calculates the gradient of L1 regularization loss of given kernel weight layer */
void AddKernelL1RegularizationGradient(unsigned int current_layer_number_of_neurons, unsigned int next_layer_number_of_neurons, struct TrainableParameters *neural_network_kernel_weights, unsigned int kernel_weights_offset, double lambda1, unsigned int batch_size)
{
	/*	if w_i < 0
	   		grad_i = -lambda1 / batch_size
	   	else
	   		grad_i = lambda1 / batch_size
	*/

	unsigned int i, j, kernel_index;
	double scale_constant = lambda1 / batch_size;

	for (i = 0; i < current_layer_number_of_neurons; i++)
	{
		for (j = 0; j < next_layer_number_of_neurons; j++)
		{
			kernel_index = i * next_layer_number_of_neurons + j + kernel_weights_offset;

			if (neural_network_kernel_weights[kernel_index].variable < 0.0)
			{
				neural_network_kernel_weights[kernel_index].gradient +=  -scale_constant;
			}
			else if (neural_network_kernel_weights[kernel_index].variable > 0.0)
			{
				neural_network_kernel_weights[kernel_index].gradient +=  scale_constant;
			}
		} // end of j loop
	} // end of i loop
} // end of AddKernelL1RegularizationGradient function

/* This function calculates the gradient of L2 regularization loss of given kernel weight layer */
void AddKernelL2RegularizationGradient(unsigned int current_layer_number_of_neurons, unsigned int next_layer_number_of_neurons, struct TrainableParameters *neural_network_kernel_weights, unsigned int kernel_weights_offset, double lambda2, unsigned int batch_size)
{
	/* grad_i = lambda2 / batch_size * w_i */

	unsigned int i, j, kernel_index;
	double scale_constant = lambda2 / batch_size;

	for (i = 0; i < current_layer_number_of_neurons; i++)
	{
		for (j = 0; j < next_layer_number_of_neurons; j++)
		{
			kernel_index = i * next_layer_number_of_neurons + j + kernel_weights_offset;

			neural_network_kernel_weights[kernel_index].gradient += scale_constant * neural_network_kernel_weights[kernel_index].variable;
		} // end of j loop
	} // end of i loop
} // end of AddKernelL2RegularizationGradient function

/* This function calculates the gradient of L1 regularization loss of given bias weight layer */
void AddBiasL1RegularizationGradient(unsigned int next_layer_number_of_neurons, struct TrainableParameters *neural_network_bias_weights, unsigned int bias_weights_offset, double lambda1, unsigned int batch_size)
{
	/*	if w_i < 0
	   		grad_i = -lambda1 / batch_size
	   	else
	   		grad_i = lambda1 / batch_size
	*/

	unsigned int i, bias_index;
	double scale_constant = lambda1 / batch_size;

	for (i = 0; i < next_layer_number_of_neurons; i++)
	{
		bias_index = i  + bias_weights_offset;

		if (neural_network_bias_weights[bias_index].variable < 0.0)
		{
			neural_network_bias_weights[bias_index].gradient +=  -scale_constant;
		}
		else if (neural_network_bias_weights[bias_index].variable > 0.0)
		{
			neural_network_bias_weights[bias_index].gradient +=  scale_constant;
		}
	} // end of i loop
} // end of AddBiasL1RegularizationGradient function

/* This function calculates the gradient of L2 regularization loss of given bias weight layer */
void AddBiasL2RegularizationGradient(unsigned int next_layer_number_of_neurons, struct TrainableParameters *neural_network_bias_weights, unsigned int bias_weights_offset, double lambda2, unsigned int batch_size)
{
	/* grad_i = lambda2 / batch_size * w_i */

	unsigned int i, bias_index;
	double scale_constant = lambda2 / batch_size;

	for (i = 0; i < next_layer_number_of_neurons; i++)
	{
		bias_index = i  + bias_weights_offset;

		neural_network_bias_weights[bias_index].gradient += scale_constant * neural_network_bias_weights[bias_index].variable;
	} // end of i loop
} // end of AddBiasL2RegularizationGradient function

/******************************************************************************************/
/*********************************** GRADIENT CLIPPING ************************************/
/******************************************************************************************/

/* This function clips gradients by their global norm */
void ApplyGradientClippingByGlobalNorm(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters)
{
	unsigned int i, j, k, kernel_index, bias_index;
	double global_norm, scale_factor;

	global_norm = CalculateGradientGlobalNorm(total_kernel_weights, neural_network_kernel_weights, total_bias_weights, neural_network_bias_weights, total_parametric_relu_neurons, parametric_relu_alphas, total_batch_normalization_neurons, batch_normalization_layer_neurons_parameters);

	printf("ApplyGradientClippingByGlobalNorm: global_norm = %.16f\n", global_norm);

	if (training_validation_hyperparameters->clip_norm < global_norm)
	{
		scale_factor = training_validation_hyperparameters->clip_norm / global_norm;

		/* Kernel */
		for (i = 0; i < total_kernel_weights; i++)
		{
			ApplyGradientGlobalNormScaleFactor(&neural_network_kernel_weights[i].gradient, scale_factor);
		} // end of i loop

		/* Bias */
		for (i = 0; i < total_bias_weights; i++)
		{
			ApplyGradientGlobalNormScaleFactor(&neural_network_bias_weights[i].gradient, scale_factor);
		} // end of i loop

		/* Parametric ReLU alphas */
		for (i = 0; i < total_parametric_relu_neurons; i++)
		{
			ApplyGradientGlobalNormScaleFactor(&parametric_relu_alphas[i].gradient, scale_factor);
		} // end of i loop


		/* Batch normalization */
		for (i = 0; i < total_batch_normalization_neurons; i++)
		{
			/* Beta */
			ApplyGradientGlobalNormScaleFactor(&batch_normalization_layer_neurons_parameters[i].batch_norm_beta.gradient, scale_factor);

			/* Gamma */
			ApplyGradientGlobalNormScaleFactor(&batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.gradient, scale_factor);
		} // end of i loop
	}
} // end of ApplyGradientClippingByGlobalNorm function

/* This function calculates the global norm of all gradients */
double CalculateGradientGlobalNorm(unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters)
{
	double global_norm = 0.0;

	/* Kernel */
	global_norm += CalculateSquaredGradientsSum(total_kernel_weights, &neural_network_kernel_weights[0].gradient, sizeof(struct TrainableParameters) / sizeof(void*));

	/* Bias */
	global_norm += CalculateSquaredGradientsSum(total_bias_weights, &neural_network_bias_weights[0].gradient, sizeof(struct TrainableParameters) / sizeof(void*));

	/* Parametric ReLU alphas */
	global_norm += CalculateSquaredGradientsSum(total_parametric_relu_neurons, &parametric_relu_alphas[0].gradient, sizeof(struct TrainableParameters) / sizeof(void*));

	/* Batch normalization */
	/* Beta */
	global_norm += CalculateSquaredGradientsSum(total_batch_normalization_neurons, &batch_normalization_layer_neurons_parameters[0].batch_norm_beta.gradient, sizeof(struct TrainableParameters) / sizeof(void*));

	/* Gamma */
	global_norm += CalculateSquaredGradientsSum(total_batch_normalization_neurons, &batch_normalization_layer_neurons_parameters[0].batch_norm_gamma.gradient, sizeof(struct TrainableParameters) / sizeof(void*));

	global_norm = sqrt(global_norm);

	return global_norm;
} // end of CalculateGradientGlobalNorm function

/* This function calculates the sum of a trainable parameter's squared gradients */
double CalculateSquaredGradientsSum(unsigned int total_gradients, double *gradient, unsigned long gradient_size)
{
	unsigned int i, gradient_index;
	double sum = 0.0;

	for (i = 0; i < total_gradients; i++)
	{
		gradient_index = i * gradient_size;

		sum += gradient[gradient_index] * gradient[gradient_index];
	} // end of i loop

	return sum;
} // end of CalculateSquaredGradientsSum function

/* This function applies the global norm scale factor to gradient */
void ApplyGradientGlobalNormScaleFactor(double *gradient, double scale_factor)
{
	(*gradient) *= scale_factor;
} // end of ApplyGradientGlobalNormScaleFactor function

/******************************************************************************************/
/********************************** GRADIENT OPTIMIZERS ***********************************/
/******************************************************************************************/

/* This function applies gradient optimizers to the currently calculated gradients */
void ApplyGradientOptimizers(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, unsigned int train_step)
{
	/* Optimizers */
	/*
	name				optimizer		parameter0						parameter1				parameter2						parameter3						parameter4
	vanilla SGD			0				NULL							NULL					NULL							NULL							NULL
	momentum			1				momentum						NULL					NULL							NULL							NULL
	nesterov momentum	2				momentum						NULL					NULL							NULL							NULL
	adagrad				3				initial_accumulator_value		NULL					NULL							NULL							NULL
	adadelta			4				rho								NULL					NULL							NULL							NULL
	rmsprop				5				decay							momentum				centered						NULL							NULL
	adam				6				beta1							beta2					NULL							NULL							NULL
	adamax				7				beta1							beta2					NULL							NULL							NULL
	nadam				8				beta1							beta2					NULL							NULL							NULL
	amsgrad				9				beta1							beta2					NULL							NULL							NULL
	ftrl				10				initial_accumulator_value		learning_rate_power		l1_regularization_strength		l2_regularization_strength		l2_shrinkage_regularization_strength
	 */

	printf("ApplyGradientOptimizers: optimizer = %d\n", training_validation_hyperparameters->optimizer);

	switch (training_validation_hyperparameters->optimizer)
	{
		case 0:
			ApplyOptimizerVanillaSGD(training_validation_hyperparameters, total_kernel_weights, neural_network_kernel_weights, total_bias_weights, neural_network_bias_weights, total_parametric_relu_neurons, parametric_relu_alphas, total_batch_normalization_neurons, batch_normalization_layer_neurons_parameters);
			break;
		case 1:
			ApplyOptimizerMomentum(training_validation_hyperparameters, total_kernel_weights, neural_network_kernel_weights, total_bias_weights, neural_network_bias_weights, total_parametric_relu_neurons, parametric_relu_alphas, total_batch_normalization_neurons, batch_normalization_layer_neurons_parameters);
			break;
		case 2:
			ApplyOptimizerNesterovMomentum(training_validation_hyperparameters, total_kernel_weights, neural_network_kernel_weights, total_bias_weights, neural_network_bias_weights, total_parametric_relu_neurons, parametric_relu_alphas, total_batch_normalization_neurons, batch_normalization_layer_neurons_parameters, train_step);
			break;
		case 3:
			ApplyOptimizerAdagrad(training_validation_hyperparameters, total_kernel_weights, neural_network_kernel_weights, total_bias_weights, neural_network_bias_weights, total_parametric_relu_neurons, parametric_relu_alphas, total_batch_normalization_neurons, batch_normalization_layer_neurons_parameters);
			break;
		case 4:
			ApplyOptimizerAdadelta(training_validation_hyperparameters, total_kernel_weights, neural_network_kernel_weights, total_bias_weights, neural_network_bias_weights, total_parametric_relu_neurons, parametric_relu_alphas, total_batch_normalization_neurons, batch_normalization_layer_neurons_parameters);
			break;
		case 5:
			ApplyOptimizerRMSProp(training_validation_hyperparameters, total_kernel_weights, neural_network_kernel_weights, total_bias_weights, neural_network_bias_weights, total_parametric_relu_neurons, parametric_relu_alphas, total_batch_normalization_neurons, batch_normalization_layer_neurons_parameters);
			break;
		case 6:
			ApplyOptimizerAdam(training_validation_hyperparameters, total_kernel_weights, neural_network_kernel_weights, total_bias_weights, neural_network_bias_weights, total_parametric_relu_neurons, parametric_relu_alphas, total_batch_normalization_neurons, batch_normalization_layer_neurons_parameters, train_step);
			break;
		case 7:
			ApplyOptimizerAdamax(training_validation_hyperparameters, total_kernel_weights, neural_network_kernel_weights, total_bias_weights, neural_network_bias_weights, total_parametric_relu_neurons, parametric_relu_alphas, total_batch_normalization_neurons, batch_normalization_layer_neurons_parameters, train_step);
			break;
		case 8:
			ApplyOptimizerNadam(training_validation_hyperparameters, total_kernel_weights, neural_network_kernel_weights, total_bias_weights, neural_network_bias_weights, total_parametric_relu_neurons, parametric_relu_alphas, total_batch_normalization_neurons, batch_normalization_layer_neurons_parameters, train_step);
			break;
		case 9:
			ApplyOptimizerAMSGrad(training_validation_hyperparameters, total_kernel_weights, neural_network_kernel_weights, total_bias_weights, neural_network_bias_weights, total_parametric_relu_neurons, parametric_relu_alphas, total_batch_normalization_neurons, batch_normalization_layer_neurons_parameters, train_step);
			break;
		case 10:
			ApplyOptimizerFTRL(training_validation_hyperparameters, total_kernel_weights, neural_network_kernel_weights, total_bias_weights, neural_network_bias_weights, total_parametric_relu_neurons, parametric_relu_alphas, total_batch_normalization_neurons, batch_normalization_layer_neurons_parameters);
			break;
		default:
			/* Unrecognized, do vanilla SGD instead */
			ApplyOptimizerVanillaSGD(training_validation_hyperparameters, total_kernel_weights, neural_network_kernel_weights, total_bias_weights, neural_network_bias_weights, total_parametric_relu_neurons, parametric_relu_alphas, total_batch_normalization_neurons, batch_normalization_layer_neurons_parameters);
			printf("ApplyGradientOptimizers: No optimizer to apply\n");
	}
} // end of ApplyGradientOptimizers function

/* This function applies the vanilla gradient descent optimizer to the currently calculated gradients */
void ApplyOptimizerVanillaSGD(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters)
{
	/*
	name				optimizer		parameter0						parameter1				parameter2						parameter3						parameter4
	vanilla SGD			0				NULL							NULL					NULL							NULL							NULL
	 */

	unsigned int i;

	/* Kernel weights */
	for (i = 0; i < total_kernel_weights; i++)
	{
		VanillaSGDOptimizer(neural_network_kernel_weights[i].gradient, &neural_network_kernel_weights[i].update, training_validation_hyperparameters->learning_rate);
	} // end of i loop

	/* Bias weights */
	for (i = 0; i < total_bias_weights; i++)
	{
		VanillaSGDOptimizer(neural_network_bias_weights[i].gradient, &neural_network_bias_weights[i].update, training_validation_hyperparameters->learning_rate);
	} // end of i loop */

	/* Parametric ReLU alphas  */
	for (i = 0; i < total_parametric_relu_neurons; i++)
	{
		VanillaSGDOptimizer(parametric_relu_alphas[i].gradient, &parametric_relu_alphas[i].update, training_validation_hyperparameters->learning_rate);
	} // end of i loop

	/* Batch normalization */
	for (i = 0; i < total_batch_normalization_neurons; i++)
	{
		/* Beta */
		VanillaSGDOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_beta.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.update, training_validation_hyperparameters->learning_rate);

		/* Gamma */
		VanillaSGDOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.update, training_validation_hyperparameters->learning_rate);
	} // end of i loop
} // end of ApplyOptimizerVanillaSGD function

/* This function is the vanilla sgd optimizer */
void VanillaSGDOptimizer(double gradient, double *update, double learning_rate)
{
	/* u = -lr * g */
	(*update) = -learning_rate * gradient;
} // end of VanillaSGDOptimizer function

/* This function applies the momentum optimizer to the currently calculated gradients */
void ApplyOptimizerMomentum(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters)
{
	/*
	name				optimizer		parameter0						parameter1				parameter2						parameter3						parameter4
	momentum			1				momentum						NULL					NULL							NULL							NULL
	 */

	unsigned int i;

	/* Kernel weights */
	for (i = 0; i < total_kernel_weights; i++)
	{
		MomentumOptimizer(neural_network_kernel_weights[i].gradient, &neural_network_kernel_weights[i].update, training_validation_hyperparameters->learning_rate, training_validation_hyperparameters->optimizer_parameter0);
	} // end of i loop

	/* Bias weights */
	for (i = 0; i < total_bias_weights; i++)
	{
		MomentumOptimizer(neural_network_bias_weights[i].gradient, &neural_network_bias_weights[i].update, training_validation_hyperparameters->learning_rate, training_validation_hyperparameters->optimizer_parameter0);
	} // end of i loop

	/* Parametric ReLU alphas */
	for (i = 0; i < total_parametric_relu_neurons; i++)
	{
		MomentumOptimizer(parametric_relu_alphas[i].gradient, &parametric_relu_alphas[i].update, training_validation_hyperparameters->learning_rate, training_validation_hyperparameters->optimizer_parameter0);
	} // end of i loop

	/* Batch normalization */
	for (i = 0; i < total_batch_normalization_neurons; i++)
	{
		/* Beta */
		MomentumOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_beta.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.update, training_validation_hyperparameters->learning_rate, training_validation_hyperparameters->optimizer_parameter0);

		/* Gamma */
		MomentumOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.update, training_validation_hyperparameters->learning_rate, training_validation_hyperparameters->optimizer_parameter0);
	} // end of i loop
} // end of ApplyOptimizerMomentum function

/* This function is the momentum optimizer */
void MomentumOptimizer(double gradient, double *update, double learning_rate, double momentum)
{
	/* u = mom * u - lr * g */
	(*update) = momentum * (*update) - learning_rate * gradient;
} // end of MomentumOptimizer function

/* This function applies the nesterov momentum optimizer to the currently calculated gradients */
void ApplyOptimizerNesterovMomentum(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, unsigned int train_step)
{
	/*
	name				optimizer		parameter0						parameter1				parameter2						parameter3						parameter4
	nesterov momentum	2				momentum						NULL					NULL							NULL							NULL
	 */

	unsigned int i;

	/* Kernel weights */
	for (i = 0; i < total_kernel_weights; i++)
	{
		NesterovMomentumOptimizer(neural_network_kernel_weights[i].variable, neural_network_kernel_weights[i].gradient, &neural_network_kernel_weights[i].aggregate0, &neural_network_kernel_weights[i].update, training_validation_hyperparameters->learning_rate, training_validation_hyperparameters->optimizer_parameter0, train_step);
	} // end of i loop

	/* Bias weights */
	for (i = 0; i < total_bias_weights; i++)
	{
		NesterovMomentumOptimizer(neural_network_bias_weights[i].variable, neural_network_bias_weights[i].gradient, &neural_network_bias_weights[i].aggregate0, &neural_network_bias_weights[i].update, training_validation_hyperparameters->learning_rate, training_validation_hyperparameters->optimizer_parameter0, train_step);
	} // end of i loop

	/* Parametric ReLU alphas */
	for (i = 0; i < total_parametric_relu_neurons; i++)
	{
		NesterovMomentumOptimizer(parametric_relu_alphas[i].variable, parametric_relu_alphas[i].gradient, &parametric_relu_alphas[i].aggregate0, &parametric_relu_alphas[i].update, training_validation_hyperparameters->learning_rate, training_validation_hyperparameters->optimizer_parameter0, train_step);
	} // end of i loop

	/* Batch normalization */
	for (i = 0; i < total_batch_normalization_neurons; i++)
	{
		/* Beta */
		NesterovMomentumOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_beta.variable, batch_normalization_layer_neurons_parameters[i].batch_norm_beta.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.aggregate0, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.update, training_validation_hyperparameters->learning_rate, training_validation_hyperparameters->optimizer_parameter0, train_step);

		/* Gamma */
		NesterovMomentumOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.variable, batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.aggregate0, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.update, training_validation_hyperparameters->learning_rate, training_validation_hyperparameters->optimizer_parameter0, train_step);
	} // end of i loop
} // end of ApplyOptimizerNesterovMomentum function

/* This function is the nesterov momentum optimizer */
void NesterovMomentumOptimizer(double variable, double gradient, double *m_moment_aggregate, double *update, double learning_rate, double momentum, unsigned int train_step)
{
	double gradient_descent_step;

	if (train_step == 0)
	{
		/* Effectively momentum equals zero in this branch (to save on needless multadds) */

		/* gd = -lr * g */
		gradient_descent_step = -learning_rate * gradient;

		/* u = gd */
		(*update) = gradient_descent_step;

		/* m = w + gd = w - lr * g */
		(*m_moment_aggregate) = variable + gradient_descent_step;
	}
	else
	{
		/* gd = -lr * g */
		gradient_descent_step = -learning_rate * gradient;

		/* u = mom * (w + gd - m) + gd */
		(*update) = momentum * (variable + gradient_descent_step - (*m_moment_aggregate)) + gradient_descent_step;

		/* m = w + gd = w - lr * g */
		(*m_moment_aggregate) = variable + gradient_descent_step;
	}
} // end of MomentumOptimizer function

/* This function applies the adagrad optimizer to the currently calculated gradients */
void ApplyOptimizerAdagrad(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters)
{
	/*
	name				optimizer		parameter0						parameter1				parameter2						parameter3						parameter4
	adagrad				3				initial_accumulator_value		NULL					NULL							NULL							NULL
	 */

	unsigned int i;

	/* Kernel weights */
	for (i = 0; i < total_kernel_weights; i++)
	{
		AdagradOptimizer(neural_network_kernel_weights[i].gradient, &neural_network_kernel_weights[i].aggregate1, &neural_network_kernel_weights[i].update, training_validation_hyperparameters->learning_rate);
	} // end of i loop

	/* Bias weights */
	for (i = 0; i < total_bias_weights; i++)
	{
		AdagradOptimizer(neural_network_bias_weights[i].gradient, &neural_network_bias_weights[i].aggregate1, &neural_network_bias_weights[i].update, training_validation_hyperparameters->learning_rate);
	} // end of i loop

	/* Parametric ReLU alphas */
	for (i = 0; i < total_parametric_relu_neurons; i++)
	{
		AdagradOptimizer(parametric_relu_alphas[i].gradient, &parametric_relu_alphas[i].aggregate1, &parametric_relu_alphas[i].update, training_validation_hyperparameters->learning_rate);
	} // end of i loop

	/* Batch normalization */
	for (i = 0; i < total_batch_normalization_neurons; i++)
	{
		/* Beta */
		AdagradOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_beta.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.aggregate1, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.update, training_validation_hyperparameters->learning_rate);

		/* Gamma */
		AdagradOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.aggregate1, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.update, training_validation_hyperparameters->learning_rate);
	} // end of i loop
} // end of ApplyOptimizerAdagrad function

/* This function is the adagrad optimizer */
void AdagradOptimizer(double gradient, double *accumulator_aggregate, double *update, double learning_rate)
{
	/* a = a + g * g */
	(*accumulator_aggregate) += gradient * gradient;

	/* u = -lr * g / (sqrt(a) + eps) */
	(*update) = -learning_rate * gradient / (sqrt((*accumulator_aggregate)) + epsilon);
} // end of AdagradOptimizer function

/* This function applies the adadelta optimizer to the currently calculated gradients */
void ApplyOptimizerAdadelta(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters)
{
	/*
	name				optimizer		parameter0						parameter1				parameter2						parameter3						parameter4
	adadelta			4				rho								NULL					NULL							NULL							NULL
	 */

	unsigned int i;

	/* Kernel weights */
	for (i = 0; i < total_kernel_weights; i++)
	{
		AdadeltaOptimizer(neural_network_kernel_weights[i].gradient, &neural_network_kernel_weights[i].aggregate0, &neural_network_kernel_weights[i].aggregate1, &neural_network_kernel_weights[i].update, training_validation_hyperparameters->optimizer_parameter0);
	} // end of i loop

	/* Bias weights */
	for (i = 0; i < total_bias_weights; i++)
	{
		AdadeltaOptimizer(neural_network_bias_weights[i].gradient, &neural_network_bias_weights[i].aggregate0, &neural_network_bias_weights[i].aggregate1, &neural_network_bias_weights[i].update, training_validation_hyperparameters->optimizer_parameter0);
	} // end of i loop

	/* Parametric ReLU alphas */
	for (i = 0; i < total_parametric_relu_neurons; i++)
	{
		AdadeltaOptimizer(parametric_relu_alphas[i].gradient, &parametric_relu_alphas[i].aggregate0, &parametric_relu_alphas[i].aggregate1, &parametric_relu_alphas[i].update, training_validation_hyperparameters->optimizer_parameter0);
	} // end of i loop

	/* Batch normalization */
	for (i = 0; i < total_batch_normalization_neurons; i++)
	{
		/* Beta */
		AdadeltaOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_beta.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.aggregate0, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.aggregate1, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.update, training_validation_hyperparameters->optimizer_parameter0);

		/* Gamma */
		AdadeltaOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.aggregate0, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.aggregate1, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.update, training_validation_hyperparameters->optimizer_parameter0);
	} // end of i loop
} // end of ApplyOptimizerAdadelta function

/* This function is the adadelta optimizer */
void AdadeltaOptimizer(double gradient, double *accumulator_aggregate, double *delta_accumulator_aggregate, double *update, double rho)
{
	/* a = rho * a + (1 - rho) * g * g */
	(*accumulator_aggregate) = rho * (*accumulator_aggregate) + (1.0 - rho) * gradient * gradient;

	/* u = -sqrt(da + eps) / sqrt(a + eps) * g */
	(*update) = -sqrt((*delta_accumulator_aggregate) + epsilon) / sqrt((*accumulator_aggregate) + epsilon) * gradient;

	/* da = rho * da + (1 - rho) * u * u */
	(*delta_accumulator_aggregate) = rho * (*delta_accumulator_aggregate) + (1.0 - rho) * (*update) * (*update);
} // end of AdadeltaOptimizer function

/* This function applies the rmsprop optimizer to the currently calculated gradients */
void ApplyOptimizerRMSProp(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters)
{
	/*
	name				optimizer		parameter0						parameter1				parameter2						parameter3						parameter4
	rmsprop				5				decay							momentum				centered						NULL							NULL
	 */

	unsigned int i;

	/* Kernel weights */
	for (i = 0; i < total_kernel_weights; i++)
	{
		RMSPropOptimizer(neural_network_kernel_weights[i].gradient, &neural_network_kernel_weights[i].aggregate0, &neural_network_kernel_weights[i].aggregate1, &neural_network_kernel_weights[i].update, training_validation_hyperparameters->learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1, (int)training_validation_hyperparameters->optimizer_parameter2);
	} // end of i loop

	/* Bias weights */
	for (i = 0; i < total_bias_weights; i++)
	{
		RMSPropOptimizer(neural_network_bias_weights[i].gradient, &neural_network_bias_weights[i].aggregate0, &neural_network_bias_weights[i].aggregate1, &neural_network_bias_weights[i].update, training_validation_hyperparameters->learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1, (int)training_validation_hyperparameters->optimizer_parameter2);
	} // end of i loop

	/* Parametric ReLU alphas */
	for (i = 0; i < total_parametric_relu_neurons; i++)
	{
		RMSPropOptimizer(parametric_relu_alphas[i].gradient, &parametric_relu_alphas[i].aggregate0, &parametric_relu_alphas[i].aggregate1, &parametric_relu_alphas[i].update, training_validation_hyperparameters->learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1, (int)training_validation_hyperparameters->optimizer_parameter2);
	} // end of i loop

	/* Batch normalization */
	for (i = 0; i < total_batch_normalization_neurons; i++)
	{
		/* Beta */
		RMSPropOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_beta.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.aggregate0, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.aggregate1, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.update, training_validation_hyperparameters->learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1, (int)training_validation_hyperparameters->optimizer_parameter2);

		/* Gamma */
		RMSPropOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.aggregate0, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.aggregate1, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.update, training_validation_hyperparameters->learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1, (int)training_validation_hyperparameters->optimizer_parameter2);
	} // end of i loop
} // end of ApplyOptimizerRMSProp function

/* This function is the rmsprop optimizer */
void RMSPropOptimizer(double gradient, double *m_moment_aggregate, double *accumulator_aggregate, double *update, double learning_rate, double decay, double momentum, int centered)
{
	if (centered == 0)
	{
		/* a = rho * a + (1 - rho) * g * g */
		(*accumulator_aggregate) = decay * (*accumulator_aggregate) + (1.0 - decay) * gradient * gradient;

		/* u = -(mom * u + lr * g / sqrt(a + eps)) */
		(*update) = -(momentum * (*update) + learning_rate * gradient / sqrt((*accumulator_aggregate) + epsilon));
	}
	else
	{
		/* m = rho * m + (1 - rho) * g */
		(*m_moment_aggregate) = decay * (*m_moment_aggregate) + (1.0 - decay) * gradient;

		/* a = rho * a + (1 - rho) * g * g */
		(*accumulator_aggregate) = decay * (*accumulator_aggregate) + (1.0 - decay) * gradient * gradient;

		/* u = -(mom * u + lr * g / sqrt(a - m * m + eps)) */
		(*update) = -(momentum * (*update) + learning_rate * gradient / sqrt((*accumulator_aggregate) - (*m_moment_aggregate) * (*m_moment_aggregate) + epsilon));
	}
} // end of RMSPropOptimizer function

/* This function applies the adam optimizer to the currently calculated gradients */
void ApplyOptimizerAdam(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, unsigned int train_step)
{
	/*
	name				optimizer		parameter0						parameter1				parameter2						parameter3						parameter4
	adam				6				beta1							beta2					NULL							NULL							NULL
	 */

	unsigned int i;

	/* elr = initial_learning_rate * sqrt(1 - beta2^(t + 1)) / (1 - beta1^(t + 1)) */
	double effective_learning_rate = training_validation_hyperparameters->learning_rate * sqrt(1.0 - pow(training_validation_hyperparameters->optimizer_parameter1, train_step + 1)) / (1.0 - pow(training_validation_hyperparameters->optimizer_parameter0, train_step + 1));

	/* Kernel weights */
	for (i = 0; i < total_kernel_weights; i++)
	{
		AdamOptimizer(neural_network_kernel_weights[i].gradient, &neural_network_kernel_weights[i].aggregate0, &neural_network_kernel_weights[i].aggregate1, &neural_network_kernel_weights[i].update, effective_learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1);
	} // end of i loop

	/* Bias weights */
	for (i = 0; i < total_bias_weights; i++)
	{
		AdamOptimizer(neural_network_bias_weights[i].gradient, &neural_network_bias_weights[i].aggregate0, &neural_network_bias_weights[i].aggregate1, &neural_network_bias_weights[i].update, effective_learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1);
	} // end of i loop

	/* Parametric ReLU alphas */
	for (i = 0; i < total_parametric_relu_neurons; i++)
	{
		AdamOptimizer(parametric_relu_alphas[i].gradient, &parametric_relu_alphas[i].aggregate0, &parametric_relu_alphas[i].aggregate1, &parametric_relu_alphas[i].update, effective_learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1);
	} // end of i loop

	/* Batch normalization */
	for (i = 0; i < total_batch_normalization_neurons; i++)
	{
		/* Beta */
		AdamOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_beta.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.aggregate0, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.aggregate1, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.update, effective_learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1);

		/* Gamma */
		AdamOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.aggregate0, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.aggregate1, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.update, effective_learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1);
	} // end of i loop
} // end of ApplyOptimizerAdam function

/* This function is the adam optimizer */
void AdamOptimizer(double gradient, double *m_moment_aggregate, double *v_moment_aggregate, double *update, double effective_learning_rate, double beta1, double beta2)
{
	/* m = beta1 * m + (1 - beta1) * g */
	(*m_moment_aggregate) = beta1 * (*m_moment_aggregate) + (1.0 - beta1) * gradient;

	/* v = beta2 * v + (1 - beta2) * g * g */
	(*v_moment_aggregate) = beta2 * (*v_moment_aggregate) + (1.0 - beta2) * gradient * gradient;

	/* u = -elr * m / (sqrt(v) + eps) */
	(*update) = -effective_learning_rate * (*m_moment_aggregate) / (sqrt((*v_moment_aggregate)) + epsilon);
} // end of AdamOptimizer function

/* This function applies the adamax optimizer to the currently calculated gradients */
void ApplyOptimizerAdamax(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, unsigned int train_step)
{
	/*
	name				optimizer		parameter0						parameter1				parameter2						parameter3						parameter4
	adamax				7				beta1							beta2					NULL							NULL							NULL
	 */

	unsigned int i;

	/* elr = initial_learning_rate / (1 - beta1^(t + 1)) */
	double effective_learning_rate = training_validation_hyperparameters->learning_rate / (1.0 - pow(training_validation_hyperparameters->optimizer_parameter0, train_step + 1));

	/* Kernel weights */
	for (i = 0; i < total_kernel_weights; i++)
	{
		AdamaxOptimizer(neural_network_kernel_weights[i].gradient, &neural_network_kernel_weights[i].aggregate0, &neural_network_kernel_weights[i].aggregate1, &neural_network_kernel_weights[i].update, effective_learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1);
	} // end of i loop

	/* Bias weights */
	for (i = 0; i < total_bias_weights; i++)
	{
		AdamaxOptimizer(neural_network_bias_weights[i].gradient, &neural_network_bias_weights[i].aggregate0, &neural_network_bias_weights[i].aggregate1, &neural_network_bias_weights[i].update, effective_learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1);
	} // end of i loop

	/* Parametric ReLU alphas */
	for (i = 0; i < total_parametric_relu_neurons; i++)
	{
		AdamaxOptimizer(parametric_relu_alphas[i].gradient, &parametric_relu_alphas[i].aggregate0, &parametric_relu_alphas[i].aggregate1, &parametric_relu_alphas[i].update, effective_learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1);
	} // end of i loop

	/* Batch normalization */
	for (i = 0; i < total_batch_normalization_neurons; i++)
	{
		/* Beta */
		AdamaxOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_beta.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.aggregate0, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.aggregate1, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.update, effective_learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1);

		/* Gamma */
		AdamaxOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.aggregate0, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.aggregate1, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.update, effective_learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1);
	} // end of i loop
} // end of ApplyOptimizerAdamax function

/* This function is the adamax optimizer */
void AdamaxOptimizer(double gradient, double *m_moment_aggregate, double *v_moment_aggregate, double *update, double effective_learning_rate, double beta1, double beta2)
{
	/* m = beta1 * m + (1 - beta1) * g */
	(*m_moment_aggregate) = beta1 * (*m_moment_aggregate) + (1.0 - beta1) * gradient;

	/* v = max(beta2 * v, abs(g)) */
	(*v_moment_aggregate) = fmax(beta2 * (*v_moment_aggregate), fabs(gradient));

	/* u = -elr * m / (sqrt(v) + eps) */
	(*update) = -effective_learning_rate * (*m_moment_aggregate) / (sqrt((*v_moment_aggregate)) + epsilon);
} // end of AdamaxOptimizer function

/* This function applies the nadam optimizer to the currently calculated gradients */
void ApplyOptimizerNadam(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, unsigned int train_step)
{
	/*
	name				optimizer		parameter0						parameter1				parameter2						parameter3						parameter4
	nadam				8				beta1							beta2					NULL							NULL							NULL
	 */

	unsigned int i;

	/* elr = initial_learning_rate * sqrt(1 - beta2^(t + 1)) / (1 - beta1^(t + 1)) */
	double effective_learning_rate = training_validation_hyperparameters->learning_rate * sqrt(1.0 - pow(training_validation_hyperparameters->optimizer_parameter1, train_step + 1)) / (1.0 - pow(training_validation_hyperparameters->optimizer_parameter0, train_step + 1));

	/* Kernel weights */
	for (i = 0; i < total_kernel_weights; i++)
	{
		NadamOptimizer(neural_network_kernel_weights[i].gradient, &neural_network_kernel_weights[i].aggregate0, &neural_network_kernel_weights[i].aggregate1, &neural_network_kernel_weights[i].update, effective_learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1);
	} // end of i loop

	/* Bias weights */
	for (i = 0; i < total_bias_weights; i++)
	{
		NadamOptimizer(neural_network_bias_weights[i].gradient, &neural_network_bias_weights[i].aggregate0, &neural_network_bias_weights[i].aggregate1, &neural_network_bias_weights[i].update, effective_learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1);
	} // end of i loop

	/* Parametric ReLU alphas */
	for (i = 0; i < total_parametric_relu_neurons; i++)
	{
		NadamOptimizer(parametric_relu_alphas[i].gradient, &parametric_relu_alphas[i].aggregate0, &parametric_relu_alphas[i].aggregate1, &parametric_relu_alphas[i].update, effective_learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1);
	} // end of i loop

	/* Batch normalization */
	for (i = 0; i < total_batch_normalization_neurons; i++)
	{
		/* Beta */
		NadamOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_beta.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.aggregate0, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.aggregate1, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.update, effective_learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1);

		/* Gamma */
		NadamOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.aggregate0, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.aggregate1, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.update, effective_learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1);
	} // end of i loop
} // end of ApplyOptimizerNadam function

/* This function is the nadam optimizer */
void NadamOptimizer(double gradient, double *m_moment_aggregate, double *v_moment_aggregate, double *update, double effective_learning_rate, double beta1, double beta2)
{
	/* m = beta1 * m + (1 - beta1) * g */
	(*m_moment_aggregate) = beta1 * (*m_moment_aggregate) + (1.0 - beta1) * gradient;

	/* v = beta2 * v + (1 - beta2) * g * g */
	(*v_moment_aggregate) = beta2 * (*v_moment_aggregate) + (1.0 - beta2) * gradient * gradient;

	/* u = -elr * (beta1 * m + (1 - beta1) * g) / (sqrt(v) + eps) */
	(*update) = -effective_learning_rate * (beta1 * (*m_moment_aggregate) + (1.0 - beta1) * gradient) / (sqrt((*v_moment_aggregate)) + epsilon);
} // end of NadamOptimizer function

/* This function applies the amsgrad optimizer to the currently calculated gradients */
void ApplyOptimizerAMSGrad(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, unsigned int train_step)
{
	/*
	name				optimizer		parameter0						parameter1				parameter2						parameter3						parameter4
	amsgrad				9				beta1							beta2					NULL							NULL							NULL
	 */

	unsigned int i;

	/* elr = initial_learning_rate * sqrt(1 - beta2^(t + 1)) / (1 - beta1^(t + 1)) */
	double effective_learning_rate = training_validation_hyperparameters->learning_rate * sqrt(1.0 - pow(training_validation_hyperparameters->optimizer_parameter1, train_step + 1)) / (1.0 - pow(training_validation_hyperparameters->optimizer_parameter0, train_step + 1));

	/* Kernel weights */
	for (i = 0; i < total_kernel_weights; i++)
	{
		AMSGradOptimizer(neural_network_kernel_weights[i].gradient, &neural_network_kernel_weights[i].aggregate0, &neural_network_kernel_weights[i].aggregate1, &neural_network_kernel_weights[i].aggregate2, &neural_network_kernel_weights[i].update, effective_learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1);
	} // end of i loop

	/* Bias weights */
	for (i = 0; i < total_bias_weights; i++)
	{
		AMSGradOptimizer(neural_network_bias_weights[i].gradient, &neural_network_bias_weights[i].aggregate0, &neural_network_bias_weights[i].aggregate1, &neural_network_bias_weights[i].aggregate2, &neural_network_bias_weights[i].update, effective_learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1);
	} // end of i loop

	/* Parametric ReLU alphas */
	for (i = 0; i < total_parametric_relu_neurons; i++)
	{
		AMSGradOptimizer(parametric_relu_alphas[i].gradient, &parametric_relu_alphas[i].aggregate0, &parametric_relu_alphas[i].aggregate1, &parametric_relu_alphas[i].aggregate2, &parametric_relu_alphas[i].update, effective_learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1);
	} // end of i loop

	/* Batch normalization */
	for (i = 0; i < total_batch_normalization_neurons; i++)
	{
		/* Beta */
		AMSGradOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_beta.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.aggregate0, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.aggregate1, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.aggregate2, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.update, effective_learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1);

		/* Gamma */
		AMSGradOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.aggregate0, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.aggregate1, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.aggregate2, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.update, effective_learning_rate, training_validation_hyperparameters->optimizer_parameter0, training_validation_hyperparameters->optimizer_parameter1);
	} // end of i loop
} // end of ApplyOptimizerAMSGrad function

/* This function is the amsgrad optimizer */
void AMSGradOptimizer(double gradient, double *m_moment_aggregate, double *v_moment_aggregate, double *v_moment_hat_aggregate, double *update, double effective_learning_rate, double beta1, double beta2)
{
	/* m = beta1 * m + (1 - beta1) * g */
	(*m_moment_aggregate) = beta1 * (*m_moment_aggregate) + (1.0 - beta1) * gradient;

	/* v = beta2 * v + (1 - beta2) * g * g */
	(*v_moment_aggregate) = beta2 * (*v_moment_aggregate) + (1.0 - beta2) * gradient * gradient;

	/* vhat = max(vhat, v) */
	(*v_moment_hat_aggregate) = fmax((*v_moment_hat_aggregate), (*v_moment_aggregate));

	/* u = -elr * m / (sqrt(vhat) + eps) */
	(*update) = -effective_learning_rate * (*m_moment_aggregate) / (sqrt((*v_moment_hat_aggregate)) + epsilon);
} // end of AMSGradOptimizer function

/* This function applies the ftrl optimizer to the currently calculated gradients */
void ApplyOptimizerFTRL(struct TrainingValidationHyperparameters *training_validation_hyperparameters, unsigned int total_kernel_weights, struct TrainableParameters *neural_network_kernel_weights, unsigned int total_bias_weights, struct TrainableParameters *neural_network_bias_weights, unsigned int total_parametric_relu_neurons, struct TrainableParameters *parametric_relu_alphas, unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters)
{
	/*
	name				optimizer		parameter0						parameter1				parameter2						parameter3						parameter4
	ftrl				10				initial_accumulator_value		learning_rate_power		l1_regularization_strength		l2_regularization_strength		l2_shrinkage_regularization_strength
	 */

	unsigned int i;

	/* Kernel weights */
	for (i = 0; i < total_kernel_weights; i++)
	{
		FTRLOptimizer(neural_network_kernel_weights[i].variable, neural_network_kernel_weights[i].gradient, &neural_network_kernel_weights[i].aggregate0, &neural_network_kernel_weights[i].aggregate1, &neural_network_kernel_weights[i].aggregate2, &neural_network_kernel_weights[i].update, training_validation_hyperparameters->learning_rate, training_validation_hyperparameters->optimizer_parameter1, training_validation_hyperparameters->optimizer_parameter2, training_validation_hyperparameters->optimizer_parameter3, training_validation_hyperparameters->optimizer_parameter4);
	} // end of i loop

	/* Bias weights */
	for (i = 0; i < total_bias_weights; i++)
	{
		FTRLOptimizer(neural_network_bias_weights[i].variable, neural_network_bias_weights[i].gradient, &neural_network_bias_weights[i].aggregate0, &neural_network_bias_weights[i].aggregate1, &neural_network_bias_weights[i].aggregate2, &neural_network_bias_weights[i].update, training_validation_hyperparameters->learning_rate, training_validation_hyperparameters->optimizer_parameter1, training_validation_hyperparameters->optimizer_parameter2, training_validation_hyperparameters->optimizer_parameter3, training_validation_hyperparameters->optimizer_parameter4);
	} // end of i loop

	/* Parametric ReLU alphas */
	for (i = 0; i < total_parametric_relu_neurons; i++)
	{
		FTRLOptimizer(parametric_relu_alphas[i].variable, parametric_relu_alphas[i].gradient, &parametric_relu_alphas[i].aggregate0, &parametric_relu_alphas[i].aggregate1, &parametric_relu_alphas[i].aggregate2, &parametric_relu_alphas[i].update, training_validation_hyperparameters->learning_rate, training_validation_hyperparameters->optimizer_parameter1, training_validation_hyperparameters->optimizer_parameter2, training_validation_hyperparameters->optimizer_parameter3, training_validation_hyperparameters->optimizer_parameter4);
	} // end of i loop

	/* Batch normalization */
	for (i = 0; i < total_batch_normalization_neurons; i++)
	{
		/* Beta */
		FTRLOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_beta.variable, batch_normalization_layer_neurons_parameters[i].batch_norm_beta.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.aggregate0, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.aggregate1, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.aggregate2, &batch_normalization_layer_neurons_parameters[i].batch_norm_beta.update, training_validation_hyperparameters->learning_rate, training_validation_hyperparameters->optimizer_parameter1, training_validation_hyperparameters->optimizer_parameter2, training_validation_hyperparameters->optimizer_parameter3, training_validation_hyperparameters->optimizer_parameter4);

		/* Gamma */
		FTRLOptimizer(batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.variable, batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.gradient, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.aggregate0, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.aggregate1, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.aggregate2, &batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.update, training_validation_hyperparameters->learning_rate, training_validation_hyperparameters->optimizer_parameter1, training_validation_hyperparameters->optimizer_parameter2, training_validation_hyperparameters->optimizer_parameter3, training_validation_hyperparameters->optimizer_parameter4);
	} // end of i loop
} // end of ApplyOptimizerFTRL function

/* This function is the ftrl optimizer */
void FTRLOptimizer(double variable, double gradient, double *accumulator_aggregate, double *accumulator_new_aggregate, double *linear_aggregate, double *update, double learning_rate, double learning_rate_power, double l1_regularization_strength, double l2_regularization_strength, double l2_shrinkage_regularization_strength)
{
	double quadratic, grad_with_shrinkage;

	if (l2_shrinkage_regularization_strength == 0.0)
	{
		/* accum_new = accum + g * g */
		(*accumulator_new_aggregate) = (*accumulator_aggregate) + gradient * gradient;

		/* linear += g + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var */
		(*linear_aggregate) += gradient + (pow((*accumulator_new_aggregate), -learning_rate_power) - pow((*accumulator_aggregate), -learning_rate_power)) / learning_rate * variable;

		/* quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2 */
		quadratic = 1.0 / (pow((*accumulator_new_aggregate), learning_rate_power) * learning_rate) + 2.0 * l2_regularization_strength;

		if (fabs((*linear_aggregate)) > l2_regularization_strength)
		{
			/* u = (sign(linear) * l1 - linear) / quadratic - var */
			(*update) = (sign((*linear_aggregate)) * l1_regularization_strength - (*linear_aggregate)) / quadratic - variable;
		}
		else
		{
			/* u = -var */
			(*update) = -variable;
		}

		/* accum = accum_new */
		(*accumulator_aggregate) = (*accumulator_new_aggregate);
	}
	else
	{
		/* grad_with_shrinkage = g + 2 * l2_shrinkage * var */
		grad_with_shrinkage = gradient + 2.0 * l2_shrinkage_regularization_strength * variable;

		/* accum_new = accum + grad_with_shrinkage * grad_with_shrinkage */
		(*accumulator_new_aggregate) = (*accumulator_aggregate) + grad_with_shrinkage * grad_with_shrinkage;

		/* linear += grad_with_shrinkage + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var */
		(*linear_aggregate) += grad_with_shrinkage + (pow((*accumulator_new_aggregate), -learning_rate_power) - pow((*accumulator_aggregate), -learning_rate_power)) / learning_rate * variable;

		/* quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2 */
		quadratic = 1.0 / (pow((*accumulator_new_aggregate), learning_rate_power) * learning_rate) + 2.0 * l2_regularization_strength;

		if (fabs((*linear_aggregate)) > l1_regularization_strength)
		{
			/* u = (sign(linear) * l1 - linear) / quadratic - var */
			(*update) = (sign((*linear_aggregate)) * l1_regularization_strength - (*linear_aggregate)) / quadratic - variable;
		}
		else
		{
			/* u = -var */
			(*update) = -variable;
		}

		/* accum = accum_new */
		(*accumulator_aggregate) = (*accumulator_new_aggregate);
	}
} // end of FTRLOptimizer function

/******************************************************************************************/
/****************************** UPDATE TRAINABLE PARAMETERS *******************************/
/******************************************************************************************/

/* This function updates the neural network's trainable parameters using this training batch's finalized gradients */
void UpdateNeuralNetworkTrainableParameters(unsigned int total_parameters, struct TrainableParameters *neural_network_parameters)
{
	unsigned int i;

	for (i = 0; i < total_parameters; i++)
	{
		neural_network_parameters[i].variable += neural_network_parameters[i].update;
	} // end of i loop
} // end of UpdateNeuralNetworkTrainableParameters function

/* This function updates the batch normalization layer parameters using this training batch's finalized gradients */
void UpdateBatchNormalizationParameters(unsigned int total_batch_normalization_neurons, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters)
{
	unsigned int i;

	for (i = 0; i < total_batch_normalization_neurons; i++)
	{
		batch_normalization_layer_neurons_parameters[i].batch_norm_beta.variable += batch_normalization_layer_neurons_parameters[i].batch_norm_beta.update;
		batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.variable += batch_normalization_layer_neurons_parameters[i].batch_norm_gamma.update;
	} // end of i loop
} // end of update_neural_network_bias_weights function

/******************************************************************************************/
/*********************************** SAVE FINAL MODEL *************************************/
/******************************************************************************************/

/* This function saves the final model's trainable parameters */
void SaveFinalModelTrainableParameters(unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, struct TrainableParameters *neural_network_kernel_weights, unsigned int *kernel_weights_offset_indices, struct TrainableParameters *neural_network_bias_weights, unsigned int *bias_weights_offset_indices, struct TrainableParameters *parametric_relu_alphas, unsigned int *parametric_relu_alpha_offset_indices, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, unsigned int *batch_normalization_layer_offset_indices)
{
	unsigned int i, j, k, array_index;

	/* Save final parameters */

	FILE *outfile_kernel_weights = fopen("outputs/kernel_weights.txt", "w");
	if (outfile_kernel_weights == NULL)
	{
		printf("main: Failed to open file %s\n", "outputs/kernel_weights.txt");
	}
	else
	{
		for (i = 0; i < number_of_layers - 1; i++)
		{
			for (j = 0; j < neural_network_layers_hyperparameters[i].number_of_neurons; j++)
			{
				for (k = 0; k < neural_network_layers_hyperparameters[i + 1].number_of_neurons; k++)
				{
					array_index = j * neural_network_layers_hyperparameters[i + 1].number_of_neurons + k + kernel_weights_offset_indices[i];

					fprintf(outfile_kernel_weights, "%.16f\t", neural_network_kernel_weights[array_index].variable);
				} // end of k loop
				fprintf(outfile_kernel_weights, "\n");
			} // end of j loop
		} // end of i loop
		fclose(outfile_kernel_weights);
	}

	FILE *outfile_bias_weights = fopen("outputs/bias_weights.txt", "w");
	if (outfile_bias_weights == NULL)
	{
		printf("main: Failed to open file %s\n", "outputs/bias_weights.txt");
	}
	else
	{
		for (i = 0; i < number_of_layers - 1; i++)
		{
			if (neural_network_layers_hyperparameters[i + 1].bias_weight_initialization_type >= 0)
			{
				for (j = 0; j < neural_network_layers_hyperparameters[i + 1].number_of_neurons; j++)
				{
					array_index = j + bias_weights_offset_indices[i];

					fprintf(outfile_bias_weights, "%.16f\t", neural_network_bias_weights[array_index].variable);
				} // end of j loop
				fprintf(outfile_bias_weights, "\n");
			}
		} // end of i loop
		fclose(outfile_bias_weights);
	}

	FILE *outfile_parametric_relu_alphas = fopen("outputs/parametric_relu_alphas.txt", "w");
	if (outfile_parametric_relu_alphas == NULL)
	{
		printf("main: Failed to open file %s\n", "outputs/parametric_relu_alphas.txt");
	}
	else
	{
		for (i = 0; i < number_of_layers - 1; i++)
		{
			if (neural_network_layers_hyperparameters[i + 1].activation_type == 5) // parametric relu
			{
				for (j = 0; j < neural_network_layers_hyperparameters[i + 1].number_of_neurons; j++)
				{
					array_index = j + parametric_relu_alpha_offset_indices[i];

					fprintf(outfile_parametric_relu_alphas, "%.16f\t", parametric_relu_alphas[array_index].variable);
				} // end of j loop
				fprintf(outfile_parametric_relu_alphas, "\n");
			}
		} // end of i loop
		fclose(outfile_parametric_relu_alphas);
	}

	FILE *outfile_batch_norm_parameters = fopen("outputs/batch_norm_parameters.txt", "w");
	if (outfile_batch_norm_parameters == NULL)
	{
		printf("main: Failed to open file %s\n", "outputs/batch_norm_parameters.txt");
	}
	else
	{
		for (i = 0; i < number_of_layers - 1; i++)
		{
			if (neural_network_layers_hyperparameters[i + 1].batch_norm_flag == 1)
			{
				for (j = 0; j < neural_network_layers_hyperparameters[i + 1].number_of_neurons; j++)
				{
					array_index = j + batch_normalization_layer_offset_indices[i];

					fprintf(outfile_batch_norm_parameters, "%.16f\t%.16f\t%.16f\t%.16f\n", batch_normalization_layer_neurons_parameters[array_index].batch_norm_moving_mean, batch_normalization_layer_neurons_parameters[array_index].batch_norm_moving_variance, batch_normalization_layer_neurons_parameters[array_index].batch_norm_beta.variable, batch_normalization_layer_neurons_parameters[array_index].batch_norm_gamma.variable);
				} // end of j loop
			}
		} // end of i loop
		fclose(outfile_batch_norm_parameters);
	}
} // end of SaveFinalModelTrainableParameters function

/******************************************************************************************/
/************************************* DEBUG PRINT ****************************************/
/******************************************************************************************/

/* This function prints the FeedForward tensors if we are in debug print mode */
void DebugPrintFeedForward(unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, unsigned int batch_size, double *batch_features_tensor, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int *neural_network_neurons_minus_inputs_offset_indices, struct TrainableParameters *neural_network_kernel_weights, unsigned int *kernel_weights_offset_indices, struct TrainableParameters *neural_network_bias_weights, unsigned int *bias_weights_offset_indices, struct TrainableParameters *parametric_relu_alphas, unsigned int *parametric_relu_alpha_offset_indices, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, double *batch_normalization_layer_xmu, unsigned int *batch_normalization_layer_offset_indices)
{
	unsigned int i, j, k, index;

	printf("FeedForward: batch_features_tensor:\n");
	for (i = 0; i < batch_size; i++)
	{
		for (j = 0; j < neural_network_layers_hyperparameters[0].number_of_neurons; j++)
		{
			index = i * neural_network_layers_hyperparameters[0].number_of_neurons + j;

			printf("%.16f\t", batch_features_tensor[index]);
		} // end of j loop
		printf("\n");
	} // end of i loop
	printf("\n");

	printf("FeedForward: neural_network_layer_weighted_sums:\n");
	for (i = 0; i < number_of_layers - 1; i++)
	{
		printf("FeedForward: Neuron layer %u\n", i + 1);
		for (j = 0; j < batch_size; j++)
		{
			for (k = 0; k < neural_network_layers_hyperparameters[i + 1].number_of_neurons; k++)
			{
				index = j * neural_network_layers_hyperparameters[i + 1].number_of_neurons + k + batch_size * neural_network_neurons_minus_inputs_offset_indices[i];

				printf("%.16f\t", neural_network_layer_neurons[index].weighted_sum);
			} // end of k loop
			printf("\n");
		} // end of j loop
		printf("\n");
	} // end of i loop

	printf("FeedForward: neural_network_layer_activations:\n");
	for (i = 0; i < number_of_layers - 1; i++)
	{
		printf("FeedForward: Neuron layer %u\n", i + 1);
		for (j = 0; j < batch_size; j++)
		{
			for (k = 0; k < neural_network_layers_hyperparameters[i + 1].number_of_neurons; k++)
			{
				index = j * neural_network_layers_hyperparameters[i + 1].number_of_neurons + k + batch_size * neural_network_neurons_minus_inputs_offset_indices[i];

				printf("%.16f\t", neural_network_layer_neurons[index].activation);
			} // end of k loop
			printf("\n");
		} // end of j loop
		printf("\n");
	} // end of i loop

	printf("FeedForward: batch_normalization_layer_xmu:\n");
	for (i = 0; i < number_of_layers - 1; i++)
	{
		if (neural_network_layers_hyperparameters[i + 1].batch_norm_flag == 1)
		{
			printf("FeedForward: Neuron layer %u\n", i + 1);
			for (j = 0; j < batch_size; j++)
			{
				for (k = 0; k < neural_network_layers_hyperparameters[i + 1].number_of_neurons; k++)
				{
					index = j * neural_network_layers_hyperparameters[i + 1].number_of_neurons + k + batch_size * batch_normalization_layer_offset_indices[i];
					printf("%.16f\t", batch_normalization_layer_xmu[index]);
				} // end of k loop
				printf("\n");
			} // end of j loop
			printf("\n");
		}
	} // end of i loop
} // end of DebugPrintFeedForward function

/* This function prints the Backpropagation tensors if we are in debug print mode */
void DebugPrintBackpropagation(unsigned int number_of_layers, struct NeuralNetworkLayerHyperparameters *neural_network_layers_hyperparameters, unsigned int batch_size, double *batch_labels_tensor, struct NeuralNetworkLayerNeuron *neural_network_layer_neurons, unsigned int *neural_network_neurons_minus_inputs_offset_indices, struct TrainableParameters *neural_network_kernel_weights, unsigned int *kernel_weights_offset_indices, struct TrainableParameters *neural_network_bias_weights, unsigned int *bias_weights_offset_indices, struct TrainableParameters *parametric_relu_alphas, unsigned int *parametric_relu_alpha_offset_indices, struct BatchNormalizationLayerNeuronParameters *batch_normalization_layer_neurons_parameters, unsigned int *batch_normalization_layer_offset_indices)
{
	unsigned int i, j, k, index;

	printf("DebugPrintBackpropagation: batch_labels_tensor:\n");
	for (i = 0; i < batch_size; i++)
	{
		for (j = 0; j < neural_network_layers_hyperparameters[number_of_layers - 1].number_of_neurons; j++)
		{
			index = i * neural_network_layers_hyperparameters[number_of_layers - 1].number_of_neurons + j;

			printf("%.16f\t", batch_labels_tensor[index]);
		} // end of j loop
		printf("\n");
	} // end of i loop
	printf("\n");

	printf("DebugPrintBackpropagation: neural_network_layer_deltas:\n");
	for (i = 0; i < number_of_layers - 1; i++)
	{
		printf("DebugPrintBackpropagation: Neuron layer %u\n", i + 1);
		for (j = 0; j < batch_size; j++)
		{
			for (k = 0; k < neural_network_layers_hyperparameters[i + 1].number_of_neurons; k++)
			{
				index = j * neural_network_layers_hyperparameters[i + 1].number_of_neurons + k + batch_size * neural_network_neurons_minus_inputs_offset_indices[i];

				printf("%.16f\t", neural_network_layer_neurons[index].delta);
			} // end of k loop
			printf("\n");
		} // end of j loop
		printf("\n");
	} // end of i loop

	printf("DebugPrintBackpropagation: neural_network_kernel_weights_gradients:\n");
	for (i = 0; i < number_of_layers - 1; i++)
	{
		printf("DebugPrintBackpropagation: Kernel gradient matrix %u\n", i);
		for (j = 0; j < neural_network_layers_hyperparameters[i].number_of_neurons; j++)
		{
			for (k = 0; k < neural_network_layers_hyperparameters[i + 1].number_of_neurons; k++)
			{
				index = j * neural_network_layers_hyperparameters[i + 1].number_of_neurons + k + kernel_weights_offset_indices[i];

				printf("%.16f\t", neural_network_kernel_weights[index].gradient);
			} // end of k loop
			printf("\n");
		} // end of j loop
		printf("\n");
	} // end of i loop
	printf("\n");

	printf("DebugPrintBackpropagation: neural_network_bias_weights_gradients:\n");
	for (i = 0; i < number_of_layers - 1; i++)
	{
		if (neural_network_layers_hyperparameters[i + 1].bias_weight_initialization_type >= 0)
		{
			printf("DebugPrintBackpropagation: Bias gradient vector %u\n", i);
			for (j = 0; j < neural_network_layers_hyperparameters[i + 1].number_of_neurons; j++)
			{
				index = j + bias_weights_offset_indices[i];

				printf("%.16f\t", neural_network_bias_weights[index].gradient);
			} // end of j loop
			printf("\n");
		}
	} // end of i loop
	printf("\n");

	printf("DebugPrintBackpropagation: parametric_relu_alphas_gradients:\n");
	for (i = 0; i < number_of_layers - 1; i++)
	{
		if (neural_network_layers_hyperparameters[i + 1].activation_type == 5)
		{
			printf("DebugPrintBackpropagation: Parametric relu alpha gradient vector %u\n", i);
			for (j = 0; j < neural_network_layers_hyperparameters[i + 1].number_of_neurons; j++)
			{
				index = j + parametric_relu_alpha_offset_indices[i];

				printf("%.16f\t", parametric_relu_alphas[index].gradient);
			} // end of j loop
			printf("\n");
		}
	} // end of i loop
	printf("\n");

	printf("DebugPrintBackpropagation: batch_norm_beta_gradients:\n");
	for (i = 0; i < number_of_layers - 1; i++)
	{
		if (neural_network_layers_hyperparameters[i + 1].batch_norm_flag == 1)
		{
			printf("DebugPrintBackpropagation: Beta vector %u\n", i);
			for (j = 0; j < neural_network_layers_hyperparameters[i + 1].number_of_neurons; j++)
			{
				index = j + batch_normalization_layer_offset_indices[i];

				printf("%.16f\t", batch_normalization_layer_neurons_parameters[index].batch_norm_beta.gradient);
			} // end of j loop
			printf("\n");
		}
	} // end of i loop
	printf("\n");

	printf("DebugPrintBackpropagation: batch_norm_gamma_gradients:\n");
	for (i = 0; i < number_of_layers - 1; i++)
	{
		if (neural_network_layers_hyperparameters[i + 1].batch_norm_flag == 1)
		{
			printf("DebugPrintBackpropagation: Gamma vector %u\n", i);
			for (j = 0; j < neural_network_layers_hyperparameters[i + 1].number_of_neurons; j++)
			{
				index = j + batch_normalization_layer_offset_indices[i];

				printf("%.16f\t", batch_normalization_layer_neurons_parameters[index].batch_norm_gamma.gradient);
			} // end of j loop
			printf("\n");
		}
	} // end of i loop
	printf("\n");

	printf("DebugPrintBackpropagation: neural_network_kernel_weights:\n");
	for (i = 0; i < number_of_layers - 1; i++)
	{
		printf("DebugPrintBackpropagation: Kernel matrix %u\n", i);
		for (j = 0; j < neural_network_layers_hyperparameters[i].number_of_neurons; j++)
		{
			for (k = 0; k < neural_network_layers_hyperparameters[i + 1].number_of_neurons; k++)
			{
				index = j * neural_network_layers_hyperparameters[i + 1].number_of_neurons + k + kernel_weights_offset_indices[i];

				printf("%.16f\t", neural_network_kernel_weights[index].variable);
			} // end of k loop
			printf("\n");
		} // end of j loop
		printf("\n");
	} // end of i loop
	printf("\n");

	printf("DebugPrintBackpropagation: neural_network_bias_weights:\n");
	for (i = 0; i < number_of_layers - 1; i++)
	{
		if (neural_network_layers_hyperparameters[i + 1].bias_weight_initialization_type >= 0)
		{
			printf("DebugPrintBackpropagation: Bias vector %u\n", i);
			for (j = 0; j < neural_network_layers_hyperparameters[i + 1].number_of_neurons; j++)
			{
				index = j + bias_weights_offset_indices[i];

				printf("%.16f\t", neural_network_bias_weights[index].variable);
			} // end of j loop
			printf("\n");
		}
	} // end of i loop
	printf("\n");

	printf("DebugPrintBackpropagation: parametric_relu_alphas:\n");
	for (i = 0; i < number_of_layers - 1; i++)
	{
		if (neural_network_layers_hyperparameters[i + 1].activation_type == 5)
		{
			printf("DebugPrintBackpropagation: Parametric relu alpha vector %u\n", i);
			for (j = 0; j < neural_network_layers_hyperparameters[i + 1].number_of_neurons; j++)
			{
				index = j + parametric_relu_alpha_offset_indices[i];

				printf("%.16f\t", parametric_relu_alphas[index].variable);
			} // end of j loop
			printf("\n");
		}
	} // end of i loop
	printf("\n");

	printf("DebugPrintBackpropagation: batch_norm_beta:\n");
	for (i = 0; i < number_of_layers - 1; i++)
	{
		if (neural_network_layers_hyperparameters[i + 1].batch_norm_flag == 1)
		{
			printf("DebugPrintBackpropagation: Beta vector %u\n", i);
			for (j = 0; j < neural_network_layers_hyperparameters[i + 1].number_of_neurons; j++)
			{
				index = j + batch_normalization_layer_offset_indices[i];

				printf("%.16f\t", batch_normalization_layer_neurons_parameters[index].batch_norm_beta.variable);
			} // end of j loop
			printf("\n");
		}
	} // end of i loop
	printf("\n");

	printf("DebugPrintBackpropagation: batch_norm_gamma:\n");
	for (i = 0; i < number_of_layers - 1; i++)
	{
		if (neural_network_layers_hyperparameters[i + 1].batch_norm_flag == 1)
		{
			printf("DebugPrintBackpropagation: Gamma vector %u\n", i);
			for (j = 0; j < neural_network_layers_hyperparameters[i + 1].number_of_neurons; j++)
			{
				index = j + batch_normalization_layer_offset_indices[i];

				printf("%.16f\t", batch_normalization_layer_neurons_parameters[index].batch_norm_gamma.variable);
			} // end of j loop
			printf("\n");
		}
	} // end of i loop
	printf("\n");
} // end of DebugPrintBackpropagation function
