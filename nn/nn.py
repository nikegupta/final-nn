# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike
import random

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.
        lammbda: weight of L1 loss

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch,
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str,
        lammbda: float
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size
        self._lammbda = lammbda

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        Z_curr = (A_prev @ W_curr.T)
        for i in range(Z_curr.shape[0]):
            Z_curr[i] += b_curr.T[0]

        if activation == 'relu':
            A_curr = self._relu(Z_curr)
        elif activation == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)
        else:
            raise ValueError('Invalid activation function provided')
        
        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        cache = {}

        A_curr = X

        #1th layer activation (X)
        cache.update({'A1': A_curr})

        for i in range(1,len(self.arch)+1):
            A_curr, Z_curr = self._single_forward(self._param_dict[f'W{i}'],
                                                  self._param_dict[f'b{i}'],
                                                  A_curr,
                                                  self.arch[i-1]['activation'])
            cache.update({f'A{i+1}': A_curr, f'Z{i+1}': Z_curr})

        return cache

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """

        m = A_prev.shape[0]

        if activation_curr == 'sigmoid':
            dZ = self._sigmoid_backprop(dA_curr,Z_curr)
        elif activation_curr == 'relu':
            dZ = self._relu_backprop(dA_curr,Z_curr)
        else:
            raise ValueError('Invalid activation function provided')
        
        dW_curr = dZ.T @ A_prev / m
        db_curr = (np.sum(dZ,axis=0,keepdims=True) / m).T
        dA_prev = dZ @ W_curr

        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        gradient = {}

        #calculate loss gradient for output layer
        #lp1 is short for layer plus 1, refering to index i+1 activation
        if self._loss_func == 'binary cross entropy':
            dA_lp1 = self._binary_cross_entropy_backprop(y,y_hat)
        elif self._loss_func == 'mean squared error':
            dA_lp1 = self._mean_squared_error_backprop(y,y_hat)
        else:
            raise ValueError('Invalid loss function provided')
        
        #loop through layers in reverse order
        for i in range(len(self.arch),-1,-1):

            #if outermost layer, use dA_lp1
            if i == len(self.arch):
                dA_curr = dA_lp1

            #else, do normal backprop
            else:

                #current index of param dict
                idx = i + 1

                #run single pass backprop
                dA_lp1, dW_curr, db_curr = self._single_backprop(self._param_dict[f'W{idx}'],
                                                                  self._param_dict[f'b{idx}'],
                                                                  cache[f'Z{idx+1}'],
                                                                  cache[f'A{idx}'],
                                                                  dA_curr,
                                                                  self.arch[i]['activation'])
                
                #set dA
                dA_curr = dA_lp1

                #store graidents
                gradient[f'W{idx}'] = dW_curr
                gradient[f'b{idx}'] = db_curr

        return gradient

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """

        keys = list(self._param_dict.keys())
        for key in keys:
            current_weight = self._param_dict[key]

            if key[0] == 'W':
                self._param_dict[key] = current_weight - ((self._lr * (1 / self._batch_size) * grad_dict[key]) + abs(self._lammbda * current_weight))

            else:
                self._param_dict[key] = current_weight - (self._lr * (1 / self._batch_size) * grad_dict[key])

            
    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        random.seed(self._seed)

        #make lists for return
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        #set iteration value for while loop
        iteration = 1

        #main while loop
        while iteration <= self._epochs:

            # Create batches
            num_batches = int(X_train.shape[0] / self._batch_size)
            X_batch = np.array_split(X_train, num_batches)
            y_batch = np.array_split(y_train, num_batches)

            #make list of batches in random order
            i_list = list(range(num_batches))
            random.shuffle(i_list)

            # Iterate through batches (one of these loops is one epoch of training)
            train_loss_cur_epoch = []
            val_loss_cur_epoch = []
            for i in i_list:
                X_train_1 = X_batch[i]
                y_train_1 = y_batch[i]
                
                #make prediction and get training loss
                y_hat = self.predict(X_train_1)
                if self._loss_func == 'binary cross entropy':
                    train_loss = self._binary_cross_entropy(y_train_1,y_hat)
                elif self._loss_func == 'mean squared error':
                    train_loss = self._mean_squared_error(y_train_1,y_hat)
                else:
                    raise ValueError('Invalid loss function provided')
                train_loss_cur_epoch.append(train_loss)

                #calculate gradient and update weights
                cache = self.forward(X_train_1)
                gradient = self.backprop(y_train_1,y_hat,cache)
                self._update_params(gradient)

                # compute validation loss
                y_hat = self.predict(X_val)
                if self._loss_func == 'binary cross entropy':
                    val_loss = self._binary_cross_entropy(y_val,y_hat)
                elif self._loss_func == 'mean squared error':
                    val_loss = self._mean_squared_error(y_val,y_hat)
                else:
                    raise ValueError('Invalid loss function provided')
                val_loss_cur_epoch.append(val_loss)

            #average loss per epoch
            per_epoch_loss_train.append(np.sum(train_loss_cur_epoch) / (num_batches * self._batch_size))
            per_epoch_loss_val.append(np.sum(val_loss_cur_epoch) / (num_batches * X_val.shape[0]))
            iteration += 1

        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        
        cache = self.forward(X)
        keys = list(cache.keys())

        #second to last key corresponds to last activation layer
        return cache[keys[-2]]

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """

        return 1 / (1 + np.exp(-Z))

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        
        sZ = self._sigmoid(Z)
        return (dA * sZ * (1 - sZ))

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """

        return np.where(Z <= 0, 0, Z)

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        return dA * np.where(Z > 0, Z, 0)

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """ 
        loss = (y * np.log10(y_hat)) + ((1 - y) * np.log10(1 - y_hat))
        return np.sum(loss) * (-1 / y.shape[0])

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """

        return - ((y / y_hat) - ((1 - y) / (1- y_hat)))


    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        loss = pow((y - y_hat),2)
        return np.sum(loss) / y.shape[0]

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """

        return (2 / y.shape[0]) * (y_hat - y)
