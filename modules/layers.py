import numpy as np

class ConvolutionLayer:

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        np.random.seed(2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Initialize weights and biases randomly
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32)
        self.bias = np.random.randn(out_channels).astype(np.float32)

        self.inputs = None
        self.has_weights = True

    def forward(self, inputs):
        """Forward pass for a convolution layer

        Args:
            inputs (np.ndarray):
                array of shape
                (batch_size, in_channels, height, width)

        Returns: (np.ndarray):
            array of shape
            (batch_size, out_channels, new_height, new_width)
        """
        self.inputs = inputs
        # ================ Insert Code Here ================

        # print("output_width:", output_width)
        # print("output_height:", output_height)
        # print("self.out_channels", self.out_channels)
        # print("input_width:", input_width)
        # print("input_height:", input_height)
        # print("in_channels:", in_channels)
        # print("self.weights.shape",self.weights.shape)
        
        # self.inputs = inputs
        batch_size, in_channels, input_height, input_width = inputs.shape
        output_height = (input_height - self.kernel_size) // self.stride + 1
        output_width = (input_width - self.kernel_size) // self.stride + 1
        outputs = np.zeros((batch_size, self.out_channels, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                receptive_field = inputs[:, :, i * self.stride:i * self.stride + self.kernel_size,
                                  j * self.stride:j * self.stride + self.kernel_size]
                outputs[:, :, i, j] = np.sum(receptive_field[:, None, :, :, :] * self.weights, axis=(2, 3, 4)) + self.bias

        return outputs
        # ==================================================

    def backward(self, d_outputs):
        """Backward pass of convolution layer

        Args:
            d_outputs (np.ndarray):
                derivative of loss with respect to the output
                of the layer. Will have shape
                (batch_size, out_channels, new_height, new_width)

        Returns: (dict):
            Dictionary containing the derivatives of loss with respect to
            the weights and bias and input of the layer. The keys of
            the dictionary should be "d_weights", "d_bias", and "d_out"

        """

        # self.inputs = None
        if self.inputs is None:
            raise NotImplementedError(
                "Need to call forward function before backward function"
            )

        # ================ Insert Code Here ================
        batch_size, out_channels, out_height, out_width = d_outputs.shape
        d_inputs = np.zeros_like(self.inputs)
        d_weights = np.zeros_like(self.weights)
        d_bias = np.zeros_like(self.bias)

        for i in range(out_height):
            for j in range(out_width):
                for k in range(out_channels):  # Loop over each filter
                    
                    # the input slice corresponding to this output pixel
                    input_slice = self.inputs[:, :, i * self.stride:i * self.stride + self.kernel_size,
                                  j * self.stride:j * self.stride + self.kernel_size]
                    
                    # derivative input update
                    d_inputs[:, :, i * self.stride:i * self.stride + self.kernel_size,
                    j * self.stride:j * self.stride + self.kernel_size] += \
                        d_outputs[:, k, i, j][:, np.newaxis, np.newaxis, np.newaxis] * self.weights[k, :, :, :]
                    
                    # bwp w update
                    d_weights[k, :, :, :] += np.sum(
                        input_slice[:, :, :, :] * d_outputs[:, k, i, j][:, np.newaxis, np.newaxis, np.newaxis], axis=0)
                    
                d_bias += np.sum(d_outputs[:, :, i, j], axis=0)

        return {"d_weights": d_weights, "d_bias": d_bias, "d_out": d_inputs}
        # ==================================================

    def update(self, d_weights, d_bias, learning_rate):

        # ================ Insert Code Here ================
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
        # ==================================================


class Flatten:
    def __init__(self):
        self.inputs_shape = None
        self.has_weights = False

    def forward(self, inputs):
        self.inputs_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, d_outputs):
        return {"d_out": d_outputs.reshape(self.inputs_shape)}


class LinearLayer:
    def __init__(self, in_features, out_features):
        np.random.seed(1)
        self.in_features = in_features
        self.out_features = out_features

        self.weights = np.random.rand(out_features, in_features).astype(np.float32)
        self.bias = np.random.rand(out_features).astype(np.float32)

        self.inputs = None
        self.has_weights = True

    def forward(self, inputs):
        """Forward pass for a linear layer

        Args:
            inputs (np.ndarray):
                array of shape (batch_size, in_features)

        Returns: (np.ndarray):
            array of shape (batch_size, out_features)
        """

        # ================ Insert Code Here ================
        self.inputs = inputs
        return np.dot(inputs, self.weights.T) + self.bias
        # ==================================================

    def backward(self, d_outputs):
        """Backward pass of Linear layer

        Args:
            d_outputs (np.ndarray):
                derivative of loss with respect to the output
                of the layer. Will have shape
                (batch_size, out_features)

        Returns: (dict):
            Dictionary containing the derivatives of loss with respect to
            the weights and bias and input of the layer. The keys of
            the dictionary should be "d_weights", "d_bias", and "d_out"
        """
        if self.inputs is None:
            raise NotImplementedError(
                "Need to call forward function before backward function"
            )

        # gradients wrt weights
        d_weights = np.dot(d_outputs.T, self.inputs)

        # gradient wrt bias
        d_bias = np.sum(d_outputs, axis=0)

        # gradient wrt input
        d_input = np.dot(d_outputs, self.weights)

        return {"d_weights": d_weights, "d_bias": d_bias, "d_out": d_input}
        # ==================================================

    def update(self, d_weights, d_bias, learning_rate):

        # ================ Insert Code Here ================
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
        # ==================================================
