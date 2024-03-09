from modules.layers import LinearLayer, ConvolutionLayer, Flatten
from modules.loss import CrossEntropy
from modules.activation import ReLU, Softmax
from modules.model import Model

from utils import load_dataset, compute_accuracy

# # my stuff starts
# from utils import (
#     plot_confusion_matrix,
#     plot_history,
#     plot_predictions,
#     load_dataset,
#     compute_accuracy,
# )
# # my stuff ends

class TestAccuracy:

    def test_one(self):
        """Test a model for getting accuracy > 90%"""
        x_train, x_test, y_train, y_test = load_dataset()

        layer_list = [
            ConvolutionLayer(in_channels=1, out_channels=3, kernel_size=5),
            ReLU(),
            Flatten(),
            LinearLayer(48, 10),
            Softmax(),
        ]

        model = Model(layer_list, CrossEntropy())

        model.train(x_train, y_train, 25, 0.01, 32)

        predictions = model.predict(x_test)
        accuracy = compute_accuracy(predictions, y_test)

        # # my stuff starts
        # history = model.train(
        #     x_train,
        #     y_train,
        #     epochs=25,
        #     learning_rate=0.01,
        #     batch_size=32,
        # )
        # plot_history(history)
        # plot_predictions(x_test, predictions, accuracy)
        # plot_confusion_matrix(y_test, predictions)
        # print("YOUR ACC:", accuracy)
        # # my stuff ends

        assert accuracy > 0.9
