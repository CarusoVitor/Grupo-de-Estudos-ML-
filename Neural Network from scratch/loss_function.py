import numpy as np
from nnfs.datasets import spiral_data



# =================
#   LOSS FUNCTIONS
# =================

class Loss:
    def calculate(self, y_hat, y):
        '''
            Recebe dois batches:
                * 'ŷ' (output da rede, predição) e
                * 'y' (valores verdade).

            Retorna a loss mresultante do batch.
        '''
        sample_losses = self.forward(y_hat, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class LossCategoricalCrossEntropy(Loss):
    def forward(self, y_hat, y):
        '''
            Recebe dois batches:
                * 'ŷ' (output da rede, predição) e
                * 'y' (valores verdade).

            Retorna a loss resultante de cada sample.
        '''
        samples = len(y_hat)
        y_hat_clipped = np.clip(y_hat, 1e-7, 1-1e-7)

        #
        if len(y.shape) == 1:
            predictions_confidences = y_hat_clipped[range(samples), y]
        else:
            predictions_confidences = np.sum(y_hat_clipped*y, axis=1)

        # print(predictions_confidences)

        negative_log_likelihoods = -np.log(predictions_confidences)
        # print(negative_log_likelihoods)

        return negative_log_likelihoods


# def categorical_loss_entropy(x: np.ndarray, y: np.ndarray):
#     log_part = np.log(x) # [log(0.7), log(0.1), log(0.2)]
#     mult_part = log_part * y # [log(0.7), log(0.1), log(0.2)]) * [1, 0, 0]
#     loss = -np.sum(mult_part, axis=1) # -sum(log(0.7) * 1, log(0.1) * 0, log(0.2) * 0])
#     return loss
