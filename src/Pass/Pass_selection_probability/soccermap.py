"""Implements the SoccerMap architecture."""
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import matplotlib.pyplot as plt
from utils import ToSoccerMapTensor


class _FeatureExtractionLayer(nn.Module):
    """The 2D-convolutional feature extraction layer of the SoccerMap architecture.

    The probability at a single location is influenced by the information we
    have of nearby pixels. Therefore, convolutional filters are used for
    spatial feature extraction.

    Two layers of 2D convolutional filters with a 5 × 5 receptive field and
    stride of 1 are applied, each one followed by a ReLu activation function.
    To keep the same dimensions after the convolutional filters, symmetric
    padding is applied. It fills the padding cells with values that are
    similar to those around it, thus avoiding border-image artifacts that can
    hinder the model’s predicting ability and visual representation.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, 32, kernel_size=(5, 5), stride=1, padding="valid")
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=1, padding="valid")
        # (left, right, top, bottom)
        self.symmetric_padding = nn.ReplicationPad2d((2, 2, 2, 2))

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.symmetric_padding(x)
        x = F.relu(self.conv_2(x))
        x = self.symmetric_padding(x)
        return x


class _PredictionLayer(nn.Module):
    """The prediction layer of the SoccerMap architecture.

    The prediction layer consists of a stack of two convolutional layers, the
    first with 32 1x1 convolutional filters followed by an ReLu activation
    layer, and the second consists of one 1x1 convolutional filter followed by
    a linear activation layer. The spatial dimensions are kept at each step
    and 1x1 convolutions are used to produce predictions at each location.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 32, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(32, 1, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)  # linear activation
        return x


class _UpSamplingLayer(nn.Module):
    """The upsampling layer of the SoccerMap architecture.

    The upsampling layer provides non-linear upsampling by first applying a 2x
    nearest neighbor upsampling and then two layers of convolutional filters.
    The first convolutional layer consists of 32 filters with a 3x3 activation
    field and stride 1, followed by a ReLu activation layer. The second layer
    consists of 1 layer with a 3x3 activation field and stride 1, followed by
    a linear activation layer. This upsampling strategy has been shown to
    provide smoother outputs.
    """

    def __init__(self):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding="valid")
        self.conv2 = nn.Conv2d(32, 1, kernel_size=(3, 3), stride=1, padding="valid")
        self.symmetric_padding = nn.ReplicationPad2d((1, 1, 1, 1))

    def forward(self, x: torch.Tensor):
        x = self.up(x)
        x = F.relu(self.conv1(x))
        x = self.symmetric_padding(x)
        x = self.conv2(x)  # linear activation
        x = self.symmetric_padding(x)
        return x


class _FusionLayer(nn.Module):
    """The fusion layer of the SoccerMap architecture.

    The fusion layer merges the final prediction surfaces at different scales
    to produce a final prediction. It concatenates the pair of matrices and
    passes them through a convolutional layer of one 1x1 filter.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=(1, 1), stride=1)

    def forward(self, x: List[torch.Tensor]):
        out = self.conv(torch.cat(x, dim=1))  # linear activation
        return out


class SoccerMapPassSelect(nn.Module):
    """SoccerMap architecture.

    SoccerMap is a deep learning architecture that is capable of estimating
    full probability surfaces for pass probability, pass slection likelihood
    and pass expected values from spatiotemporal data.

    The input consists of a stack of c matrices of size lxh, each representing a
    subset of the available spatiotemporal information in the current
    gamestate. The specific choice of information for each of these c slices
    might vary depending on the problem being solved

    Parameters
    ----------
    in_channels : int, default: 13
        The number of spatiotemporal input channels.

    References
    ----------
    .. [1] Fernández, Javier, and Luke Bornn. "Soccermap: A deep learning
       architecture for visually-interpretable analysis in soccer." Joint
       European Conference on Machine Learning and Knowledge Discovery in
       Databases. Springer, Cham, 2020.
    """

    def __init__(self, in_channels=13):
        super().__init__()

        # Convolutions for feature extraction at 1x, 1/2x and 1/4x scale
        self.features_x1 = _FeatureExtractionLayer(in_channels)
        self.features_x2 = _FeatureExtractionLayer(64)
        self.features_x4 = _FeatureExtractionLayer(64)

        # Layers for down and upscaling and merging scales
        self.up_x2 = _UpSamplingLayer()
        self.up_x4 = _UpSamplingLayer()
        self.down_x2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.down_x4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.fusion_x2_x4 = _FusionLayer()
        self.fusion_x1_x2 = _FusionLayer()

        # Prediction layers at each scale
        self.prediction_x1 = _PredictionLayer()
        self.prediction_x2 = _PredictionLayer()
        self.prediction_x4 = _PredictionLayer()
        
    def forward(self, x):
        # Feature extraction
        f_x1 = self.features_x1(x)
        f_x2 = self.features_x2(self.down_x2(f_x1))
        f_x4 = self.features_x4(self.down_x4(f_x2))

        # Prediction
        pred_x1 = self.prediction_x1(f_x1)
        pred_x2 = self.prediction_x2(f_x2)
        pred_x4 = self.prediction_x4(f_x4)

        # Fusion
        x4x2combined = self.fusion_x2_x4([self.up_x4(pred_x4), pred_x2])
        combined = self.fusion_x1_x2([self.up_x2(x4x2combined), pred_x1])



        # Transform combined into a vector
        combined_vector = combined.view(-1)

        # Apply softmax
        softmax_vector = F.softmax(combined_vector, dim=0)
        print(softmax_vector)


        # Transform back to the original shape
        combined_softmax = softmax_vector.view_as(combined)

        return combined_softmax


def pixel(surface, mask):
    """Return the prediction at a single pixel.

    This custom layer is used to evaluate the loss at the pass destination.

    Parameters
    ----------
    surface : torch.Tensor
        The final prediction surface.
    mask : torch.Tensor
        A sparse spatial representation of the final pass destination.

    Returns
    -------
    torch.Tensor
        The prediction at the cell on the surface that matches the actual
        pass destination.
    """
    masked = surface * mask
    value = torch.sum(masked, dim=(3, 2))
    return value


def test_soccer_map_random():
    torch.manual_seed(42)
    
    # Definindo as dimensões da entrada
    batch_size = 1
    in_channels = 19
    height = 68
    width = 104

    # Criando uma entrada aleatória
    input_tensor = torch.randn(batch_size, in_channels, height, width)

    # Instanciando o modelo SoccerMap
    model = SoccerMapPassSelect(in_channels=in_channels)

    # Passando a entrada pelo modelo
    output = model(input_tensor)
    
    mask = np.zeros((height, width), dtype=int)

    pos_x = 21
    pos_y = 21

    # Colocando o valor 1 na posição (21, 21)
    mask[pos_x, pos_y] = 1
    mask = torch.tensor(mask)
    print(pixel(output,mask))

    print(output)
    # Convertendo a saída para numpy e removendo dimensões extras
    output_np = output.detach().numpy().squeeze()


    # Plotando o heatmap
    plt.imshow(output_np, cmap='jet', interpolation='gaussian')
    plt.colorbar()
    plt.title("Heatmap da saída do SoccerMap")
    plt.xlabel("Largura")
    plt.ylabel("Altura")
    plt.show()




if __name__ == "__main__":
    test_soccer_map_random()