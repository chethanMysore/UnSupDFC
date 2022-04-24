#!/usr/bin/env python

# from __future__ import print_function, division
"""

Purpose :

"""
import torch.nn
import torch
import torch.nn as nn

__author__ = "Chethan Radhakrishna and Soumick Chatterjee"
__credits__ = ["Chethan Radhakrishna", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Chethan Radhakrishna"
__email__ = "chethan.radhakrishna@st.ovgu.de"
__status__ = "Development"


class ConvComponent3D(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1, bias=True, no_relu=False):
        super(ConvComponent3D, self).__init__()
        self.conv = nn.Sequential()
        self.conv.add_module("conv_1", nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                                                 stride=stride, padding=padding, bias=bias))
        if not no_relu:
            self.conv.add_module("p_relu_1", nn.PReLU(init=0.25))
        self.conv.add_module("bn_1", nn.BatchNorm3d(num_features=out_channels))
        # self.conv.add_module("conv_2", nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=k_size,
        #                                          stride=stride, padding=padding, bias=bias))
        # if not no_relu:
        #     self.conv.add_module("p_relu_2", nn.PReLU(init=0.25))
        # self.conv.add_module("bn_2", nn.BatchNorm3d(num_features=out_channels))

    def forward(self, x):
        x = self.conv(x)
        return x


class LinearClassifier(nn.Module):
    """
    Linear Classifier to create response map from learned features
    """
    def __init__(self, in_features, num_classes):
        super(LinearClassifier, self).__init__()
        self.classifier = torch.nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        x = self.classifier(x)
        return x


class DFC3D(nn.Module):
    """

    """

    def __init__(self, in_ch=1, init_filter=64, num_conv=3, num_classes=3):
        super(DFC3D, self).__init__()

        self.n_channels = init_filter
        self.num_conv = num_conv
        self.conv_components = []
        # First Convolution Block
        self.conv_block1 = ConvComponent3D(in_channels=in_ch, out_channels=self.n_channels)

        # m-1 such convolution blocks
        for i in range(self.num_conv - 1):
            self.conv_components.append(ConvComponent3D(in_channels=self.n_channels,
                                                        out_channels=self.n_channels * 2).cuda())
            self.n_channels = self.n_channels * 2
            # If MaxPool is necessary or any transitions add it here

        self.conv_blocks = nn.Sequential(*self.conv_components)

        # Response map computation using 1D convolution
        self.conv1D = ConvComponent3D(in_channels=self.n_channels, out_channels=num_classes, k_size=1, stride=1,
                                      padding=0, no_relu=True)
        # self.normaliser = nn.BatchNorm3d(num_features=num_classes)
        # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        # First initial convolution
        comp_op = self.conv_block1(x)

        # m-1 such convolution blocks
        comp_op = self.conv_blocks(comp_op)

        # Response map computation using 1D convolution
        res_map = self.conv1D(comp_op)
        return res_map
