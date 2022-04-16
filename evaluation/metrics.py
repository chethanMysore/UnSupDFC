#!/usr/bin/env python

"""

Purpose :

"""

import torch
import torch.nn as nn
import torch.utils.data

__author__ = "Chethan Radhakrishna and Soumick Chatterjee"
__credits__ = ["Chethan Radhakrishna", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Chethan Radhakrishna"
__email__ = "chethan.radhakrishna@st.ovgu.de"
__status__ = "Development"


class Dice(nn.Module):
    """
    Class used to get dice_loss and dice_score
    """

    def __init__(self, smooth=1):
        super(Dice, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred_f = torch.flatten(y_pred)
        y_true_f = torch.flatten(y_true)
        intersection = torch.sum(y_true_f * y_pred_f)
        union = torch.sum(y_true_f + y_pred_f)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score
        return dice_loss, dice_score


class IOU(nn.Module):
    def __init__(self, smooth=1):
        super(IOU, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred_f = torch.flatten(y_pred)
        y_true_f = torch.flatten(y_true)
        intersection = torch.sum(y_true_f * y_pred_f)
        union = torch.sum(y_true_f + y_pred_f) - intersection
        score = (intersection + self.smooth) / (union + self.smooth)
        return score


class FocalTverskyLoss(nn.Module):
    def __init__(self, smooth=1, gamma=0.75, alpha=0.7):
        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        y_true_pos = torch.flatten(y_true)
        y_pred_pos = torch.flatten(y_pred)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        pt_1 = (true_pos + self.smooth) / (
                true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)
        # return pow((1 - pt_1), self.gamma)
        return pow(abs(1 - pt_1), self.gamma)


def get_metric(y_pred, y_true):
    y_true_pos = torch.flatten(y_true)
    y_pred_pos = torch.flatten(y_pred)
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
    intersection = true_pos
    union = torch.sum(y_true_pos + y_pred_pos)
    return true_pos, false_neg, false_pos, intersection, union


def get_losses(logger, true_pos, false_neg, false_pos, intersection, union):
    smooth = 1
    gamma = 0.75
    alpha = 0.7

    dice_score = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice_score
    iou = (intersection + smooth) / (union - intersection + smooth)
    pt_1 = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
    floss = pow((1 - pt_1), gamma)

    logger.info("True Positive:" + str(true_pos) + " False_Negative:" + str(false_neg) + " False_Positive:" +
                str(false_pos))
    logger.info("Floss:" + str(floss) + " diceloss:" + str(dice_loss) + " iou:" + str(iou))
    return floss, dice_loss, iou
