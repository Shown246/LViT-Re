import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, jaccard_score
import cv2
from torch import nn
import torch.nn.functional as F
import math
from functools import wraps
import warnings
import weakref
from PIL import Image
from numpy import average, dot, linalg
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer

class WeightedBCE(nn.Module):
    """Weighted Binary Cross Entropy loss with configurable positive/negative class weights"""
    
    def __init__(self, weights=[0.4, 0.6]):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit_pixel, truth_pixel):
        # Flatten predictions and ground truth
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert (logit.shape == truth.shape)
        
        # Calculate binary cross entropy
        loss = F.binary_cross_entropy(logit, truth, reduction='none')
        
        # Separate positive and negative samples
        pos = (truth > 0.5).float()
        neg = (truth < 0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        
        # Apply weighted loss normalized by class frequency
        loss = (self.weights[0] * pos * loss / pos_weight + self.weights[1] * neg * loss / neg_weight).sum()
        return loss


class WeightedDiceLoss(nn.Module):
    """Dice loss with pixel-level weighting based on ground truth"""
    
    def __init__(self, weights=[0.5, 0.5]):  # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, logit, truth, smooth=1e-5):
        batch_size = len(logit)
        logit = logit.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert (logit.shape == truth.shape)
        
        p = logit.view(batch_size, -1)
        t = truth.view(batch_size, -1)
        
        # Create weight map based on ground truth values
        w = truth.detach()
        w = w * (self.weights[1] - self.weights[0]) + self.weights[0]
        
        # Apply weights to predictions and targets
        p = w * (p)
        t = w * (t)
        
        # Calculate Dice coefficient
        intersection = (p * t).sum(-1)
        union = (p * p).sum(-1) + (t * t).sum(-1)
        dice = 1 - (2 * intersection + smooth) / (union + smooth)

        loss = dice.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    """Standard Dice loss for binary segmentation"""
    
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, inputs, targets):
        N = targets.size()[0]
        smooth = 1
        
        # Flatten spatial dimensions
        input_flat = inputs.view(N, -1)
        targets_flat = targets.view(N, -1)
        
        # Calculate Dice coefficient per sample
        intersection = input_flat + targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        
        # Return 1 - Dice as loss
        loss = 1 - N_dice_eff.sum() / N
        return loss


class MultiClassDiceLoss(nn.Module):
    """Dice loss for multi-class segmentation (averages across 5 classes)"""
    
    def __init__(self, weight=None, ignore_index=None):
        super(MultiClassDiceLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.dice_loss = WeightedDiceLoss()

    def forward(self, inputs, targets):
        assert inputs.shape == targets.shape, "predict & target shape do not match"
        total_loss = 0
        
        # Calculate Dice loss for each of 5 classes
        for i in range(5):
            dice_loss = self.dice_loss(inputs[:, i], targets[:, i])
            total_loss += dice_loss
            total_loss = total_loss / 5
        return total_loss


class DiceLoss(nn.Module):
    """Dice loss with one-hot encoding and per-class loss calculation"""
    
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        """Convert class indices to one-hot encoding"""
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        """Calculate Dice loss for a single class"""
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        
        # Convert targets to one-hot encoding
        target = self._one_hot_encoder(target)
        
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        
        # Calculate loss for specific classes and overall
        class_wise_dice = []
        loss = 0.0
        dice1 = self._dice_loss(inputs[:, 1], target[:, 1]) * weight[1]
        dice2 = self._dice_loss(inputs[:, 2], target[:, 2]) * weight[2]
        dice3 = self._dice_loss(inputs[:, 3], target[:, 3]) * weight[3]
        
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        
        return loss / self.n_classes, dice1, dice2, dice3


class WeightedDiceCE(nn.Module):
    """Combined Dice and Cross Entropy loss"""
    
    def __init__(self, dice_weight=0.5, CE_weight=0.5):
        super(WeightedDiceCE, self).__init__()
        self.CE_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(4)  # OSIC: 4, RITE: 5
        self.CE_weight = CE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        """Calculate Dice coefficients for evaluation"""
        dice, dice1, dice2, dice3 = self.dice_loss(inputs, targets)
        hard_dice_coeff = 1 - dice
        dice01 = 1 - dice1
        dice02 = 1 - dice2
        dice03 = 1 - dice3
        torch.cuda.empty_cache()
        return hard_dice_coeff, dice01, dice02, dice03

    def forward(self, inputs, targets):
        targets = targets.long()
        dice_CE_loss = self.dice_loss(inputs, targets)
        torch.cuda.empty_cache()
        return dice_CE_loss


class WeightedDiceBCE_unsup(nn.Module):
    """Combined Dice and BCE loss with additional unsupervised LV loss term"""
    
    def __init__(self, dice_weight=1, BCE_weight=1):
        super(WeightedDiceBCE_unsup, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        """Threshold predictions and calculate hard Dice coefficient"""
        inputs[inputs >= 0.5] = 1
        inputs[inputs < 0.5] = 0
        targets[targets > 0] = 1
        targets[targets <= 0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets, LV_loss):
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        # Combine losses with 0.1 weight for LV loss
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE + 0.1 * LV_loss
        return dice_BCE_loss


class WeightedDiceBCE(nn.Module):
    """Combined Dice and BCE loss for binary segmentation"""
    
    def __init__(self, dice_weight=1, BCE_weight=1):
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        """Threshold predictions and calculate hard Dice coefficient"""
        inputs[inputs >= 0.5] = 1
        inputs[inputs < 0.5] = 0
        targets[targets > 0] = 1
        targets[targets <= 0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE
        return dice_BCE_loss


def auc_on_batch(masks, pred):
    """Compute mean AUC score across batch"""
    aucs = []
    for i in range(pred.shape[1]):
        prediction = pred[i][0].cpu().detach().numpy()
        mask = masks[i].cpu().detach().numpy()
        aucs.append(roc_auc_score(mask.reshape(-1), prediction.reshape(-1)))
    return np.mean(aucs)


def iou_on_batch(masks, pred):
    """Compute mean IoU (Jaccard) score across batch"""
    ious = []
    for i in range(pred.shape[0]):
        pred_tmp = pred[i][0].cpu().detach().numpy()
        mask_tmp = masks[i].cpu().detach().numpy()
        
        # Threshold predictions and masks
        pred_tmp[pred_tmp >= 0.5] = 1
        pred_tmp[pred_tmp < 0.5] = 0
        mask_tmp[mask_tmp > 0] = 1
        mask_tmp[mask_tmp <= 0] = 0
        
        ious.append(jaccard_score(mask_tmp.reshape(-1), pred_tmp.reshape(-1)))
    return np.mean(ious)


def dice_coef(y_true, y_pred):
    """Calculate Dice coefficient between two binary masks"""
    smooth = 1e-5
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_on_batch(masks, pred):
    """Compute mean Dice coefficient across batch"""
    dices = []
    for i in range(pred.shape[0]):
        pred_tmp = pred[i][0].cpu().detach().numpy()
        mask_tmp = masks[i].cpu().detach().numpy()
        
        # Threshold predictions and masks
        pred_tmp[pred_tmp >= 0.5] = 1
        pred_tmp[pred_tmp < 0.5] = 0
        mask_tmp[mask_tmp > 0] = 1
        mask_tmp[mask_tmp <= 0] = 0
        
        dices.append(dice_coef(mask_tmp, pred_tmp))
    return np.mean(dices)


def save_on_batch(images1, masks, pred, names, vis_path):
    """Save predicted masks and ground truth masks as images"""
    for i in range(pred.shape[0]):
        pred_tmp = pred[i][0].cpu().detach().numpy()
        mask_tmp = masks[i].cpu().detach().numpy()
        
        # Convert to 0-255 range
        pred_tmp[pred_tmp >= 0.5] = 255
        pred_tmp[pred_tmp < 0.5] = 0
        mask_tmp[mask_tmp > 0] = 255
        mask_tmp[mask_tmp <= 0] = 0

        cv2.imwrite(vis_path + names[i][:-4] + "_pred.jpg", pred_tmp)
        cv2.imwrite(vis_path + names[i][:-4] + "_gt.jpg", mask_tmp)


class _LRScheduler(object):
    """Base class for learning rate schedulers"""

    def __init__(self, optimizer, last_epoch=-1):
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        # Wrap optimizer.step() to track step count
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                return method

            instance_ref = weakref.ref(method.__self__)
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0

        self.step()

    def state_dict(self):
        """Returns scheduler state as dict (excludes optimizer)"""
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads scheduler state from dict"""
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """Return last computed learning rate"""
        return self._last_lr

    def get_lr(self):
        """Compute learning rate (to be implemented by subclasses)"""
        raise NotImplementedError

    def step(self, epoch=None):
        """Update learning rate"""
        # Warn if step order is incorrect
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        # Update learning rates
        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class CosineAnnealingWarmRestarts(_LRScheduler):
    """Cosine annealing scheduler with warm restarts (SGDR)
    
    Learning rate follows cosine curve and periodically restarts to initial value.
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0  # Initial restart period
        self.T_i = T_0  # Current restart period
        self.T_mult = T_mult  # Period multiplier after restart
        self.eta_min = eta_min  # Minimum learning rate

        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)
        self.T_cur = self.last_epoch

    def get_lr(self):
        """Calculate learning rate using cosine annealing formula"""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)

        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Update learning rate and handle warm restarts"""
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            # Increment step counter
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            
            # Check if restart is needed
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            # Manual epoch setting
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    # Calculate which restart cycle we're in
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        # Apply new learning rates
        with _enable_get_lr_call(self):
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


def read_text(filename):
    """Read text descriptions from Excel file and pad short descriptions"""
    df = pd.read_excel(filename)
    text = {}
    for i in df.index.values:
        count = len(df.Description[i].split())
        # Pad descriptions shorter than 9 words
        if count < 9:
            df.Description[i] = df.Description[i] + ' EOF XXX' * (9 - count)
        text[df.Image[i]] = df.Description[i]
    return text  # Dict mapping image names to descriptions


def read_text_LV(filename):
    """Read text descriptions from Excel file (LV variant with longer padding)"""
    df = pd.read_excel(filename)
    text = {}
    for i in df.index.values:
        count = len(df.Description[i].split())
        # Pad descriptions shorter than 30 words
        if count < 30:
            df.Description[i] = df.Description[i] + ' EOF XXX' * (20 - count)
        text[df.Image[i]] = df.Description[i]
    return text


def get_thum(image, size=(224, 224), greyscale=False):
    """Resize image to thumbnail and optionally convert to grayscale"""
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        image = image.convert('L')
    return image


def img_similarity_vectors_via_numpy(image1, image2):
    """Calculate cosine similarity between two images using pixel vectors"""
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    
    # Convert images to vectors and calculate norms
    for image in images:
        vector = []
        for pixel_turple in image.getdata():
            vector.append(average(pixel_turple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    
    # Calculate cosine similarity
    a, b = vectors
    a_norm, b_norm = norms
    res = dot(a / a_norm, b / b_norm)
    return res
