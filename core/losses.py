#!/usr/bin/env python
"""
Simplified Loss Functions for Gut Tissue Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks"""
    def __init__(self, smooth=1e-5, ignore_index=-100):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        
    def forward(self, predicted, target):
        """
        Computes Dice Loss
        Args:
            predicted: tensor of shape [B, C, H, W] (logits)
            target: tensor of shape [B, H, W] with class indices
        """
        # Get shape information
        batch_size = predicted.size(0)
        n_classes = predicted.size(1)
        
        # Handle out-of-range indices in target
        if target.max() >= n_classes:
            target = torch.clamp(target, 0, n_classes - 1)
        
        # Ensure target is long type
        target = target.long().to(predicted.device)
        
        # Convert predictions to probabilities
        probs = F.softmax(predicted, dim=1)
        
        # Calculate Dice coefficient for each class
        dice_score = 0.0
        valid_classes = 0
        
        for cls in range(n_classes):
            # Create binary masks for this class
            pred_mask = probs[:, cls, ...].contiguous().view(batch_size, -1)
            target_mask = (target == cls).contiguous().view(batch_size, -1).float()
            
            # Skip if class not present in target
            if target_mask.sum() == 0:
                continue
            
            # Calculate intersection and union
            intersection = (pred_mask * target_mask).sum(dim=1)
            pred_sum = pred_mask.sum(dim=1)
            target_sum = target_mask.sum(dim=1)
            
            # Calculate Dice coefficient
            dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            dice_score += dice.mean()
            valid_classes += 1
        
        # Average over valid classes
        if valid_classes > 0:
            dice_score /= valid_classes
        else:
            return torch.tensor(0.0, device=predicted.device, requires_grad=True)
        
        # Return Dice loss (1 - Dice coefficient)
        return 1.0 - dice_score


class IoULoss(nn.Module):
    """Intersection over Union (IoU) Loss for segmentation tasks"""
    def __init__(self, smooth=1e-5, ignore_index=-100):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        
    def forward(self, predicted, target):
        """
        Computes IoU Loss
        Args:
            predicted: tensor of shape [B, C, H, W] (logits)
            target: tensor of shape [B, H, W] with class indices
        """
        # Get shape information
        batch_size = predicted.size(0)
        n_classes = predicted.size(1)
        
        # Handle out-of-range indices in target
        if target.max() >= n_classes:
            target = torch.clamp(target, 0, n_classes - 1)
        
        # Ensure target is long type
        target = target.long().to(predicted.device)
        
        # Convert predictions to probabilities
        probs = F.softmax(predicted, dim=1)
        
        # Calculate IoU for each class
        iou_score = 0.0
        valid_classes = 0
        
        for cls in range(n_classes):
            # Create binary masks for this class
            pred_mask = probs[:, cls, ...].contiguous().view(batch_size, -1)
            target_mask = (target == cls).contiguous().view(batch_size, -1).float()
            
            # Skip if class not present in target
            if target_mask.sum() == 0:
                continue
            
            # Calculate intersection and union
            intersection = (pred_mask * target_mask).sum(dim=1)
            pred_sum = pred_mask.sum(dim=1)
            target_sum = target_mask.sum(dim=1)
            union = pred_sum + target_sum - intersection
            
            # Calculate IoU
            iou = (intersection + self.smooth) / (union + self.smooth)
            iou_score += iou.mean()
            valid_classes += 1
        
        # Average over valid classes
        if valid_classes > 0:
            iou_score /= valid_classes
        else:
            return torch.tensor(0.0, device=predicted.device, requires_grad=True)
        
        # Return IoU loss (1 - IoU coefficient)
        return 1.0 - iou_score

