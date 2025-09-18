import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Args:
            gamma (float): focusing parameter gamma >= 0
            alpha (Tensor or float, optional): class-wise weight. 
                - Tensor of shape [num_classes] for per-class weight.
                - Float scalar for uniform weight.
            reduction (str): 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: raw logits of shape (batch_size, num_classes)
        targets: binary targets (0 or 1) of same shape
        """
        # Sigmoid to get probabilities
        probas = torch.sigmoid(inputs)
        probas = torch.clamp(probas, min=1e-6, max=1 - 1e-6)  # avoid log(0)
        
        # Focal loss per element
        loss = -(
            targets * (1 - probas) ** self.gamma * torch.log(probas) +
            (1 - targets) * probas ** self.gamma * torch.log(1 - probas)
        )

        # Apply alpha weighting if specified
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            if alpha.ndim == 0:
                loss *= alpha
            else:
                loss *= alpha.unsqueeze(0)  # broadcast

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
