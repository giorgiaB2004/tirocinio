import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-5, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] (raw outputs from model)
        targets: [B, H, W] (ground truth class indices)
        """
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        safe_targets = torch.clamp(targets, 0, self.num_classes - 1) #pheno
        targets_one_hot = F.one_hot(safe_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float() #pheno
        #targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1)  # [B, 1, H, W]
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask

        intersection = torch.sum(probs * targets_one_hot, dim=(0, 2, 3))
        union = torch.sum(probs + targets_one_hot, dim=(0, 2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()
    
class DiceCELoss(nn.Module):
    def __init__(self, num_classes, weight=None, dice_weight=1.0, ce_weight=1.0, ignore_index=None):
        super(DiceCELoss, self).__init__()
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, targets):
        loss_dice = self.dice(logits, targets)
        loss_ce = self.ce(logits, targets)
        return self.dice_weight * loss_dice + self.ce_weight * loss_ce
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', ignore_index=-1):
        """
        Args:
            gamma: focusing parameter.
            alpha: class-wise weights (can be a float or tensor of shape [num_classes]).
            reduction: 'mean', 'sum' or 'none'.
            ignore_index: ignore label index.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, inputs, targets):
        """
        inputs: (B, C, H, W) logits
        targets: (B, H, W) ground truth class indices
        """
        if self.alpha is not None and isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha.to(inputs.device)

        logpt = F.log_softmax(inputs, dim=1)  # (B, C, H, W)
        pt = torch.exp(logpt)  # (B, C, H, W)

        # Flatten
        logpt = logpt.permute(0, 2, 3, 1).reshape(-1, inputs.shape[1])  # (N, C)
        pt = pt.permute(0, 2, 3, 1).reshape(-1, inputs.shape[1])
        targets = targets.view(-1)  # (N,)

        if self.ignore_index >= 0:
            valid_mask = targets != self.ignore_index
            logpt = logpt[valid_mask]
            pt = pt[valid_mask]
            targets = targets[valid_mask]

        logpt = logpt.gather(1, targets.unsqueeze(1))  # (N, 1)
        pt = pt.gather(1, targets.unsqueeze(1))  # (N, 1)

        focal_term = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets].unsqueeze(1)  # (N, 1)
            loss = -alpha_t * focal_term * logpt
        else:
            loss = -focal_term * logpt

        loss = loss.squeeze()

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss



class SIGReg(torch.nn.Module):
    def __init__(self, knots=17, num_slices=256):
        super().__init__()
        self.num_slices = num_slices
        
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        device = proj.device
        
        # Se l'input arriva dalla UNet con feature map spaziali [B, C, H, W]
        # dobbiamo portarlo a due dimensioni: [Vettori totali, Feature/Canali]
        if proj.dim() == 4:
            # Sposta i canali alla fine e unisci Batch, Altezza e Larghezza
            proj = proj.permute(0, 2, 3, 1).flatten(0, 2)
        elif proj.dim() == 3: # Se ha forma [B, Sequence_Length, C]
            proj = proj.flatten(0, 1)
            
        N = proj.size(0) # Numero totale di campioni su cui calcolare la statistica
        
        # 2. Generazione della matrice di proiezione casuale sul device corretto
        A = torch.randn(proj.size(-1), self.num_slices, device=device)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-8) 
        
        x_t = (proj @ A).unsqueeze(-1) * self.t.to(device)
        err = (x_t.cos().mean(dim=0) - self.phi).square() + x_t.sin().mean(dim=0).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
    
class CombinedSegmentationLoss(nn.Module):
    def __init__(self, seg_loss_fn, reg_loss_fn, lambda_reg=1e-3):
        super().__init__()
        self.seg_loss_fn = seg_loss_fn
        self.reg_loss_fn = reg_loss_fn
        self.lambda_reg = lambda_reg

    def forward(self, y_pred, bottleneck, y_true):
        loss_seg = self.seg_loss_fn(y_pred, y_true)
        loss_reg = self.reg_loss_fn(bottleneck)
        return loss_seg + (self.lambda_reg * loss_reg)


"""
class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-5, ignore_index=None, weight=None):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.weight = weight # <-- IMPORTANTE: Salva il peso nell'oggetto!

    def forward(self, logits, targets):
        
        #logits: [B, C, H, W] 
        #targets: [B, H, W] 
        
        probs = F.softmax(logits, dim=1)  
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1)  
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask

        intersection = torch.sum(probs * targets_one_hot, dim=(0, 2, 3))
        union = torch.sum(probs + targets_one_hot, dim=(0, 2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        if self.weight is not None:
            w = self.weight.to(logits.device)
            return 1.0 - ((dice * w).sum() / w.sum())
        else:
            return 1.0 - dice.mean()
"""