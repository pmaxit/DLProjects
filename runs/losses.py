import torch


class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing = 0, num_classes=2, ignore_value = 1):
        super().__init__()
        self.confidence = 1 - label_smoothing
        self.smoothing = label_smoothing
        self.num_classes = num_classes
        self.ignore_value = ignore_value
        
    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing/ (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        # calculate mask 
        mask = (target != self.ignore_value).unsqueeze(1)
    
        # mask for input
        pred = pred * mask
        
        # finding out the cross entropy SUM(Q(x) * log P(X))
        return torch.mean(torch.sum(-true_dist * F.log_softmax(pred,dim=-1) * mask, dim = -1))

        
class LabelSmoothingLoss2(torch.nn.Module):
    def __init__(self, label_smoothing: float = 0.1, reduction="mean", weight=None, ignore_value=1):
        super().__init__()
        self.epsilon = label_smoothing
        self.reduction = reduction
        self.weight = weight
        self.ignore_value = ignore_value

    def reduce_loss(self, loss):
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def linear_combination(self, x, y):
        return self.epsilon * x + (1 - self.epsilon) * y

    def forward(self, preds, target):
        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        if self.training:
            n = preds.size(-1)
            log_preds = F.log_softmax(preds, dim=-1)
            loss = self.reduce_loss(-log_preds.sum(dim=-1))
            nll = F.nll_loss(
                log_preds, target, reduction=self.reduction, weight=self.weight, ignore_index=self.ignore_value
            )
            
            
            return self.linear_combination(loss / n, nll)
        else:
            return torch.nn.functional.cross_entropy(preds, target, weight=self.weight)