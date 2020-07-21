import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from maskrcnn_benchmark import _C
EPISILON=1e-6

def softmax_focal_loss_cuda(logits, targets, gamma, alpha):
    num_classes = logits.shape[1] # 2?3? (n,c)
    gamma = gamma
    alpha = alpha
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(0, num_classes, dtype=dtype, device=device).unsqueeze(0)
    # print('softmax logits', logits)
    p = torch.softmax(logits, dim=1)
    # print('softmax_value', p)  
    t = targets.unsqueeze(1)
    # print('softmax tar', t)
    term1 = (1 - p) ** gamma * torch.log(p+EPISILON)
    alpha = torch.tensor([[1-alpha, alpha, alpha]]).float().cuda(logits.get_device()) # keep balance 
    # term2 = p ** gamma * torch.log(1 - p+EPISILON)
    losses = -(t == class_range).float() * term1 * alpha #- ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)
    loss_isnan = torch.isnan(losses)
    assert torch.sum(loss_isnan) == 0, ['softmax loss', losses]
    losses = losses.sum(dim=1, keepdim=True)
    return losses

class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        device = logits.device
        if logits.is_cuda:
            loss_func = softmax_focal_loss_cuda
        else:
            loss_func = softmax_focal_loss_cpu

        loss = loss_func(logits, targets, self.gamma, self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
