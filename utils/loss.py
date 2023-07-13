import torch
import torch.nn as nn
import torch.nn.functional as F
from .gather import GatherLayer
from torch.autograd import Variable


class ProbabilityLoss(nn.Module):
    def __init__(self):
        super(ProbabilityLoss, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = nn.KLDivLoss(reduction='sum')

    def forward(self, logits1, logits2):
        assert logits1.size() == logits2.size()
        softmax1 = self.softmax(logits1)
        softmax2 = self.softmax(logits2)

        probability_loss = self.criterion(softmax1.log(), softmax2)
        return probability_loss


class BatchLoss(nn.Module):
    def __init__(self, batch_size, world_size):
        super(BatchLoss, self).__init__()
        self.batch_size = batch_size
        self.world_size = world_size

    def forward(self, activations, ema_activations):
        assert activations.size() == ema_activations.size()
        N = self.batch_size * self.world_size
        # gather data from all the processes
        if self.world_size > 1:
            activations = torch.cat(GatherLayer.apply(activations), dim=0)
            ema_activations = torch.cat(GatherLayer.apply(ema_activations), dim=0)
        # reshape as N*C
        activations = activations.view(N, -1)
        ema_activations = ema_activations.view(N, -1)

        # form N*N similarity matrix
        similarity = activations.mm(activations.t())
        norm = torch.norm(similarity, 2, 1).view(-1, 1)
        similarity = similarity / norm

        ema_similarity = ema_activations.mm(ema_activations.t())
        ema_norm = torch.norm(ema_similarity, 2, 1).view(-1, 1)
        ema_similarity = ema_similarity / ema_norm

        batch_loss = (similarity - ema_similarity) ** 2 / N
        return batch_loss


class ChannelLoss(nn.Module):
    def __init__(self, batch_size, world_size):
        super(ChannelLoss, self).__init__()
        self.batch_size = batch_size
        self.world_size = world_size

    def forward(self, activations, ema_activations):
        assert activations.size() == ema_activations.size()
        N = self.batch_size * self.world_size
        # gather data from all the processes
        if self.world_size > 1:
            activations = torch.cat(GatherLayer.apply(activations), dim=0)
            ema_activations = torch.cat(GatherLayer.apply(ema_activations), dim=0)
        # reshape as N*C
        activations = activations.view(N, -1)
        ema_activations = ema_activations.view(N, -1)

        # form C*C channel-wise similarity matrix
        similarity = activations.t().mm(activations)
        norm = torch.norm(similarity, 2, 1).view(-1, 1)
        similarity = similarity / norm

        ema_similarity = ema_activations.t().mm(ema_activations)
        ema_norm = torch.norm(ema_similarity, 2, 1).view(-1, 1)
        ema_similarity = ema_similarity / ema_norm

        channel_loss = (similarity - ema_similarity) ** 2 / N
        return channel_loss


class GCELoss(nn.Module):
    def __init__(self, num_classes=10, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()


class pNorm(nn.Module):
    def __init__(self, p=0.5):
        super(pNorm, self).__init__()
        self.p = p

    def forward(self, pred, p=None):
        if p:
            self.p = p
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1)
        norm = torch.sum(pred ** self.p, dim=1)
        return norm.mean()


class GCEandRS(nn.Module):
    def __init__(self, num_classes=10, q=0.7, tau=10, p=0.1, lamb=1.2):
        super(GCEandRS, self).__init__()
        self.criterion = GCELoss(num_classes=num_classes, q=q)
        self.tau = tau
        self.p = p
        self.lamb = lamb
        self.norm = pNorm(p=p)

    def forward(self, out, y):
        out = F.normalize(out, dim=1)
        loss = self.criterion(out / self.tau, y) + self.lamb * self.norm(out / self.tau, self.p)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()