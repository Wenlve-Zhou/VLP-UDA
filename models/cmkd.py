import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.tools import LambdaSheduler

class CMKD(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.lamb = LambdaSheduler(max_iter=args.max_iter)
        self.args = args

    def calibrated_coefficient(self, pred, pred_pretrained):
        distance = F.kl_div(pred.log(), pred_pretrained, reduction='none').sum(-1)
        coe = torch.exp(-distance).detach()
        return coe

    def calibrated_coefficient1(self, pred):
        epsilon = 1e-5
        H = -pred * torch.log(pred + epsilon)
        H = H.sum(dim=1)
        coe = torch.exp(-H).detach()
        return coe

    def gini_impurity(self,pred,coe=1.0):
        sum_dim = torch.sum(pred, dim=0).unsqueeze(dim=0).detach()
        return torch.sum(coe * (1 - torch.sum(pred ** 2 / sum_dim, dim=-1)))

    def regularization_term(self, target_pred_clip, source_logit_clip, source_label,lamb):
        return self.args.lambda2*F.cross_entropy(source_logit_clip, source_label) + \
            self.args.lambda3*lamb*self.gini_impurity(target_pred_clip)

    def forward(self, target_logit, target_logit_clip, source_logit_clip, source_label, label_set=None):
        target_pred = F.softmax(target_logit, dim=1)
        target_pred_clip = F.softmax(target_logit_clip,dim=-1)
        coe = self.calibrated_coefficient(target_pred, target_pred_clip)
        target_pred_mix = 0.5*(target_pred+target_pred_clip.detach())
        lamb = self.lamb.lamb()
        if label_set is not None:
            task_loss = self.args.lambda1 * lamb * self.gini_impurity(target_pred[:,label_set], coe)
            distill_loss = self.args.lambda1 * lamb * self.gini_impurity(target_pred_mix[:,label_set], 1 - coe)
            reg_loss = self.regularization_term(target_pred_clip[:,label_set], source_logit_clip, source_label, lamb)
        else:
            task_loss = self.args.lambda1 * lamb * self.gini_impurity(target_pred,coe)
            distill_loss = self.args.lambda1 * lamb *self.gini_impurity(target_pred_mix,1-coe)
            reg_loss = self.regularization_term(target_pred_clip, source_logit_clip, source_label,lamb)
        self.lamb.step()
        return task_loss + distill_loss + reg_loss

