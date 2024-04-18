import torch.nn as nn
import numpy as np
import os
import copy
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


class LambdaSheduler(nn.Module):
    def __init__(self, gamma=1.0, max_iter=1000, **kwargs):
        super(LambdaSheduler, self).__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
        return lamb

    def step(self):
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)

def save_model(model,args):
    base_network = copy.deepcopy(model.base_network.model.visual)
    task_head = copy.deepcopy(model.classifier_layer)
    if args.rst:
        teacher_base_network = model.teacher_model.model.visual
        sparse_checkpoint = {}
        for (name, param), param_ in zip(base_network.named_parameters(), teacher_base_network.parameters()):
            param = param - param_
            param = param.to_sparse()
            sparse_checkpoint[name] = param

        path = os.path.join(args.log_dir,f"sparse_{args.model_name}.pt")

        torch.save({
            'backbone_state_dict': sparse_checkpoint,
            'head_state_dict': task_head.state_dict(),
        }, path)
    else:

        path = os.path.join(args.log_dir, f"{args.model_name}.pt")

        torch.save({
            'backbone_state_dict': base_network.state_dict(),
            'head_state_dict': task_head.state_dict(),
        }, path)

def load_checkpoint(model, args):
    model = model.cpu()
    if args.rst:
        checkpoint_dir= os.path.join(args.log_dir, f"sparse_{args.model_name}.pt")
        checkpoints = torch.load(checkpoint_dir,map_location="cpu")
        for param,res_param in zip(model.base_network.model.visual.parameters(),checkpoints["backbone_state_dict"].values()):
            res_param = res_param.to_dense()
            param.data += res_param.data
        model.classifier_layer.load_state_dict(checkpoints["head_state_dict"])
    else:
        checkpoint_dir = os.path.join(args.log_dir, f"{args.model_name}.pt")
        checkpoints = torch.load(checkpoint_dir, map_location="cpu")
        model.base_network.model.visual.load_state_dict(checkpoints["backbone_state_dict"])
        model.classifier_layer.load_state_dict(checkpoints["head_state_dict"])
    return model.to(args.device)