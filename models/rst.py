import copy
import os
import torch

def training(model, args):
    student = model.base_network.model.visual
    teacher = model.teacher_model.model.visual
    threshold = args.rst_threshold
    for (name,param1), param2 in zip(student.named_parameters(),teacher.parameters()):
        mask = torch.abs(param1.data-param2.data) > threshold
        param1.data = mask.float() * param1.data + (~mask).float() * param2.data

def dsp_calculation(model):
    total_params = 0
    student = model.base_network.model.visual
    teacher = model.teacher_model.model.visual
    classifier = model.classifier_layer

    # DSP of Backbone
    for (name,param1), param2 in zip(student.named_parameters(),teacher.parameters()):
        param = param1-param2
        total_params += torch.nonzero(param).size(0)

    # DSP of Classifier
    for name,param in classifier.named_parameters():
        total_params += param.numel()

    return round(total_params/1e6, 4)