import torch
from utils.utils import AverageMeter, accuracy
import pandas as pd
from .resnet_orig import *
from .resnext import *
import numpy as np

class inference_noise():
    def __init__(self, modelname, checkpoint_dir, loader, K, avg_num, snr_range=[0]):
        self.checkpoint = torch.load(checkpoint_dir)
        self.K = K        
        self.loader = loader
        self.modelname = modelname
        self.avg_num=avg_num
        self.snr_range = snr_range
    
    def set_model(self, snr, conv1_noise, linear_noise):
        # change main model at will
        self.model = globals()[self.modelname](self.K, snr=snr, inference=True, conv1_noise=conv1_noise, linear_noise=linear_noise)
        try:
            self.model.load_state_dict(self.checkpoint)
        except:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in self.checkpoint["model"].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            
            self.model.load_state_dict(new_state_dict)
        self.model.cuda()
    
    def validate(self):
        top1 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(self.loader):
                target = target.long().cuda()
                input_var = input.cuda()
                target_var = target.cuda()

                # compute output
                output = self.model(input_var)
                output = output.float()

                # measure accuracy and record loss
                prec1 = accuracy(output.data, target)[0]
                top1.update(prec1.item(), input.size(0))
        return top1.avg
    
    def val(self, conv1_noise=True, linear_noise=True):
        # single pass inference
        accs = []
        for snr in self.snr_range:
            a = []
            for i in range(self.avg_num):
                self.set_model(snr, conv1_noise=conv1_noise, linear_noise=linear_noise)
                a.append(self.validate())
            acc = np.mean(a)
            accs.append({"acc": acc, "snr": snr, "conv1":conv1_noise, "linear":linear_noise})
        return accs
    
    def val_all(self):
        # inference for all noise options
        results = []
        for conv1 in [True, False]:
            for linear in [True, False]:
                results.extend(self.val(conv1, linear))
        return results