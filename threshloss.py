import torch
import torch.nn as nn
import sys

class ThreshLoss(nn.Module):
    def __init__(self, version="all", loss_type = "hinge", class_num=None):
        super(ThreshLoss, self).__init__()
        if version == "all":
            self.mask_dict = {0:torch.tensor([1,1,1]),
                        1:torch.tensor([-1,1,1]),
                        2:torch.tensor([-1,-1,1]),
                        3:torch.tensor([-1,-1,-1])}
        elif version == "immediate":
            self.mask_dict = {0:torch.tensor([1,0,0]),
                        1:torch.tensor([-1,1,0]),
                        2:torch.tensor([0,-1,1]),
                        3:torch.tensor([0,0,-1])}
        
        self.MSELoss = nn.MSELoss(reduction="sum")
        self.relu = nn.ReLU()
        self.loss_type = loss_type
        
        #TODO: add class number balance

    def create_mask(self, labels):
        mask = torch.zeros(labels.shape[0], 3)
        for i,l in enumerate(labels):
            mask[i] = self.mask_dict[l.item()]
        return mask
        
    def forward(self, labels, logits, thresh):
        #labels: [64]
        #logits: [64,1]

        m1 = (logits-thresh)*(-1)
        mask  = self.create_mask(labels).to(logits.get_device())
        m2 = m1 * mask

        if self.version == "modifiedLS": 
            loss = torch.sum(torch.pow((1-m2)*(m2<1),2))
        elif self.version == "hinge":
            loss = torch.sum(self.relu(1-m2))
        elif self.version == "smoothed":
            loss = torch.sum((1-m2)*(1-m2)/2*(m2<1)*(m2>0)) + torch.sum((0.5-m2)*(m2<=0))
        else:
            sys.exit()

        return loss