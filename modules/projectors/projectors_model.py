from torch import nn
import torch
import os
from configs import config_projectors


class Projectors(nn.Module):
    def __init__(self,
                 dim=1024,
                 config=config_projectors):
        super(Projectors, self).__init__()

        self.projector_facial_expression = nn.Linear(dim, dim)
        self.projector_gesture = nn.Linear(dim, dim)
        self.config = config

    def forward(self, vision):
        facial_expression = self.projector_facial_expression(vision)
        gesture = self.projector_gesture(vision)

        return facial_expression, gesture

    def save_model(self, epoch=None):
        model_path = os.path.join(self.config.MELD.Path.checkpoints_path, 'projectors_' + str(epoch) + '.pth')
        print("Projectors saved at {}".format(model_path))
        torch.save(self.state_dict(), model_path)
