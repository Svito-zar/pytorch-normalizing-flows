from torch import nn
import torch


class MyLeakyReLU(nn.Module):
    """
    Customizable LeakyRelu with learned slopes
    """

    def __init__(self, dim):
        super().__init__()
        self.forward_coeff = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.backward_coeff = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, x):
        z = x*self.forward_coeff*(x>=0) + x*self.backward_coeff*(x<0)

        return z

    def backward(self, z):
        x = z * (z >= 0) / self.forward_coeff  + z * (z < 0) / self.backward_coeff
        return x


    def log_det(self,x):
        return (x <= 0).sum(dim=1) * torch.log(self.backward_coeff) + (x > 0).sum(dim=1) * torch.log(self.forward_coeff)
