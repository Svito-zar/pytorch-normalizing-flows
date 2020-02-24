import torch
import torch.nn.functional as F
from torch import nn

from Nets import LeafParam, MLP
from Tools import MyLeakyReLU
import math

from torch.distributions import MultivariateNormal


class VanillaFlow(nn.Module):
    """
    Using Neural Networks as is for constructing the Flow
    + LR decomposition for tractability
    """

    def __init__(self, dim):
        super().__init__()
        self.matrix = nn.Parameter(torch.eye(dim, dim), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.activation = MyLeakyReLU(dim)

    def forward(self, x):
        tria_matrix = torch.triu(self.matrix)

        z = self.activation.forward(F.linear(x, tria_matrix) + self.bias)

        log_det = torch.torch.slogdet(tria_matrix).logabsdet + self.activation.log_det(self.activation.backward(z))

        return z, log_det

    def backward(self, z):
        tria_matrix = torch.triu(self.matrix)

        z_min_bias = torch.transpose(self.activation.backward(z) - self.bias, 0, 1)
        x_hat = torch.triangular_solve(z_min_bias, tria_matrix, upper=True)[0]
        x = torch.transpose(x_hat, 0, 1)

        log_det = -torch.torch.slogdet(tria_matrix).logabsdet - self.activation.log_det(self.activation.backward(z))

        return x, log_det


class AffineConstantFlow(nn.Module):
    """
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """

    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None

    def forward(self, x):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class CouplingLayer(nn.Module):
    """Coupling layer in RealNVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the `s` and `t` network.
        num_blocks (int): Number of residual blocks in the `s` and `t` network.
        reverse_mask (bool): Whether to reverse the mask. Useful for alternating masks.
    """

    def __init__(self, in_features, mid_channels, reverse_mask):
        super(CouplingLayer, self).__init__()

        # Save mask info
        self.reverse_mask = reverse_mask

        # Build scale and translate network
        in_features //= 2

        self.st_net = MLP(in_features, in_features * 2, mid_channels)

        # Learnable scale for s
        self.rescale = nn.utils.weight_norm(Rescale(in_features))

    def forward(self, x):

        # Channel-wise mask
        if self.reverse_mask:
            x_id, x_change = x.chunk(2, dim=1)
        else:
            x_change, x_id = x.chunk(2, dim=1)

        #import pdb; pdb.set_trace()

        st = self.st_net(x_id)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s)
        # ToDo: fix rescale
        # s = self.rescale(torch.tanh(s))

        # Scale and translate
        exp_s = s.exp()
        if torch.isnan(exp_s).any():
            raise RuntimeError('Scale factor has NaN entries')
        x_change = (x_change + t) * exp_s

        # Add log-determinant of the Jacobian
        log_det = s.view(s.size(0), -1).sum(-1)

        if self.reverse_mask:
            x = torch.cat((x_id, x_change), dim=1)
        else:
            x = torch.cat((x_change, x_id), dim=1)

        return x, log_det

    def backward(self, z):

        # Channel-wise mask
        if self.reverse_mask:
            z_id, z_change = z.chunk(2, dim=1)
        else:
            z_change, z_id = z.chunk(2, dim=1)

        st = self.st_net(z_id)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s)

        inv_exp_s = s.mul(-1).exp()
        if torch.isnan(inv_exp_s).any():
            raise RuntimeError('Scale factor has NaN entries')
        z_change = z_change * inv_exp_s - t

        if self.reverse_mask:
            x = torch.cat((z_id, z_change), dim=1)
        else:
            x = torch.cat((z_change, z_id), dim=1)

        # Add log-determinant of the Jacobian
        log_det = - s.view(s.size(0), -1).sum(-1)

        return x, log_det


class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.

    Args:
        num_channels (int): Number of channels in the input.
    """

    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1))

    def forward(self, x):
        x = self.weight * x
        return x


class Invertible1x1Conv(nn.Module):
    """
    As introduced in Glow paper.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.P = P  # remains fixed during optimization
        self.L = nn.Parameter(L)  # lower triangular portion
        self.S = nn.Parameter(U.diag())  # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1))  # "crop out" diagonal, stored in S

    def _assemble_W(self):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x):
        W = self._assemble_W()
        z = x @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def backward(self, z):
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        x = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return x, log_det


class SlowMAF(nn.Module):
    """
    Masked Autoregressive Flow, slow version with explicit networks per dim
    """

    def __init__(self, dim, parity, net_class=MLP, nh=24):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleDict()
        self.layers[str(0)] = LeafParam(2)
        for i in range(1, dim):
            self.layers[str(i)] = net_class(i, 2, nh)
        self.order = list(range(dim)) if parity else list(range(dim))[::-1]

    def forward(self, x):
        z = torch.zeros_like(x)
        log_det = torch.zeros(x.size(0))
        for i in range(self.dim):
            st = self.layers[str(i)](x[:, :i])
            s, t = st[:, 0], st[:, 1]
            z[:, self.order[i]] = x[:, i] * torch.exp(s) + t
            log_det += s
        return z, log_det

    def backward(self, z):
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.size(0))
        for i in range(self.dim):
            st = self.layers[str(i)](x[:, :i])
            s, t = st[:, 0], st[:, 1]
            x[:, i] = (z[:, self.order[i]] - t) * torch.exp(-s)
            log_det += -s
        return x, log_det


class ActNorm(AffineConstantFlow):
    """
    Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False

    def forward(self, x):
        # first batch is used for init
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None  # for now
            self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
            self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done = True
        return super().forward(x)


class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m).to(x.get_device())
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m).to(z.get_device())
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det


class CondPrior(nn.Module):
    """ A conditioned prior which is defined by a different mean and variance for each example in a batch """

    def __init__(self, means, variances, device):
        super().__init__()
        self.means = means.to(device)
        self.variances = variances.to(device)
        self.batch_size, self.dim = means.shape
        self.device = device

    def log_prob(self, x):
        """Returns the log-probability of `data` given  parameters `sigma` and `mu`
        """

        target = x.float()

        out_shape = self.variances.shape[-1]
        const = .5 * out_shape * math.log(2 * math.pi)

        log_gauss = - const - torch.sum(((target - self.means) / self.variances) ** 2 / 2
                                        + torch.log(self.variances), [1])

        """ alt_prior = MultivariateNormal(torch.zeros(2), torch.diag(torch.ones(2)))

        log_gauss = alt_prior.log_prob(x).view(x.size(0), -1).sum(1)"""

        return log_gauss

    def sample(self, number_of_samples):
        N = number_of_samples[0]

        sampl_sz = min(self.batch_size, N)

        epsilon = torch.randn((sampl_sz, self.dim)).to(self.device)
        curr_means = self.means[:sampl_sz]
        curr_sigma = self.variances[:sampl_sz]

        sample = curr_means + curr_sigma * epsilon

        """ 
        alt_prior = MultivariateNormal(torch.zeros(2), torch.diag(torch.ones(2)))

        sample = alt_prior.sample((N,))

        """

        return sample


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, flows):
        super().__init__()
        self.flow = NormalizingFlow(flows)

    def forward(self, x, prior):
        zs, log_det = self.flow.forward(x)
        prior_logprob = prior.log_prob(zs[-1])  # .view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z):
        xs, log_det = self.flow.backward(z)
        return xs, log_det

    def sample(self, num_samples, prior):
        z = prior.sample((num_samples,))
        xs, _ = self.flow.backward(z)
        return xs
