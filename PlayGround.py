import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn import datasets
from torch.distributions import MultivariateNormal

from Flows import VanillaFlow, AffineConstantFlow, SlowMAF, NormalizingFlowModel, ActNorm, RealNVP, CouplingLayer

from Nets import MLP

BATCH_SZ = 128



SEED = 2334
torch.manual_seed(SEED)

##########################         DATASETS    ##############################

# Lightweight datasets

class DatasetMixture:
    """ 4 mixture of gaussians """

    def sample(self, n):
        assert n % 4 == 0
        r = np.r_[np.random.randn(n // 4, 2) * 0.5 + np.array([5, 0]),
                  np.random.randn(n // 4, 2) * 0.5 + np.array([0, 0]),
                  np.random.randn(n // 4, 2) * 0.5 + np.array([5, 5]),
                  np.random.randn(n // 4, 2) * 0.5 + np.array([-10, 5])]
        return torch.from_numpy(r.astype(np.float32))

class DatasetMoons:
    """ two half-moons """
    def sample(self, n):
        moons = datasets.make_moons(n_samples=n, noise=0.05)[0].astype(np.float32)
        return torch.from_numpy(moons)


d = DatasetMoons()

x = d.sample(BATCH_SZ)
plt.figure(figsize=(4, 4))
plt.scatter(x[:, 0], x[:, 1], s=5, alpha=0.5)
plt.axis('equal');
#plt.show()

##########################   FLOW itself   ##############################
netw = MLP(2,2,2)

aff_flow = [AffineConstantFlow(dim=2) for i in range(3)]
coupling_flow = [CouplingLayer(2, 5, i%2==1) for i in range(3)]
maf_flow = [SlowMAF(dim=2, parity=True) for _ in aff_flow]
norms = [ActNorm(dim=2) for _ in coupling_flow]
flows = list(itertools.chain(*zip(aff_flow, maf_flow, coupling_flow, norms)))

#new_flow = [RealNVP(2,2,20,8) for i in range(2)]


# construct the model
model = NormalizingFlowModel(flows)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)  # todo tune WD
print("number of params: ", sum(p.numel() for p in model.parameters()))

model.train()
for k in range(20000):

    x = d.sample(BATCH_SZ)

    # Calculate NN-based prior
    mean = torch.mean(netw(x), dim=0)
    variance = torch.var(netw(x), dim=0)
    prior = MultivariateNormal(mean, torch.diag(variance))

    zs, prior_logprob, log_det = model.forward(x, prior)
    logprob = prior_logprob + log_det
    loss = -torch.sum(logprob)  # NLL

    model.zero_grad()
    loss.backward()
    optimizer.step()

    if k % 3000 == 0:

        print(k,": ", loss.item())

        model.eval()

        x = d.sample(128)

        mean = torch.mean(netw(x), dim=0)
        variance = torch.var(netw(x), dim=0)
        prior = MultivariateNormal(mean, torch.diag(variance))

        zs, prior_logprob, log_det = model(x, prior)
        z = zs[-1]

        reconstr, inv_log_det = model.backward(z)
        r = reconstr[-1].detach().numpy()

        x_np = x.detach().numpy()
        z = z.detach().numpy()
        p = prior.sample([128]).squeeze()
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.scatter(p[:, 0], p[:, 1], c='g', s=5)
        plt.scatter(z[:, 0], z[:, 1], c='r', s=5)
        plt.legend(['prior', 'x->z'])
        plt.axis('scaled')
        plt.title('x -> z')


        x = d.sample(128)

        zs, prior_logprob, log_det = model(x, prior)
        z = zs[-1]

        reconstr, inv_log_det = model.backward(z)

        r = reconstr[-1].detach().numpy()

        det = log_det + inv_log_det
        print("Det check: ", det.sum())

        """

        x = x.detach().numpy()
        z = z.detach().numpy()
        p = model.prior.sample([128])  # .squeeze()
        plt.figure(figsize=(10, 5))
        plt.subplot(121)

        p_x, det = model.backward(p)
        p_x = p_x[0]

        # plt.scatter(p[:,0], p[:,1], c='g', s=5)
        plt.scatter(r[:, 0], r[:, 1], c='g', s=5)
        plt.scatter(p_x[:, 0], p_x[:, 1], c='r', s=5)
        plt.scatter(x[:, 0], x[:, 1], c='b', s=5)

        plt.legend(['reconstr', 'p(z)->x', 'data'])
        plt.axis('scaled')
        plt.title('x -> z')
        plt.show()

        """


        

        zs = model.sample(128 * 8, prior)
        z = zs[-1]
        z = z.detach().numpy()

        plt.subplot(122)
        plt.scatter(x[:, 0], x[:, 1], c='r', s=5, alpha=0.5)
        plt.scatter(z[:, 0], z[:, 1], c='g', s=5, alpha=0.5)
        plt.scatter(r[:, 0], r[:, 1], c='b', s=5, alpha=0.5)

        plt.legend(['data', 'z->x', 'reconstr'])
        plt.axis('scaled')
        plt.title('z -> x')
        plt.show()
        





"""

x = d.sample(128)
        zs, prior_logprob, log_det = model(x)
        z = zs[-1]

        reconstr, inv_log_det = model.backward(z)

        r = reconstr[-1].detach().numpy()

        det = log_det + inv_log_det
        print("Det check: ", det.sum())

        x = x.detach().numpy()
        z = z.detach().numpy()
        p = model.prior.sample([128])  # .squeeze()
        plt.figure(figsize=(10, 5))
        plt.subplot(121)

        p_x, det = model.backward(p)
        p_x = p_x[0]

        # plt.scatter(p[:,0], p[:,1], c='g', s=5)
        plt.scatter(r[:, 0], r[:, 1], c='g', s=5)
        plt.scatter(p_x[:, 0], p_x[:, 1], c='r', s=5)
        plt.scatter(x[:, 0], x[:, 1], c='b', s=5)

        plt.legend(['reconstr', 'p(z)->x', 'data'])
        plt.axis('scaled')
        plt.title('x -> z')
        plt.show()

zs = model.sample(128*8)
z = zs[-1]
z = z.detach().numpy()
plt.subplot(122)
plt.scatter(x[:,0], x[:,1], c='b', s=5, alpha=0.5)
plt.scatter(z[:,0], z[:,1], c='r', s=5, alpha=0.5)
plt.legend(['data', 'z->x'])
plt.axis('scaled')
plt.title('z -> x')
plt.show()
"""