import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import MultivariateNormal
from torchvision import datasets, transforms

#from skimage.transform import downscale_local_mean
from skimage import io, transform

from Flows import (
    AffineConstantFlow,
    SlowMAF,
    NormalizingFlowModel,
    ActNorm,
    CouplingLayer,
)

from Flows import CondPrior as my_prior

from Nets import MLP

batch_size = 512
num_epochs = 10
dim = 64

SEED = 2334
torch.manual_seed(SEED)

torch.set_default_tensor_type(torch.DoubleTensor)

def plot_samples(ground_truth, produced_images, epoch):
    """
    Plot 3 different pictures: ground truth digits
    alongside with the sample from the model for that digit

    Args:
        ground_truth:     ground truth images from the MNIST dataset
        produced_images:  digits sampled from the MDN
        epoch:            at which epoch are we visualizing

    Returns:
        nothing, saves plots in the 'figures' folder

    """

    fig = plt.figure()
    # Make two raws with three image in each
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(ground_truth[0][0], cmap="gray")
    plt.title("Original")
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(ground_truth[1][0], cmap="gray")
    plt.title("Original")
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(ground_truth[2][0], cmap="gray")
    plt.title("Original")
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(produced_images[0].reshape((8, 8)), cmap="gray")
    plt.title("Sampled")
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(produced_images[1].reshape((8, 8)), cmap="gray")
    plt.title("Sampled")
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(produced_images[2].reshape((8, 8)), cmap="gray")
    plt.title("Sampled")

    plt.savefig("figures/Results_{}epoch.png".format(epoch))

    plt.close()


def plot_rdf(ground_truth, means, variances, epoch):
    """
    Plot all means and variances for a given example

    Args:
        ground_truth:  digit reference from the ground truth
        means:         means of all the components of GMM
        variances:     variances of all the components of GMM
        weights:       weights of all the components of GMM
        num_mix:       number of mixtures in GMM
        epoch:         at which epoch of training are we

    Returns:

    """

    num_mix = 1

    # Plot reference digit
    fig, ax = plt.subplots(3, num_mix, )

    # Plot all the parameters for each component of GMM
    for comp in range(num_mix):
        ax[0, comp].imshow(ground_truth[0], cmap="gray", extent=(-56, 56, -56, 56))
        ax[0, comp].set_title("Original")

        # mean
        ax[1, comp].imshow(
            means[comp].reshape((28, 28)), cmap="gray", extent=(-56, 56, -56, 56)
        )
        ax[1, comp].set_title("Mean")

        # mean
        ax[2, comp].imshow(
            variances[comp].reshape((28, 28)), cmap="gray", extent=(-56, 56, -56, 56)
        )
        ax[2, comp].set_title("Variance")

    plt.savefig("figures/pdf_{}epoch.png".format(epoch))


def conditioning(input, netw):
    conditioning = netw(input)
    means, log_sigma = conditioning.chunk(2, dim=1)

    sigma = torch.exp(log_sigma) + 1e-10

    return means, sigma


#### Transform

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self):
        self.init = True

    def __call__(self, sample):
        image = sample.data.numpy()

        sub_sample = sample

        sub_sampled_image = transform.resize(image[0], (8, 8))

        sub_sample.data = torch.tensor(sub_sampled_image).unsqueeze(0)

        return sub_sample #{'image': img, 'label': landmarks}

##########################         DATASETS    ##############################

# get the dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
             Rescale()]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), Rescale()]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

##########################   FLOW itself   ##############################

device = "cuda" if torch.cuda.is_available() else "cpu"

netw = MLP(1, dim * 2, 64).to(device)

N = 12

aff_flow = [AffineConstantFlow(dim=dim) for i in range(N)]
coupling_flow = [CouplingLayer(dim, 64, i % 2 == 1) for i in range(N-6)]
maf_flow = [SlowMAF(dim=dim, parity=True) for i in range(1)]
norms = [ActNorm(dim=dim) for _ in coupling_flow]
flows = list(itertools.chain(*zip(aff_flow, norms, coupling_flow, maf_flow )))

# construct the model
model = NormalizingFlowModel(flows, device).to(device)

#mean = torch.zeros(batch_size, dim)
#variance = torch.ones(batch_size, dim)
#prior = MultivariateNormal(mean, torch.diag(variance))
#prior = my_prior(mean, variance, "cpu")

# optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-8)  # todo tune WD
print("number of params: ", sum(p.numel() for p in model.parameters()))

model.train()

#########################       TRAIN       ########################################


losses = []

# train the model
for epoch in range(num_epochs):

    for batch_idx, (labels, minibatch) in enumerate(train_loader):

        # Take care of the full batches
        if labels.shape[0] < batch_size:
            continue

        gt = labels[0:3]

        labels = labels.reshape(batch_size, dim).to(device)
        digits = minibatch.unsqueeze(1).double().to(device)

        model.zero_grad()

        # Calculate NN-based prior
        means, variances = conditioning(digits, netw)
        prior = my_prior(means, variances, device)

        prior.log_prob(labels)

        # prior = MultivariateNormal(mean, torch.diag(variance))

        """mean = torch.zeros(dim)
            variance = torch.eye(dim)
            prior = MultivariateNormal(mean, variance)"""

        zs, prior_logprob, log_det = model.forward(labels, prior)
        logprob = prior_logprob + log_det

        # DEBUG
        max_prob = logprob.max()
        min_prob = logprob.min()
        print("Log likelihood is in the range: [", min_prob.item(), ":", max_prob.item(),"]")

        loss = -torch.sum(logprob)  # NLL

        model.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(netw.parameters(), 0.5)

        optimizer.step()

        print(
            "Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(minibatch),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item(),
            )
        )

        if batch_idx > 200 or epoch > 0:
            losses.append(loss.item())

        # Visualize
        if batch_idx == 0:
            digits = minibatch[:10].unsqueeze(1).double().to(device)

            # Calculate NN-based prior
            means, variances = conditioning(digits, netw)
            variances = variances * 0.01
            prior = my_prior(means, variances, device)

            # prior = MultivariateNormal(mean, torch.diag(variance))

            x = labels[:batch_size] #10

            zs, prior_logprob, log_det = model(x, prior)
            z = zs[-1]

            reconstr, inv_log_det = model.backward(z)

            r = reconstr[-1].detach().numpy()

            det = log_det + inv_log_det
            print("Det check: ", det.sum())

            plot_samples(gt, r[:3], epoch+100)

            zs = model.sample(128, prior)
            z = zs[-1]
            images = z.cpu().detach().numpy()

            plot_samples(gt, images[:3], epoch)

            """print(
                "Variance is in range [",
                variances.min().cpu().detach().numpy(),
                ":",
                variances.max().cpu().detach().numpy(),
                "]",
            )"""

            """# PLOT PDF

                mu, _,_ = model(mean.unsqueeze(0), prior)
                mu = mu[-1].detach().numpy()
                sigma,_,_ = model(variance, prior)
                sigma = sigma[-1]

                plot_samples(gt, mu[:3], epoch+100)

                #plot_rdf(gt[0], mu[0], sigma[0], epoch)"""

plt.plot(losses)

plt.show()

print(
    "The training is compelete! \nVisualization results have been saved in the folder 'figures'"
)

"""if plot_pdf:
                    mu = pca.inverse_transform(mu.detach().numpy())
                    sigma = pca.inverse_transform(sigma.detach().numpy())

                    plot_rdf(gt_f[0],mu[0],sigma[0],pi[0],numb_mix_densities,epoch)
                """
