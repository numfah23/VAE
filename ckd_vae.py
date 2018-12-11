import os

import numpy as np
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# add importing more packages
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

########################################################################

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ

########################################################################

class ckd_Dataset(Dataset): #Combines a dataset and a sampler, and provides iterators over the dataset
    def __init__(self, fname):
    	data_np = np.genfromtxt(fname, delimiter=',')
    	data_torch = Variable(torch.FloatTensor(data_np))
        self.data = data_torch
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# load ckd dataset
def ckd_setup_data_loaders(batch_size=39, use_cuda=False):
	# use scaled data (mean=0, cov=1)
    train_unlab_set = ckd_Dataset('../train_data_unlab_scaled.csv') 
    train_lab_set = ckd_Dataset('../train_data_lab_scaled.csv') 
    val_set = ckd_Dataset('../val_data_scaled.csv') 
    test_set = ckd_Dataset('../test_data_scaled.csv') 

    train_unlab_loader = DataLoader(dataset=train_unlab_set,
        batch_size=batch_size, shuffle=True)
    train_lab_loader = DataLoader(dataset=train_lab_set,
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set,
        batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set,
        batch_size=batch_size, shuffle=False)

    # added val_loader
    return train_unlab_loader, train_lab_loader, val_loader, test_loader

########################################################################

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Decoder, self).__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)

        # replaced all 784 with 1335 (multiple locations)
        self.fc21 = nn.Linear(hidden_dim, 1335)
        # add fc22 for sqrt cov (fc21 is mean vector)
        self.fc22 = nn.Linear(hidden_dim, 1335)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 1335
        # change return value to parameter for the output Normal
        # loc_img = self.sigmoid(self.fc21(hidden))
        # return loc_img

        # return a mean vector and a (positive) square root covariance
        recon_loc = self.fc21(hidden)
        # recon_scale = torch.exp(self.fc22(hidden))

        # try softplus instead of exp
        recon_scale = 1e-6 + self.softplus(self.fc22(hidden))
        return recon_loc, recon_scale

########################################################################

class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Encoder, self).__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(1335, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, 1335)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        # z_scale = torch.exp(self.fc22(hidden))

        # try softplus instead of exp
        z_scale = 1e-6 + self.softplus(self.fc22(hidden))
        return z_loc, z_scale

########################################################################

class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    # changed z_dim to 1 (severity score 1 dim)
    def __init__(self, z_dim=1, hidden_dim=400, use_cuda=False):
        super(VAE, self).__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x, unlab, label):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.iarange("data", x.shape[0]):
            # for unlabelled data, train normally
            if unlab:
                # setup hyperparameters for prior p(z)
                z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
                z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
                # sample from prior (value will be sampled by guide when computing the ELBO)
                z = pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1))
                # decode the latent code z
                # loc_img = self.decoder.forward(z)
                # score against actual images
                # pyro.sample("obs", dist.Bernoulli(loc_img).independent(1), obs=x.reshape(-1, 1335))

                # decoder outputs mean and sqroot cov, sample from normal
                recon_loc, recon_scale = self.decoder.forward(z)
                pyro.sample("obs", dist.Normal(recon_loc, recon_scale).independent(1), obs=x.reshape(-1, 1335))
            # for labelled data, 
            else:
                # check if 0 or 5
                if label == 0:
                    z_loc = torch.Tensor([[-3],[-3],[-3]])
                if label == 5:
                    z_loc = torch.Tensor([[3],[3],[3]])

                z_scale = torch.Tensor([[0.1],[0.1],[0.1]])
                z = pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1))
                recon_loc, recon_scale = self.decoder.forward(z)
                pyro.sample("obs", dist.Normal(recon_loc, recon_scale).independent(1), obs=x.reshape(-1, 1335))

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x, unlab, label):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.iarange("data", x.shape[0]):
            if unlab:
                # use the encoder to get the parameters used to define q(z|x)
                z_loc, z_scale = self.encoder.forward(x)

                # print z_scale
                # sample the latent code z
                pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1))
            else:
                # check if 0 or 5
                if label == 0:
                    z_loc = torch.Tensor([[-3],[-3],[-3]])
                if label == 5:
                    z_loc = torch.Tensor([[3],[3],[3]])
                z_scale = torch.Tensor([[0.1],[0.1],[0.1]])
                pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        # loc_img = self.decoder(z)
        # return loc_img

        # output loc and scale then sample from normal
        recon_loc, recon_scale = self.decoder(z)
        loc = dist.Normal(recon_loc, recon_scale).sample()
        return loc

########################################################################
# train now takes in both unlab loader and lab loader
def train(svi, train_unlab_loader, train_lab_loader, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader

    # train_loader only has 1 item (also test_loader)
    for x in train_unlab_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x, True, None)

    # add another for loop for labelled data
    # keep track of which labels (first 120 are 0, last 120 are 5)
    count = 0
    for x in train_lab_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()

        # if first half, pass in 0 otherwise 5
        if count < 120:
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x, False, 0)
        else:
            epoch_loss += svi.step(x, False, 5)
        count += 1

    # return epoch loss
    normalizer_train = len(train_unlab_loader.dataset) + len(train_lab_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

########################################################################

def evaluate(svi, test_loader, use_cuda=False):
    # initialize loss accumulator
    test_loss = 0.
    # compute the loss over the entire test set
    for x in test_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x, True, None)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test

########################################################################

# plot elbo
def plot_elbo(elbo, data_type, freq):
	plt.plot([freq * i for i in range(len(elbo))], elbo)
	plt.xlabel("Number of Epochs")
	if data_type == "train":
		plt.ylabel("ELBO for Training Data")
	else:
		plt.ylabel("ELBO for Testing Data")
	plt.show()

########################################################################

# Run options
# LEARNING_RATE = 1.0e-3
LEARNING_RATE = 1.0e-6
USE_CUDA = False

# Run only for a single iteration for testing
# NUM_EPOCHS = 1 if smoke_test else 1000
NUM_EPOCHS = 10
TEST_FREQUENCY = 5

########################################################################

train_unlab_loader, train_lab_loader, val_loader, test_loader = ckd_setup_data_loaders(batch_size=3, use_cuda=USE_CUDA)

# clear param store
pyro.clear_param_store()

# setup the VAE
vae = VAE(use_cuda=USE_CUDA)

# setup the optimizer
adam_args = {"lr": LEARNING_RATE}
optimizer = Adam(adam_args)

# setup the inference algorithm
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())
# trace elbo: bbvi to max elbo (don't have to do integral)
# gradient as expectation of another gradient (noisy gradient of elbo)

# reparameterization trick: for backprop over random variable (only need for encoder)

train_elbo = []
test_elbo = []
# training loop
for epoch in range(NUM_EPOCHS):
    total_epoch_loss_train = train(svi, train_unlab_loader, train_lab_loader, use_cuda=USE_CUDA)
    train_elbo.append(-total_epoch_loss_train)
    print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

    if epoch % TEST_FREQUENCY == 0:
        # report test diagnostics
        # try val loader instead
        total_epoch_loss_test = evaluate(svi, val_loader, use_cuda=USE_CUDA)
        test_elbo.append(-total_epoch_loss_test)
        print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))

# plot elbo for training and testing data
plot_elbo(train_elbo, "train", 1)
plot_elbo(test_elbo, "test", TEST_FREQUENCY)

########################################################################
# get severity scores for test data
test_preds = []
for x in test_loader:
    z_loc, z_scale = vae.encoder(x)
    z = dist.Normal(z_loc, z_scale).sample()
    test_preds.append(np.array(z))

test_preds_np = np.concatenate(test_preds).ravel()
test_labels_np = np.genfromtxt('../final_data_labels/test_labels.csv', delimiter=',')

# plot against CKD stage
plt.scatter(test_labels_np, test_preds_np)
plt.show()


# repeat for val data
val_preds = []
for x in val_loader:
    z_loc, z_scale = vae.encoder(x)
    z = dist.Normal(z_loc, z_scale).sample()
    val_preds.append(np.array(z))

val_preds_np = np.concatenate(val_preds).ravel()
# try val instead
val_labels_np = np.genfromtxt('../final_data_labels/val_labels.csv', delimiter=',')

plt.scatter(val_labels_np, val_preds_np)
plt.show()
