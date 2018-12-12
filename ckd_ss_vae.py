import os

import numpy as np
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
# from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# add importing more packages
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# might not need these, but added from ss
import argparse
from visdom import Visdom
from pyro.contrib.examples.util import print_and_log
from pyro.infer import SVI, JitTrace_ELBO, JitTraceEnum_ELBO, Trace_ELBO, TraceEnum_ELBO, config_enumerate
# 
########################################################################

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ

########################################################################

class ckd_Dataset(Dataset):
#Combines a dataset and a sampler, and provides iterators over the dataset
    def __init__(self, fname):
        data_np = np.genfromtxt(fname, delimiter=',')
        data_torch = Variable(torch.FloatTensor(data_np))
        self.data = data_torch
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

####################
# load ckd dataset
####################
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

    # added val_loader, and split train into labelled and unlabelled parts
    return train_unlab_loader, train_lab_loader, val_loader, test_loader

########################################################################

###################
# decoder network
###################
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, input_size):
        super(Decoder, self).__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)

        # replaced all 784 with 1335 (multiple locations)
        self.fc21 = nn.Linear(hidden_dim, input_size)
        # add fc22 for sqrt cov (fc21 is mean vector)
        self.fc22 = nn.Linear(hidden_dim, input_size)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, z):
        x = x.reshape(-1, input_size)

        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))

        # return a mean vector and a (positive) square root covariance
        recon_loc = self.fc21(hidden)
        # use softplus instead of exp to avoid overflow in gradients
        recon_scale = 1e-6 + self.softplus(self.fc22(hidden))
        return recon_loc, recon_scale

########################################################################

#####################
# encoder Z network
#####################
class Encoder_Z(nn.Module):
    def __init__(self, input_size, hidden_dim, z_dim):
        super(Encoder_Z, self).__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the data x
        # shape the mini-batch to be in rightmost dimension
        x = x.reshape(-1, input_size)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        # use softplus instead of exp to avoid overflow in gradients
        z_scale = 1e-6 + self.softplus(self.fc22(hidden))
        return z_loc, z_scale

########################################################################

#####################
# encoder Y network
#####################
class Encoder_Y(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Encoder_Y, self).__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, output_size)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # define the forward computation on the data x
        # shape the mini-batch to be in rightmost dimension
        x = x.reshape(-1, input_size)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a vector of probs used as parameters for
            # sampling from a categorical distribution
        # each of size batch_size x output_size
        return self.sigmoid(self.fc21(hidden))

########################################################################

#######################
# semi supervised VAE
#######################

class SSVAE(nn.Module):
    # added output_size for the dimensions of y (ckd stages 0 to 5)
    # also added input_size for the dimensions of x (lab values)
    def __init__(self, z_dim=1, hidden_dim=400, use_cuda=False, output_size=6, input_size=1335):
        super(SSVAE, self).__init__()
        # create the encoder and decoder networks
        self.encoder_y = Encoder_Y(input_size, hidden_dim, output_size)
        self.encoder_z = Encoder_Z(input_size+output_size, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim+output_size, hidden_dim, input_size)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.output_size = output_size

    ##############################
    # define the model p(x|y, z)
    ##############################
    def model(self, xs, ys=None):
        # can pass in ys as labels or not pass in anything for unlabelled
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data"):
            # setup hyperparameters for prior p(z)
            z_loc = xs.new_zeros(torch.Size((xs.shape[0], self.z_dim)))
            z_scale = xs.new_ones(torch.Size((xs.shape[0], self.z_dim)))

            # sample from prior (value will be sampled by guide when computing the ELBO)
            zs = pyro.sample("z", dist.Normal(z_loc, z_scale).independent(1))

            # if there is a label y, sample from the constant prior
                # otherwise, observe the value (i.e. score against constant prior)
            alpha_prior = xs.new_ones(torch.Size((xs.shape[0], self.output_size))) / (1.0*self.output_size)
            ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=ys)


            # decoder outputs mean and sqroot cov, sample from normal
            recon_loc, recon_scale = self.decoder.forward([zs, yz])
            pyro.sample("x", dist.Normal(recon_loc, recon_scale).independent(1), obs=xs.reshape(-1, 1335))
            
    ########################################################################
    # define the guide (i.e. variational distribution) q(y|x) and q(z|x,y)
    ########################################################################
    def guide(self, xs, ys=None):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder_y", self.encoder_y)
        pyro.module("encoder_z", self.encoder_z)
        with pyro.plate("data"):
            # if no label y, sample and score digit with q(y|x) = cat(alpha(x))
            if ys is None:
                alpha = self.encoder_y.forward(xs)
                ys = pyro.sample("y", dist.OneHotCategorical(alpha))

            # sample and score z with variational distribution
                # q(z|x,y) = normal(loc(x,y), scale(x,y))
            
            z_loc, z_scale = self.encoder.forward([xs, ys])
            pyro.sample("z", dist.Normal(z_loc, z_scale).independent(1))
    
    ########################################################################

    def classifier(self, xs):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder_y", self.encoder_y)
        with pyro.plate("data"):
            alpha = self.encoder_y.forward(xs)
            # get index with max predicted class prob
            res, ind = torch.topk(alpha, 1)

            # convert the index to one-hot tensor label
            ys = xs.new_zeros(alpha.size())
            ys = ys.scatter_(1, ind, 1.0)
            return ys

    ########################################################################
    
    # auxilary (supervised) loss from Kingma 2014 paper
    def model_classify(self, xs, ys=None):        
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder_y", self.encoder_y)
        with pyro.plate("data"):
            # this here is the extra term to yield an auxiliary loss that we do gradient descent on
            if ys is not None:
                alpha = self.encoder_y.forward(xs)
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample("y_aux", dist.OneHotCategorical(alpha), obs=ys)

    def guide_classify(self, xs, ys=None):
        # dummy guide function to accompany model_classify in inference
        pass

########################################################################
########################################################################
### TODO: update to semi-supervised below this line 
########################################################################
########################################################################

##################
# train function
##################

def train(svi, train_unlab_loader, train_lab_loader, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned by the data loader

    # modified to account for train_loader only having 1 item (also test_loader)
    for x in train_unlab_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x, True, None)

    # added another for loop for labelled data
    # keep track of which labels (*first 120 are 0, last 120 are 5*)
    count = 0
    for x in train_lab_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()

        # if first half, pass in 0 into step otherwise 5
        if count < 120:
            epoch_loss += svi.step(x, False, 0)
        else:
            epoch_loss += svi.step(x, False, 5)
        count += 1

    # return epoch loss
    normalizer_train = len(train_unlab_loader.dataset) + len(train_lab_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

########################################################################

#####################
# evaluate function
#####################

# this fn will not update the parameters
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

##############################
# function for plotting elbo
##############################

def plot_elbo(elbo, data_type, freq):
    plt.plot([freq * i for i in range(len(elbo))], elbo)
    plt.xlabel("Number of Epochs")
    if data_type == "train":
        plt.ylabel("ELBO for Training Data")
    else:
        plt.ylabel("ELBO for Testing Data")
    plt.show()

########################################################################

##################
# set parameters
##################

# Run options
# LEARNING_RATE = 1.0e-3
LEARNING_RATE = 1.0e-6
USE_CUDA = False

# Run only for a single iteration for testing
# NUM_EPOCHS = 1 if smoke_test else 1000
NUM_EPOCHS = 10
TEST_FREQUENCY = 5

########################################################################

########
# main
########

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


