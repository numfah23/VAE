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
    train_unlab_set = ckd_Dataset('../scaled_05/train_data_unlab_scaled.csv') 
    train_lab_set = ckd_Dataset('../scaled_05/train_data_lab_scaled.csv') 
    val_set = ckd_Dataset('../scaled_05/val_data_scaled.csv') 
    test_set = ckd_Dataset('../scaled_05/test_data_scaled.csv') 

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

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))

        # return a mean vector and a (positive) square root covariance
        recon_loc = self.fc21(hidden)
        # use softplus instead of exp to avoid overflow in gradients
        recon_scale = 1e-6 + self.softplus(self.fc22(hidden))
        return recon_loc, recon_scale

########################################################################

###################
# encoder network
###################
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
        # shape the mini-batch to be in rightmost dimension
        x = x.reshape(-1, 1335)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        # use softplus instead of exp to avoid overflow in gradients
        z_scale = 1e-6 + self.softplus(self.fc22(hidden))
        return z_loc, z_scale

########################################################################

#######
# VAE
#######

class VAE(nn.Module):
    # 400 hidden units
    # changed z_dim to 1 (severity score is 1 dimenstional)
    def __init__(self, z_dim=1, hidden_dim=10, use_cuda=False):
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

    ###############################
    # define the model p(x|z)p(z)
    ###############################
    def model(self, x, zs=None):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.iarange("data", x.shape[0]):
            # for unlabelled data, train normally
                # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))

            if zs is None:
                # sample from prior (value will be sampled by guide when computing the ELBO)
                z = pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1))
            
            ### for labelled data, observe label!
            else:
                z = pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1), obs=zs)

            # decoder outputs mean and sqroot cov, sample from normal
            recon_loc, recon_scale = self.decoder.forward(z)
            pyro.sample("obs", dist.Normal(recon_loc, recon_scale).independent(1), obs=x.reshape(-1, 1335))
            
    ###########################################################
    # define the guide (i.e. variational distribution) q(z|x)
    ###########################################################
    def guide(self, x, zs=None):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.iarange("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x)

            if zs is None:
                # sample the latent code z
                pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1))

            ### for labelled data, observe label!
            else:
                pyro.sample("latent", dist.Normal(z_loc, z_scale).independent(1), obs=zs)

    #############################################
    # helper function for reconstructing images
    #############################################
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()

        # output recon loc and scale then sample from normal
        recon_loc, recon_scale = self.decoder(z)
        loc = dist.Normal(recon_loc, recon_scale).sample()
        return loc

########################################################################

##################
# train function
##################

# now takes in both unlab loader and lab loader
def train(svi, train_unlab_loader, train_lab_loader, train_lab_labels, use_cuda=False):
    # initialize loss accumulator (separate for unlab and lab)
    epoch_loss_unlab = 0.
    epoch_loss_lab = 0.

    # do a training epoch over each mini-batch x returned by the data loader

    # modified to account for train_loader only having 1 item (also test_loader)
    for x in train_unlab_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # do ELBO gradient and accumulate loss
        epoch_loss_unlab += svi.step(x)

    # added another for loop for labelled data
    count = 0
    for x in train_lab_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        minib = x.shape[0]
        zs = torch.FloatTensor(train_lab_labels[count:count+minib]).reshape(minib,1)
        
        epoch_loss_lab += svi.step(x,zs)
        count += 3
    
    # return epoch loss
    normalizer_train_unlab = len(train_unlab_loader.dataset)
    normalizer_train_lab = len(train_lab_loader.dataset)
    total_epoch_loss_train_unlab = epoch_loss_unlab / normalizer_train_unlab
    total_epoch_loss_train_lab = epoch_loss_lab / normalizer_train_lab
    return total_epoch_loss_train_unlab, total_epoch_loss_train_lab

########################################################################

##################
# train function
##################

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
        test_loss += svi.evaluate_loss(x)
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
NUM_EPOCHS = 50
TEST_FREQUENCY = 5

########################################################################

########
# main
########

train_unlab_loader, train_lab_loader, val_loader, test_loader = ckd_setup_data_loaders(batch_size=3, use_cuda=USE_CUDA)
#read in train data labels (version with -3,3 instead of 0,5)
train_lab_labels = np.genfromtxt('../scaled_05/train_data_lab_labels_z.csv', delimiter=',')

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

train_unlab_elbo = []
train_lab_elbo = []
test_elbo = []

# training loop
for epoch in range(NUM_EPOCHS):
    # add unlab
    total_epoch_loss_train_unlab, total_epoch_loss_train_lab = train(svi, train_unlab_loader, train_lab_loader, train_lab_labels, use_cuda=USE_CUDA)
    train_unlab_elbo.append(-total_epoch_loss_train_unlab)
    train_lab_elbo.append(-total_epoch_loss_train_lab)
    print("[epoch %03d]  average unlabelled training loss: %.4f" % (epoch, total_epoch_loss_train_unlab))
    print("[epoch %03d]  average labelled training loss: %.4f" % (epoch, total_epoch_loss_train_lab))

    if epoch % TEST_FREQUENCY == 0:
        # report test diagnostics
        # try val loader instead
        total_epoch_loss_test = evaluate(svi, val_loader, use_cuda=USE_CUDA)
        test_elbo.append(-total_epoch_loss_test)
        print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))

# plot elbo for training and testing data
plot_elbo(train_unlab_elbo, "train", 1)
plot_elbo(train_lab_elbo, "train", 1)
plot_elbo(test_elbo, "test", TEST_FREQUENCY)

########################################################################
########################################################################
########################################################################
########################################################################

# TODO: restructure testing/val
    # get predictions, then calc acc, then make scatter + boxplots in fns
    # try diff learning rates
    # try diff num iters


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

# predict train lab to check
train_preds = []
for x in train_lab_loader:
    z_loc, z_scale = vae.encoder(x)
    z = dist.Normal(z_loc, z_scale).sample()
    train_preds.append(np.array(z))

train_preds_np = np.concatenate(train_preds).ravel()

plt.scatter(train_lab_labels, train_preds_np)
plt.show()

# #######################################################################################
# plot boxplot for test data predictions
boxplot_data_t = []
possible_labels = range(7)
for l in possible_labels:
    boxplot_data_t.append(test_preds_np[np.where(test_labels_np == l)[0]])

plt.figure()
plt.boxplot(boxplot_data_t)
plt.xticks(range(1,8),["non CKD", "stage I", "stage II", "stage III", "stage IV", "stage V", "CKD NOS"])
plt.ylabel("Predicted severity score")
plt.title("Box plot of test predicted severity vs. actual stage")
plt.show()

#######################################################################################
# plot boxplot for val data predictions
boxplot_data = []
possible_labels = range(7)
for l in possible_labels:
    boxplot_data.append(val_preds_np[np.where(val_labels_np == l)[0]])

plt.figure()
plt.boxplot(boxplot_data)
plt.xticks(range(1,8),["non CKD", "stage I", "stage II", "stage III", "stage IV", "stage V", "CKD NOS"])
plt.ylabel("Predicted severity score")
plt.title("Box plot of val predicted severity vs. actual stage")
plt.show()