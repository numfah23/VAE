# VAE
Variational Autoencoder to predict severity score for CKD patients

ckd_vae.py: unsupervised pyro vae (http://pyro.ai/examples/vae.html) that predicts latent severity score z
ckd_ss_vae.py semi-supervised M2 model (http://pyro.ai/examples/ss-vae.html) following Kingma 2014 (https://arxiv.org/pdf/1406.5298.pdf) that predicts class y corresponding to ckd stage
ckd_vae_z.py: semi-supervised model that predicts latent severity score z (semi-supervised labels are also for latent z)