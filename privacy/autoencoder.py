import argparse
import numpy as np
import time
import random
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import math
import dp_optimizer
from torch.nn.utils import clip_grad_norm_

parser = argparse.ArgumentParser()

# experimentName is the current file name without extension
experimentName = os.path.splitext(os.path.basename(__file__))[0]
experimentName = 'rdp'

parser.add_argument("--DATASETPATH", type=str,
                    default=os.path.expanduser('~/data/MIMIC/processed/out_binary.matrix'),
                    help="Dataset file")
parser.add_argument("--n_epochs_pretrain", type=int, default=1,
                    help="number of epochs of pretraining the autoencoder")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--weight_decay", type=float, default=0.00001, help="l2 regularization")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument('--n_iter_D', type=int, default=5, help='number of D iters per each G iter')

# Check the details
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)

parser.add_argument("--cuda", type=bool, default=True,
                    help="CUDA activation")
parser.add_argument("--multiplegpu", type=bool, default=True,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--num_gpu", type=int, default=2, help="Number of GPUs in case of multiple GPU")

parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent noise space")
parser.add_argument("--feature_size", type=int, default=1071, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between batches")
parser.add_argument("--epoch_time_show", type=bool, default=True, help="interval betwen image samples")
parser.add_argument("--epoch_save_model_freq", type=int, default=10, help="number of epops per model save")
parser.add_argument("--minibatch_averaging", type=bool, default=False, help="Minibatch averaging")

#### Privacy
parser.add_argument('--noise_multiplier', type=float, default=1.0)
parser.add_argument('--max_per_sample_grad_norm', type=float, default=1.0)

# Training/Testing
parser.add_argument("--training", type=bool, default=True, help="Training status")
parser.add_argument("--resume", type=bool, default=False, help="Training status")
parser.add_argument("--finetuning", type=bool, default=False, help="Training status")
parser.add_argument("--generate", type=bool, default=False, help="Generating Sythetic Data")
parser.add_argument("--evaluate", type=bool, default=False, help="Evaluation status")
parser.add_argument("--expPATH", type=str, default=os.path.expanduser('~/experiments/pytorch/' + experimentName),
                    help="Experiment path")
parser.add_argument("--modelPATH", type=str, default=os.path.expanduser('~/experiments/pytorch/' + experimentName + '/model'),
                    help="Experiment path")
opt = parser.parse_args()
print(opt)

# Create experiments DIR
if not os.path.exists(opt.expPATH):
    os.system('mkdir -p {0}'.format(opt.expPATH))

# Create models DIR
if not os.path.exists(opt.modelPATH):
    os.system('mkdir -p {0}'.format(opt.modelPATH))

# Random seed for pytorch
opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
cudnn.benchmark = True

# Check cuda
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device BUT it is not in use...")

# Activate CUDA
device = torch.device("cuda:0" if opt.cuda else "cpu")


##########################
### Dataset Processing ###
##########################

data = np.load(os.path.expanduser(opt.DATASETPATH), allow_pickle=True)

sampleSize = data.shape[0]
featureSize = data.shape[1]

# Split train-test
indices = np.random.permutation(sampleSize)
training_idx, test_idx = indices[:int(0.8 * sampleSize)], indices[int(0.8 * sampleSize):]
trainData = data[training_idx, :]
testData = data[test_idx, :]

# Trasnform Object array to float
trainData = trainData.astype(np.float32)
testData = testData.astype(np.float32)

# ave synthetic data
np.save(os.path.join(opt.expPATH, "dataTrain.npy"), trainData, allow_pickle=False)
np.save(os.path.join(opt.expPATH, "dataTest.npy"), testData, allow_pickle=False)


class Dataset:
    def __init__(self, data, transform=None):

        # Transform
        self.transform = transform

        # load data here
        self.data = data
        self.sampleSize = data.shape[0]
        self.featureSize = data.shape[1]

    def return_data(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        sample = np.clip(sample, 0, 1)

        if self.transform:
            pass

        return torch.from_numpy(sample)


# Train data loader
dataset_train_object = Dataset(data=trainData, transform=False)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_train_object, replacement=True)
dataloader_train = DataLoader(dataset_train_object, batch_size=opt.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)

# Test data loader
dataset_test_object = Dataset(data=testData, transform=False)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_test_object, replacement=True)
dataloader_test = DataLoader(dataset_test_object, batch_size=opt.batch_size,
                             shuffle=True, num_workers=0, drop_last=True)

# Generate random samples for test
random_samples = next(iter(dataloader_test))
feature_size = random_samples.size()[1]


####################
### Architecture ###
####################
class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        n_channels_base = 4

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_channels_base, kernel_size=5, stride=2, padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=n_channels_base, out_channels=2 * n_channels_base, kernel_size=5, stride=2, padding=0,
                      dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(2 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=2 * n_channels_base, out_channels=4 * n_channels_base, kernel_size=5, stride=3,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(4 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=4 * n_channels_base, out_channels=8 * n_channels_base, kernel_size=5, stride=3,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(8 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=8 * n_channels_base, out_channels=16 * n_channels_base, kernel_size=5, stride=3,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(16 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=16 * n_channels_base, out_channels=32 * n_channels_base, kernel_size=8, stride=1,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32 * n_channels_base, out_channels=16 * n_channels_base, kernel_size=5,
                               stride=1, padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16 * n_channels_base, out_channels=8 * n_channels_base, kernel_size=5,
                               stride=4, padding=0,
                               dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(8 * n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=8 * n_channels_base, out_channels=4 * n_channels_base, kernel_size=7,
                               stride=4,
                               padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(4 * n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=4 * n_channels_base, out_channels=2 * n_channels_base, kernel_size=7,
                               stride=3,
                               padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(2 * n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=2 * n_channels_base, out_channels=n_channels_base, kernel_size=7, stride=2,
                               padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=n_channels_base, out_channels=1, kernel_size=3, stride=2,
                               padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x.view(-1, 1, x.shape[1]))
        x = self.decoder(x)
        return torch.squeeze(x)

    def decode(self, x):
        x = self.decoder(x)
        return torch.squeeze(x)

###############
### Lossess ###
###############


def autoencoder_loss(x_output, y_target):
    """
    autoencoder_loss
    This implementation is equivalent to the following:
    torch.nn.BCELoss(reduction='sum') / batch_size
    As our matrix is too sparse, first we will take a sum over the features and then do the mean over the batch.
    WARNING: This is NOT equivalent to torch.nn.BCELoss(reduction='mean') as the latter one, mean over both features and batches.
    """
    epsilon = 1e-12
    term = y_target * torch.log(x_output + epsilon) + (1. - y_target) * torch.log(1. - x_output + epsilon)
    loss = torch.mean(-torch.sum(term, 1), 0)
    return loss



def weights_init(m):
    """
    Custom weight initialization.
    NOTE: Bad initialization may lead to dead model and can prevent training!
    :param m: Input argument to extract layer type
    :return: Initialized architecture
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        # nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.2)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


#############
### Model ###
#############

# Initialize generator and discriminator
autoencoderModel = Autoencoder()
autoencoderDecoder = autoencoderModel.decoder

# Define cuda Tensors
# BE careful about torch.FloatTensor([1])!!!!
# I once defined it as torch.FloatTensor(1) without brackets around 1 and everything was messed hiddenly!!
Tensor = torch.FloatTensor
one = torch.FloatTensor([1])
mone = one * -1

if torch.cuda.device_count() > 1 and opt.multiplegpu:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    autoencoderModel = nn.DataParallel(autoencoderModel, list(range(opt.num_gpu)))
    autoencoderDecoder = nn.DataParallel(autoencoderDecoder, list(range(opt.num_gpu)))

if opt.cuda:
    """
    model.cuda() will change the model inplace while input.cuda() 
    will not change input inplace and you need to do input = input.cuda()
    ref: https://discuss.pytorch.org/t/when-the-parameters-are-set-on-cuda-the-backpropagation-doesnt-work/35318
    """
    autoencoderModel.cuda()
    autoencoderDecoder.cuda()
    one, mone = one.cuda(), mone.cuda()
    Tensor = torch.cuda.FloatTensor

# Weight initialization
autoencoderModel.apply(weights_init)

# Optimizers
optimizer_A = dp_optimizer.AdamDP(
        max_per_sample_grad_norm=opt.max_per_sample_grad_norm,
        noise_multiplier=opt.noise_multiplier,
        batch_size=opt.batch_size,
        params=autoencoderModel.parameters(),
        lr=opt.lr,
        betas=(opt.b1, opt.b2),
        weight_decay=0.0001,
    )

################
### TRAINING ###
################

if opt.resume:
    #####################################
    #### Load model and optimizer #######
    #####################################

    # Loading the checkpoint
    checkpoint = torch.load(os.path.join(opt.PATH, "model_epoch_1000.pth"))

    # Load models
    generatorModel.load_state_dict(checkpoint['Generator_state_dict'])
    discriminatorModel.load_state_dict(checkpoint['Discriminator_state_dict'])
    autoencoderModel.load_state_dict(checkpoint['Autoencoder_state_dict'])
    autoencoderDecoder.load_state_dict(checkpoint['Autoencoder_Decoder_state_dict'])

    # Load optimizers
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    optimizer_A.load_state_dict(checkpoint['optimizer_A_state_dict'])

    # Load losses
    g_loss = checkpoint['g_loss']
    d_loss = checkpoint['d_loss']
    a_loss = checkpoint['a_loss']

    # Load epoch number
    epoch = checkpoint['epoch']

    generatorModel.eval()
    discriminatorModel.eval()
    autoencoderModel.eval()
    autoencoderDecoder.eval()

for epoch_pre in range(opt.n_epochs_pretrain):
    for i_batch, samples in enumerate(dataloader_train):

        # Configure input
        real_samples = Variable(samples.type(Tensor))

        # # Reset gradients (if you comment below line, it would be a mess. Think why?!!!!!!!!!)
        optimizer_A.zero_grad()

        # Microbatch processing
        for i in range(opt.batch_size):

            # Extract microbatch
            micro_batch = real_samples[i:i+1,:]

            # Reset grads
            optimizer_A.zero_grad()

            # Generate a batch of images
            recons_samples = autoencoderModel(micro_batch)

            # Loss measures generator's ability to fool the discriminator
            a_loss = autoencoder_loss(recons_samples, micro_batch)

            # Backward
            a_loss.backward()

            # Bound sensitivity
            optimizer_A.clip_grads_()

            ################### Privacy ################

        # Step
        optimizer_A.add_noise_()
        optimizer_A.step()

        batches_done = epoch_pre * len(dataloader_train) + i_batch + 1
        if batches_done % opt.sample_interval == 0:
            print(
                "[Epoch %d/%d of pretraining] [Batch %d/%d] [A loss: %.3f]"
                % (epoch_pre + 1, opt.n_epochs_pretrain, batches_done, len(dataloader_train), a_loss.item())
                , flush=True)

torch.save({
    'Autoencoder_state_dict': autoencoderModel.state_dict(),
    'optimizer_A_state_dict': optimizer_A.state_dict(),
}, os.path.join(opt.modelPATH, "aepretrained.pth"))
