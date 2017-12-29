import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


"""
This is an implementation of fully connected GAN with one hidden layer as defined in

Ian J. Goodfellow， Jean Pouget-Abadie， Mehdi Mirza， Bing Xu， David Warde-Farley， 
            Sherjil Ozair， Aaron Courville & Yoshua Bengio（2014）
Generative Adversarial Networks
arXiv preprint arXiv:1406.2661.
"""



# Parameters
mean = 4
std = 1.25

g_input_size = 1
g_hidden_size = 100
g_output_size = 1
d_input_size = 200
d_hidden_size = 100
d_output_size = 1
minibatch_size = d_input_size
d_step = 10
g_step = 1

d_learning_rate = 2e-4
g_learning_rate = 2e-4
optim_betas = (0.9, 0.999)
num_epochs = 50000
print_interval = 200

(name, preprocess, d_input_func) = ("Data and variances", lambda data: decorate_with_diffs(data, 2.0), lambda x: x * 2)

def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (0, 1)))


def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)

def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]

def decorate_with_diffs(data, exponent):
    mean = torch.mean(data.data, 1, keepdim=True)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)
    return torch.cat([data, diffs], 1)


# Create Model


class Generator(nn.Module):
    # Create a generator

    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))

        return self.fc3(x)

class Discriminator(nn.Module):
    # Create a discriminator

    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))

        return F.sigmoid(self.fc3(x))


# Train and test

d_sampler = get_distribution_sampler(mean, std)
gi_sampler = get_generator_input_sampler()
G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
D = Discriminator(input_size=d_input_func(d_input_size), hidden_size=d_hidden_size, output_size=d_output_size)
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)


for epoch in range(num_epochs):

    for i in range(d_step):
        # Train Discriminator on real and fake data
        D.zero_grad()

        # Train on Real data
        d_real_data = Variable(d_sampler(d_input_size))
        d_real_pred = D(preprocess(d_real_data))
        d_real_loss = criterion(d_real_pred, Variable(torch.ones(1)))
        d_real_loss.backward()

        # Train on Fake data
        d_fake_data = Variable(gi_sampler(minibatch_size, g_input_size))
        d_fake_data = G(d_fake_data).detach()
        d_fake_pred = D(preprocess(d_fake_data))
        d_fake_loss = criterion(d_fake_pred, Variable(torch.zeros(1)))
        d_fake_loss.backward()

        d_optimizer.step()

    for j in range(g_step):
        # Train Generator from Discriminator's responses
        G.zero_grad()

        g_fake_data = G(Variable(gi_sampler(minibatch_size, g_input_size)))
        g_fake_pred = D(preprocess(g_fake_data.t()))
        g_loss = criterion(g_fake_pred, Variable(torch.ones(1)))

        g_loss.backward()
        g_optimizer.step()

    if epoch % print_interval == 0:
        print("%s: D: %s/%s G: %s (Real: %s, Fake: %s) " % (epoch,
                                                            extract(d_real_loss)[0],
                                                            extract(d_fake_loss)[0],
                                                            extract(g_loss)[0],
                                                            stats(extract(d_real_data)),
                                                            stats(extract(d_fake_data))))






































