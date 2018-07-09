import pandas as pd
import numpy as np
import torch
import torchvision.datasets
import torchvision.models
import torchvision.transforms


class RBM():


    def __init__(self, num_visible, num_hidden, use_cuda=False):

        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k


        self.weights = torch.randn(num_visible, num_hidden) * 0.1
        self.visible_bias = torch.ones(num_visible) * 0.5
        self.hidden_bias = torch.zeros(num_hidden)



    def sample_hidden(self, visible_probabilities):
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
        hidden_probabilities = torch.sigmoid(hidden_activations)

        return hidden_probabilities , torch.bernoulli(hidden_probabilities)



    def sample_visible(self, hidden_probabilities):
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias
        visible_probabilities = torch.sigmoid(visible_activations)
        return visible_probabilities , torch.bernoulli(visible_probabilities)


    def train(self, v0, vk, ph0, phk):
        self.weights += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.visible_bias += torch.sum((v0 - vk), 0)
        self.hidden_bias += torch.sum((ph0 - phk), 0)





# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_hidden(v0)
        for k in range(10):
            _,hk = rbm.sample_hidden(vk)
            _,vk = rbm.sample_visible(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_hidden(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

       
#testing the RBM
test_loss = 0
s = 0.
for id_user in range(0, nb_users):
    v = training_set[id_user:id_user+1]     
    vt = test_set[id_user:id_user+1]        #target
    if len(vt[vt>=0]) > 0: 
        _,h = rbm.sample_hidden(v)
        _,v = rbm.sample_visible(h )
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print(' loss: '+str(test_loss/s))

test_features = []
for id_user in range(0, nb_users):
    v = training_set[id_user:id_user+1]     
    vt = test_set[id_user:id_user+1]        #target
    if len(vt[vt>=0]) > 0: 
        _,h = rbm.sample_hidden(v)
        _,v = rbm.sample_visible(h )
        test_features.append(vt[vt>=0].numpy())

test_features = np.array(test_features)

train_features = []
for id_user in range(0, nb_users):
    v = training_set[id_user:id_user+1]     
    vt = training_set[id_user:id_user+1]        #target
    if len(vt[vt>=0]) > 0: 
        _,h = rbm.sample_hidden(v)
        _,v = rbm.sample_visible(h )
        train_features.append(vt[vt>=0].numpy())

train_features = np.array(test_features)


    