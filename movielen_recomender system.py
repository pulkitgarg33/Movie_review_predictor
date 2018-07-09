



import pandas as pd
import numpy as np
import torch
import torchvision.models
import torchvision.transforms




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


#making the architecture of autoencoder
import torch.nn as nn
class Autoencoder(nn.Module):
    def __init__(self , ):
        #the use of the super function is to get all the 
        #functions of the parent class 
        super(Autoencoder , self).__init__()
        self.fc1 = nn.Linear(nb_movies , 20  ) # 1st encoding layer
        self.fc2 = nn.Linear(20 , 10) # 2nd encoding layer
        self.fc3 = nn.Linear(10 , 20) # 1st encoding layer
        self.fc4 = nn.Linear(20 , nb_movies) # 2nd decoding layer
        self.activation = nn.Sigmoid()
        
    def forward (self , x): # x is input vector
        x = self.activation(self.fc1(x)) # returns the encoded vector
        x = self.activation(self.fc2(x)) # returns the 2nd encoded vector
        x = self.activation(self.fc3(x)) #returns first decoded vector
        x = self.fc4(x)
        return x
    
autoencoder = Autoencoder()
criterion = nn.MSELoss() #criterion for the loss functiom, Mean Square Error
from torch import optim
optimizer = optim.RMSprop(autoencoder.parameters() , lr = 0.01 , weight_decay = 0.25)

from torch.autograd import Variable


# Training the Stacked Auto encoder
# Training the SAE
nb_epoch = 50
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input_vector = Variable(training_set[id_user]).unsqueeze(0)
        target = input_vector.clone()
        if torch.sum(target.data > 0) > 0:
            output = autoencoder(input_vector)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
        
# Testing the SAE
test_loss = 0
s = 0.
result = []

for id_user in range(nb_users):
    input_vector = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = autoencoder(input_vector)
        target.require_grad = False
        output[target == 0] = 0
        result.append(list(output.detach().numpy()))
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))

#making the output mtrix of reviews
s = 0.
result = []

for id_user in range(nb_users):
    input_vector = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    output = autoencoder(input_vector)
    target.require_grad = False
    output[target == 0] = 0
    result.append(list(output.detach().numpy()))
    s += 1.
result = np.array(result)

result  = result.reshape(943, 1682)

result = result.astype(int)


#making the predicted rating by an user for a movie

print ("Enter the movie id \n")
movieid = input()

print ("Enter the user id \n")
userid = input()


print("the predicted ratings are :::  ")
print(result[int(userid)  - 1][int(movieid) - 1])
