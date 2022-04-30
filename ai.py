# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import codecs, json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.autograd as autograd
from torch.autograd import Variable
from math import sqrt
import math
# Creating the architecture of the Neural Network

class Network (nn.Module):
    
    def __init__(self,inputsize,nbactions):
        super(Network, self).__init__()
        self.inputsize=inputsize
        self.nbactions=nbactions
        self.fc1=nn.Linear(inputsize,64)
        
        n = self.fc1.in_features
        y = 1.0/np.sqrt(n)
       # self.fc1.weight.data.uniform_(0.0, 1.0)
        nn.init.uniform_(self.fc1.weight, -1/sqrt(5), 1/sqrt(5))
        
        self.fc2=nn.Linear(64,64)
        n = self.fc2.in_features
        y = 1.0/np.sqrt(n)
       # self.fc1.weight.data.uniform_(0.0, 1.0)
        nn.init.uniform_(self.fc2.weight, -1/sqrt(5), 1/sqrt(5))
        
        self.fc3=nn.Linear(64,64)
        n = self.fc3.in_features
        y = 1.0/np.sqrt(n)
        #self.fc2.weight.data.uniform_(0.0, 1.0)
        nn.init.uniform_(self.fc3.weight, -1/sqrt(5), 1/sqrt(5))
        
        self.fc4=nn.Linear(64,nbactions)
        n = self.fc4.in_features
        y = 1.0/np.sqrt(n)
        #self.fc2.weight.data.uniform_(0.0, 1.0)
        nn.init.uniform_(self.fc4.weight, -1/sqrt(5), 1/sqrt(5))
        
    
    def forward(self,state):
        #m = nn.LeakyReLU(0.001)
        #m=nn.Sigmoid()
        #x=m(self.fc1(state))
      #  x1=m(self.fc2(x))
        #x2=m(self.fc3(x1))
        #q_values=m(self.fc4(x))
        
    #    print(q_values.data.shape)
        x=F.relu(self.fc1(state))
        x1=F.relu(self.fc2(x))
        #x2=F.relu(self.fc3(x1))
        #print(x2)
        q_values = self.fc4(x1)
        
        #print(self.fc4.weight)
        return q_values
    
class ReplayMemory(object):
    
    def __init__(self,capacity):
        self.capacity=capacity
        self.memory=[]
        
    def Push(self,state):

        self.memory.append(state)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        samples =list(zip(*random.sample(self.memory,batch_size)))

        return map(lambda x: torch.cat(x,0),samples)
    
class Dqn():
    
    def __init__(self, inputsize, nbaction, gamma,LR,batchsize):
        self.LR=LR
        self.batchsize=batchsize
        self.nbaction=nbaction
        self.gamma=gamma
        self.rewardwindow=[]
        self.avgloss=[]
        self.model=Network(inputsize,nbaction)
        self.memory=ReplayMemory((100000)) #10000
        self.optimizer=optim.RMSprop(self.model.parameters(), lr=self.LR)
        self.laststate=torch.Tensor(inputsize).unsqueeze(0)
        self.lastaction=0
        self.lastreward=0
        self.tdloss=0
        self.beta=0.1
        self.EPS_START = 1.0
        self.EPS_END = 0.02
        self.EPS_DECAY = 2000
        self.steps_done=0
        
    def anaelbeta(self):
        # if(self.beta<9):
        #     self.beta=self.beta+0.2
        #     print("ANELSED")
        self.steps_done += 1
        
    def fullanal(self):
        ##NO MORE RANDOM ACTIONS
        self.steps_done=100000
        
    def selectaction(self, state):
        
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        
        if sample > eps_threshold: ##TAKE THOUGHTFUL ACTION
     
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action=self.model(state)
               # print(action)
                return action.max(1)[1].view(1,1)
        else:  ##RANDOM ACTION
            #print("RANDOM")
            return torch.tensor([[random.randrange(self.nbaction)]], dtype=torch.long)
    
    
        # with torch.no_grad():
        #     QVAL=self.model(state)
            
            
        #     QVAL = QVAL - torch.max(QVAL)
            
        #     p_a_s = torch.exp(torch.mul(self.beta,QVAL))/torch.sum(torch.exp(torch.mul(self.beta,QVAL)))

        #     action_key=p_a_s.data.multinomial(num_samples=1,replacement=True)
            
    
        #     return(action_key)
                    
            # QVAL=self.model(state)
            # prob=F.softmax(QVAL*1,dim=1) #TEMP   TODO check array dimensions
            # #print(self.fc4.weight)
            # action=prob.multinomial(num_samples=1)
            # return action.data[0,0]
        
    def learn(self, batchstate,batchnextstate,batchreward,batchaction,targetmodel):
        
   
        outputs=self.model(batchstate).gather(1,batchaction.unsqueeze(1)).squeeze(1)
        
       # print(outputs)
        nextoutputs=targetmodel.model(batchnextstate).max(1)[0].detach()
       # print(outputs)
       # print(nextoutputs.shape)
        target=self.gamma*nextoutputs+batchreward
        
        tdloss=F.smooth_l1_loss(outputs,target)
        
        if(tdloss.data>20):
             print("LOSS TOO HIGH: "+str(tdloss.data))
            # print(outputs.shape)
            # print(target.shape)
        self.optimizer.zero_grad()

        tdloss.backward(retain_graph=True)
  
        for param in self.model.parameters():
            
            if (param.grad!=None):
                param.grad.data.clamp_(-1, 1)
         

         
        self.optimizer.step()
        
    def update(self, reward, newsignal,targetmodel):
        newstate=torch.Tensor(newsignal).float().unsqueeze(0)
        self.memory.Push((self.laststate,newstate,torch.LongTensor([int(self.lastaction)]),torch.Tensor([self.lastreward])))
        
        
        if len(self.memory.memory)>self.batchsize:
            
            action=self.selectaction(newstate)
            batchstate, batchnextstate,batchaction,batchreward=self.memory.sample(self.batchsize)
            self.learn(batchstate, batchnextstate,batchreward,batchaction,targetmodel)
        else:
            action=torch.tensor([[random.randrange(self.nbaction)]], dtype=torch.long)
        self.lastaction=action
        self.laststate=newstate
        self.lastreward=reward
        self.rewardwindow.append(reward)
        if len(self.rewardwindow)>1500:
            del self.rewardwindow[0]
        return action.numpy()[0][0]
    
    def score(self):
        return sum(self.rewardwindow)/(len(self.rewardwindow)+1)
    def getsdictionary(self):
        return (self.model.state_dict())
    def setsdictionary(self,param):
        self.model.load_state_dict(param)
    
    def seteval(self):
        self.model.eval()
    
    def save(self,name,data):
        torch.save({'statedict':self.model.state_dict(),
                    'optimizer':self.optimizer.state_dict(),
                    },str(name)+'.pth')
        a=np.array(data)
        b = a.tolist() # nested lists with same data, indices
        file_path = str(name)+str(".json") ## your path variable
        json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
                
    def load(self,name):
        succ=0
        if os.path.isfile(str(name)+'.pth'):
            print("loading file")
            checkpoint = torch.load(str(name)+'.pth')
            self.model.load_state_dict(checkpoint['statedict'])
            print(self.optimizer)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            obj_text = codecs.open(str(name)+str(".json"), 'r', encoding='utf-8').read()
            b_new = json.loads(obj_text)
            a_new = np.array(b_new)
            print("done")
            succ=1

        else:
            print("no load ):")
        return succ,a_new
        
            
# with torch.no_grad():
#             cstate=self.model(state)
#             print(cstate)
#             cstate=sum(cstate[0],0)
#             prob=F.softmax(cstate, dim=-1)
#             print("IMMMMM")
#             print(prob)
#             action=prob.multinomial(num_samples=1)
#             print(action)
#             return action.data[0]