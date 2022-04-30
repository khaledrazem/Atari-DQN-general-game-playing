                                        # -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 02:00:08 2021

@author: khale
"""

import gym
import cv2
import numpy as np 
from matplotlib import pyplot as plt


from ai import Dqn
from segmenttest import segmentframe
from segmenttest import getclassarray
from segmenttest import setclassarray

#HYPER-PARAMETERS

#ALE

frameskips=3  ###3
EP=9000
EvalEP=1000
movements=10000
targetupdate=1000

save=False
load=True

visual=True

games=['Seaquest-v4','SpaceInvaders-v4','Breakout-v4','BeamRider-v4','Pong-v4']
currentgame=0

#SEGMENTATION

debug=False
saveimgs=False
objctsize=32
gridsize=14

#classification

classthresh=-0.25

#DQN
LR=0.00025 #0.00025
batchsize=32 #32


env = gym.make(games[currentgame])
actions = env.action_space.n

brain=Dqn(18,actions,0.9,LR,batchsize) #3 input, (reward , images, coordinates)
target_brain=Dqn(18,actions,0.9,LR,batchsize)
target_brain.setsdictionary(brain.getsdictionary())
target_brain.seteval()

if load:

    succ,classlist=brain.load(games[currentgame])
    target_brain.setsdictionary(brain.getsdictionary())
    if succ:
        EP=0
        setclassarray(classlist)
        print("GOING FULL POWER")
        brain.fullanal()

scores=np.zeros(EP+EvalEP)
last_reward=0
rand=0
lives=0
change=0

for i in range(EP+EvalEP):
    print("episode: "+str(i))

    frame=env.reset()
    rewards=0
    action=0
    
    for j in range(movements):
        
        if visual:
            env.render()
        
        if j%frameskips==0: #and rand>0:
            
            
         
            frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
            objects=segmentframe(np.array(frame),debug,saveimgs,objctsize,gridsize,classthresh)
            objects=np.array(list(objects.flatten()))
         
            action = brain.update(last_reward, objects,target_brain)
            last_reward=0.0
            
        # elif rand<=0:
        #     print("RANDOgggM"+str(rand))
        #     action=env.action_space.sample()
        #     rand=rand+1
            
   
        if j % targetupdate == 0: 
            target_brain.setsdictionary(brain.getsdictionary())
        
        frame,s,done,c=env.step(action) # take a thoughtful action
        
        
        


        if s>0:
            rewards=rewards+s
            last_reward=1

        if currentgame!=4:
            change=c["ale.lives"]-lives
            lives=c["ale.lives"]
            
        if(change!=0):
      
            rand=0
            last_reward=-1.0
        
        if done:
            last_reward=-1.0
            rand=0
            break
        
    scores[i]=rewards
    print("score: "+str(rewards))
    brain.anaelbeta()
    
    if EP==i:
        print("GOING FULL POWER")
        brain.fullanal()
        
        
    if i%100==0:
        print("tenavg"+str(sum(scores[i-100:i])/100))
        avg=(sum(scores)/(i+1))
        print("avg"+str(avg))
        plt.plot((scores))
        plt.ylabel('Score')
        plt.xlabel('Episode')
        plt.show()
        

    env.close()
    
print("evaluation average"+str(sum(scores[-EvalEP:])/EvalEP))
print(scores)
a, b = np.polyfit(range(EP+EvalEP),scores, 1)
plt.plot(scores)
plt.plot(a*range(EP+EvalEP)+b)   
plt.ylabel('Score')
plt.xlabel('Episode')
plt.show()
print("Gradient: "+str(a))

if save and not load:
    brain.save(games[currentgame],getclassarray())