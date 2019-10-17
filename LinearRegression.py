
# coding: utf-8

# In[7]:


import pandas as pd


# In[8]:


data = pd.read_csv("data.csv")


# In[9]:


data = data[data["Gender"]=="Male"]


# In[10]:


data = data[["Height","Weight"]]


# In[11]:


data.columns = ["Weight","Height"]


# In[12]:


data.head(5)


# In[13]:


import matplotlib.pyplot as plt


# In[14]:


plt.scatter(data.Height,data.Weight)


# In[15]:


import numpy as np


# In[16]:


import random


# Sampling 

# In[17]:


newIndex = random.sample(list(data.index), 25)


# In[18]:


dataSampled = data.reindex(newIndex)


# In[19]:


plt.scatter(dataSampled.Height,dataSampled.Weight)


# <li>y = weight 
# <li>x = height 
# 
# 
# 
# <li>y => dependent variable 
# <li>x => independent variable 

# In[20]:


x = list(dataSampled.Height)
y = list(dataSampled.Weight)
N = len(dataSampled)


# Normalization according to standard score

# In[21]:


x = [(i-np.mean(x))/np.std(x)  for i in x]
y = [(i-np.mean(y))/np.std(y)  for i in y]


# In[22]:


plt.scatter(x,y)


# <li>intercept            => b 
# <li>coefficent                => m
# 

# y_ = b + m*x

# lossFunction = (y - (b + m*x))**2

# <li>d lossFunction / d b = -2*(y-(m*x+b))
# <li>d lossFunction / d m = -2*x*(y-(m*x+b))

# In[37]:


c=0
LearningRate = 0.008
b = 4
m = 3
iter = 300
liste = []
mselist = [] # mean square errors list for each iteration 
while c <iter:
        slopem = 0
        slopeb = 0
        for i in range(len(dataSampled)):

                slopem  +=  -2*x[i]*(y[i]-(m*x[i]+b)) # sum of lost function derivatives with respect to m
                
                slopeb  +=  -2*(y[i]-(m*x[i]+b))      # sum of lost function derivatives with respect to b
                
    
        b -= ((slopeb/N)  * LearningRate) # new b
        m -= ((slopem/N) * LearningRate)  # new m
        
        mse = 0
        for i in range(N):
            mse += (y[i] - (m*x[i]+b))**2 # mean square error calculation
        mselist.append(mse/N)  
        
        liste.append((b,m))
        
        c+=1
print("intercept = {} and coefficent = {} and MeanSuqareError = {}".format(b,m,mse))



# In[39]:


plt.plot(mselist,color="r")
plt.scatter(range(iter),mselist)
plt.xlabel("Iteration")
plt.ylabel("Mean Suqare Error")
plt.title("Gradient Descent")


# In[40]:


regressionLine = [ (b + m*i)    for i in range(-3,5)]


plt.scatter(x,y, color="k")
plt.plot(range(-3,5),regressionLine, color="r",linewidth=4)
plt.title("Fited Regression Line ")
plt.xlabel("X")
plt.ylabel("Y")


# In[41]:


from pylab import rcParams
rcParams['figure.figsize'] = 15, 10 
rcParams['axes.titlesize'] =18


# Animation for learning Curve

# In[42]:


for t in range(len(liste)):
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    
    ax1.annotate('f(x) = {}x  + {}'.format(np.round(liste[t][0],4),np.round(liste[t][1],4)), xy=(100, 15),size=15)
    
    plt.plot(mselist,color="k")
    plt.scatter(range(len(mselist[:t])),mselist[:t],edgecolors="r")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Suqare Error")
    plt.title("Gradient Descent")
    
    
    ax2 = plt.subplot2grid((3, 1), (1, 0),rowspan=2)
    
    regressionLine = [ (liste[t][0] + liste[t][1]*i)    for i in range(-2,2)]
    plt.scatter(x,y, color="b",edgecolors="k")
    plt.plot(range(-2,2),regressionLine, color="g",linewidth=3)
    plt.title("Fitted Regression Line ")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    
    plt.tight_layout()
    plt.savefig("image/foto{}.png".format(t)) #save every scene on local directory as .png
    plt.close()
    
    


# To make an animation with condensing whole .pngs which generated before.

# In[43]:


import os
import subprocess
os.chdir('/image')
subprocess.call(['ffmpeg', '-r', '10', '-f', 'image2', '-s', '800x600', '-i', 
                 'foto%01d.png', '-vcodec', 'libx264', '-crf', '24', '-pix_fmt', 'yuv420p', 'regression.mp4'])


# In[44]:


subprocess.call(['ffmpeg', '-y', '-i', 'regression.mp4', '-i', 'r.png', '-filter_complex', '[1]lut=a=val*0.3[a];[0][a]overlay=W-w-10:5', 
                 '-c:v', 'libx264', '-an', 'RunDict.mp4'])

