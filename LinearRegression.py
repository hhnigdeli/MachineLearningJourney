
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


data = pd.read_csv("data.csv")


# In[ ]:


data = data[data["Gender"]=="Male"]


# In[ ]:


data = data[["Height","Weight"]]


# In[ ]:


data.columns = ["Weight","Height"]


# In[ ]:


data.head(5)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.scatter(data.Height,data.Weight)


# In[ ]:


import numpy as np


# In[ ]:


import random


# Sampling 

# In[ ]:


newIndex = random.sample(list(data.index), 25)


# In[ ]:


dataSampled = data.reindex(newIndex)


# In[ ]:


plt.scatter(dataSampled.Height,dataSampled.Weight)


# <li>y = weight 
# <li>x = height 
# 
# 
# 
# <li>y => dependent variable 
# <li>x => independent variable 

# In[ ]:


x = list(dataSampled.Height)
y = list(dataSampled.Weight)
N = len(dataSampled)


# Normalization according to standard score

# In[ ]:


x = [(i-np.mean(x))/np.std(x)  for i in x]
y = [(i-np.mean(y))/np.std(y)  for i in y]


# In[ ]:


plt.scatter(x,y)


# <li>intercept            => b 
# <li>coefficent                => m
# 

# y_ = b + m*x

# lossFunction = (y - (b + m*x))**2

# <li>d lossFunction / d b = -2*(y-(m*x+b))
# <li>d lossFunction / d m = -2*x*(y-(m*x+b))

# In[ ]:


c=0
LearningRate = 0.008
b = 4
m = 3
iter = 175
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



# In[ ]:


plt.plot(mselist,color="r")
plt.scatter(range(iter),mselist)
plt.xlabel("Iteration")
plt.ylabel("Mean Suqare Error")
plt.title("Gradient Descent")


# In[ ]:


regressionLine = [ (b + m*i)    for i in range(-3,5)]


plt.scatter(x,y, color="k")
plt.plot(range(-3,5),regressionLine, color="r")
plt.title("Fited Regression Line ")
plt.xlabel("X")
plt.ylabel("Y")


# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = 15, 10 
rcParams['axes.titlesize'] =18


# Animation for learning Curve

# In[ ]:


for t in range(len(liste)):
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    
    ax1.annotate('f(x) = {}x  + {}'.format(np.round(liste[t][0],4),np.round(liste[t][1],4)), xy=(100, 15),size=15)
    
    plt.plot(mselist,color="r")
    plt.scatter(range(len(mselist[:t])),mselist[:t])
    plt.xlabel("Iteration")
    plt.ylabel("Mean Suqare Error")
    plt.title("Gradient Descent")
    
    
    ax2 = plt.subplot2grid((3, 1), (1, 0),rowspan=2)
    
    regressionLine = [ (liste[t][0] + liste[t][1]*i)    for i in range(-3,5)]
    plt.scatter(x,y, color="k")
    plt.plot(range(-3,5),regressionLine, color="r")
    plt.title("Fited Regression Line ")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    
    plt.tight_layout()
    plt.savefig("foto{}.png".format(t)) #save every scene on local directory as .png
    plt.close()
    
    


# creating video with ffmpeg the commands can run on Commond prompt and also in notebook cell with ! notation

# In[ ]:


#! ffmpeg -r 10 -f image2 -s 1280x1024 -i foto%01d.png -vcodec libx264 -crf 40 -pix_fmt yuv420p regression.mp4 


# In[ ]:




