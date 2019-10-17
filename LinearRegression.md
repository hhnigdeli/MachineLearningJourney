

```python
import pandas as pd
```


```python
data = pd.read_csv("data.csv")
```


```python
data = data[data["Gender"]=="Male"]
```


```python
data = data[["Height","Weight"]]
```


```python
data.columns = ["Weight","Height"]
```


```python
data.head(5)
```




<div>


    
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Weight</th>
      <th>Height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>73.847017</td>
      <td>241.893563</td>
    </tr>
    <tr>
      <th>1</th>
      <td>68.781904</td>
      <td>162.310473</td>
    </tr>
    <tr>
      <th>2</th>
      <td>74.110105</td>
      <td>212.740856</td>
    </tr>
    <tr>
      <th>3</th>
      <td>71.730978</td>
      <td>220.042470</td>
    </tr>
    <tr>
      <th>4</th>
      <td>69.881796</td>
      <td>206.349801</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt
```


```python
plt.scatter(data.Height,data.Weight)
```




    <matplotlib.collections.PathCollection at 0x21bdd3eefd0>








```python
import numpy as np
```


```python
import random
```

Sampling 


```python
newIndex = random.sample(list(data.index), 25)
```


```python
dataSampled = data.reindex(newIndex)
```


```python
plt.scatter(dataSampled.Height,dataSampled.Weight)
```




    <matplotlib.collections.PathCollection at 0x21be04ee470>







<li>y = weight 
<li>x = height 



<li>y => dependent variable 
<li>x => independent variable 


```python
x = list(dataSampled.Height)
y = list(dataSampled.Weight)
N = len(dataSampled)
```

Normalization according to standard score


```python
x = [(i-np.mean(x))/np.std(x)  for i in x]
y = [(i-np.mean(y))/np.std(y)  for i in y]
```


```python
plt.scatter(x,y)
```




    <matplotlib.collections.PathCollection at 0x21be030bb38>







<li>intercept            => b 
<li>coefficent                => m


y_ = b + m*x

lossFunction = (y - (b + m*x))**2

<li>d lossFunction / d b = -2*(y-(m*x+b))
<li>d lossFunction / d m = -2*x*(y-(m*x+b))


```python
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



```

    intercept = 0.03166573119029209 and coefficent = 0.9416980515850287 and MeanSuqareError = 3.628530233246005
    


```python
plt.plot(mselist,color="r")
plt.scatter(range(iter),mselist)
plt.xlabel("Iteration")
plt.ylabel("Mean Suqare Error")
plt.title("Gradient Descent")

```




    Text(0.5,1,'Gradient Descent')








```python
regressionLine = [ (b + m*i)    for i in range(-3,5)]


plt.scatter(x,y, color="k")
plt.plot(range(-3,5),regressionLine, color="r",linewidth=4)
plt.title("Fited Regression Line ")
plt.xlabel("X")
plt.ylabel("Y")
```




    Text(0,0.5,'Y')








```python
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10 
rcParams['axes.titlesize'] =18
```

Animation for learning Curve


```python
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
    
    
```

To make an animation with condensing whole .pngs which generated before.


```python
import os
import subprocess
os.chdir('/image')
subprocess.call(['ffmpeg', '-r', '10', '-f', 'image2', '-s', '800x600', '-i', 
                 'foto%01d.png', '-vcodec', 'libx264', '-crf', '24', '-pix_fmt', 'yuv420p', 'regression.mp4'])
```




    0




```python
subprocess.call(['ffmpeg', '-y', '-i', 'regression.mp4', '-i', 'r.png', '-filter_complex', '[1]lut=a=val*0.3[a];[0][a]overlay=W-w-10:5', 
                 '-c:v', 'libx264', '-an', 'RunDict.mp4'])
```




    0


