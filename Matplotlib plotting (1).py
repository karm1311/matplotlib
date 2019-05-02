
# coding: utf-8

# # Installation

# In[ ]:


#conda install matplotlib
#pip install matplotlib


# # Importing matplotlib

# In[56]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#basic code:
from matplotlib import pyplot as plt

#Plotting to our canvas
plt.plot([1,2,3],[4,5,1])

#Showing what we plotted
plt.show()


# In[57]:


from matplotlib import pyplot as plt
 
x = [5,2,7]
y = [2,16,4]
plt.plot(x,y)
plt.title('Info')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.show()


# In[62]:


from matplotlib import pyplot as plt
from matplotlib import style
 
style.use('ggplot')
x = [5,8,10]
y = [12,16,6]
x2 = [6,9,11]
y2 = [6,15,7]
plt.plot(x,y,'g',label='line one', linewidth=9)
plt.plot(x2,y2,'r',label='line two',linewidth=9)
plt.title('Chart Information')
plt.ylabel('Based on Y axis')
plt.xlabel('Based on X axis')
plt.legend()
plt.grid(True,color='k')
plt.show()


# In[64]:


from matplotlib import pyplot as plt
 
plt.bar([0.25,1.25,2.25,3.25,4.25],[50,40,70,80,20],
label="Maruti",width=.5)
plt.bar([.75,1.75,2.75,3.75,4.75],[80,20,20,50,60],
label="Jaguar", color='g',width=.5)
plt.legend()
plt.xlabel('Days')
plt.ylabel('Distance (kms)')
plt.title('Car Information')
plt.show()


# In[65]:


import matplotlib.pyplot as plt
population_age = [22,55,62,45,21,22,34,42,42,4,2,102,95,85,55,110,120,70,65,55,111,115,80,75,65,54,44,43,42,48]
bins = [0,10,20,30,40,50,60,70,80,90,100]
plt.hist(population_age, bins, histtype='bar', rwidth=0.9)
plt.xlabel('Age Groups')
plt.ylabel('Total Number of people')
plt.title('Histogram Diagram')
plt.show()


# In[68]:


import matplotlib.pyplot as plt
x = [1,1.7,2,3.5,3,3.5,3.8]
y = [7.9,8,8.4,9,9.3,10,10.2]
 
x1=[8,8.2,9,9.7,10,10.3,11]
y1=[3,3.2,4.7,4,4.5,5,5.7]
 
plt.scatter(x,y, label='high income low saving',color='r')
plt.scatter(x1,y1,label='low income high savings',color='g')
plt.xlabel('saving*100')
plt.ylabel('income*1000')
plt.title('Scatter Plot Example')
plt.legend()
plt.show()


# In[47]:


import matplotlib.pyplot as plt
days = [1,2,3,4,5]
sleeping =[7,8,6,11,7]
eating = [2,3,4,3,2]
working =[7,8,7,2,2]
playing = [8,5,7,8,13]
  
plt.plot([],[],color='m', label='Sleeping', linewidth=5)
plt.plot([],[],color='c', label='Eating', linewidth=5)
plt.plot([],[],color='r', label='Working', linewidth=5)
plt.plot([],[],color='k', label='Playing', linewidth=5)
  
plt.stackplot(days, sleeping,eating,working,playing, colors=['m','c','r','k'])
  
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Example for Stack Plot')
plt.legend()
plt.show()


# In[50]:


import matplotlib.pyplot as plt
 
days = [1,2,3,4,5]
 
sleeping =[7,8,6,11,7]
eating = [2,3,4,3,2]
working =[7,8,7,2,2]
playing = [8,5,7,8,13]
slices = [7,2,2,13]
activities = ['sleeping','eating','working','playing']
colrs = ['c','m','r','b']
 
plt.pie(slices,
  labels=activities,
  colors=colrs,
  startangle=90,
  shadow= True,
  explode=(0,0.1,0,0),
  autopct='%1.1f%%')
 
plt.title('Example for Pie Plot')
plt.show()


# In[53]:


import numpy as np
import matplotlib.pyplot as plt
 
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2))
plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2))
plt.show()


# In[55]:


import pandas as pd
import matplotlib.transforms as mtransforms

url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS'
vix = pd.read_csv(url, index_col=0, parse_dates=True, na_values='.',
                  infer_datetime_format=True,
                  squeeze=True).dropna()
ma = vix.rolling('90d').mean()
state = pd.cut(ma, bins=[-np.inf, 14, 18, 24, np.inf],
               labels=range(4))

cmap = plt.get_cmap('RdYlGn_r')
ma.plot(color='black', linewidth=1.5, marker='', figsize=(8, 4),
        label='VIX 90d MA')
ax = plt.gca()  # Get the current Axes that ma.plot() references
ax.set_xlabel('')
ax.set_ylabel('90d moving average: CBOE VIX')
ax.set_title('Volatility Regime State')
ax.grid(False)
ax.legend(loc='upper center')
ax.set_xlim(xmin=ma.index[0], xmax=ma.index[-1])

trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
for i, color in enumerate(cmap([0.2, 0.4, 0.6, 0.8])):
    ax.fill_between(ma.index, 0, 1, where=state==i,
                    facecolor=color, transform=trans)
ax.axhline(vix.mean(), linestyle='dashed', color='xkcd:dark grey',
           alpha=0.6, label='Full-period mean', marker='')


# # Basic Plotting, Plotting terminology

# In[3]:


x = np.linspace(0,10,100)
y = np.sin(x)
z = np.cos(x)


# In[4]:


x


# In[16]:


from matplotlib import pyplot as plt
 
x = [5,2,7]
y = [2,16,4]
plt.plot(x,y)
plt.title('Info')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.show()


# In[15]:


from matplotlib import pyplot as plt
 
x = [5,2,7]
y = [2,16,4]
plt.plot(x,y)
plt.title('Info')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.show()


# In[9]:


# Let’s illustrate with a stacked area graph of three time series:
rng = np.arange(50)
rnd = np.random.randint(0, 10, size=(3, rng.size))
yrs = 1950 + rng

fig, ax = plt.subplots(figsize=(5, 3))
ax.stackplot(yrs, rng + rnd, labels=['Eastasia', 'Eurasia', 'Oceania'])
ax.set_title('Combined debt growth over time')
ax.legend(loc='upper left')
ax.set_ylabel('Total debt')
ax.set_xlim(xmin=yrs[0], xmax=yrs[-1])
fig.tight_layout()


# In[10]:


x = np.random.randint(low=1, high=11, size=50)
y = x + np.random.randint(1, 5, size=x.size)
data = np.column_stack((x, y))

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                               figsize=(8, 4))

ax1.scatter(x=x, y=y, marker='o', c='r', edgecolor='b')
ax1.set_title('Scatter: $x$ versus $y$')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')

ax2.hist(data, bins=np.arange(data.min(), data.max()),
         label=('x', 'y'))
ax2.legend(loc=(0.65, 0.8))
ax2.set_title('Frequencies of $x$ and $y$')
ax2.yaxis.tick_right()

