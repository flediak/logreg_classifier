#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Logistic regression for Goofies


# # Todo
# - define classes
# - get training data
# - make convergence tests

# # references:
# - https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148
# - https://towardsdatascience.com/logistic-regression-with-python-using-optimization-function-91bd2aee79b

# In[358]:


import numpy as np
from matplotlib import pyplot as plt


# # read data

# ### clean

# # plot data

# # feature scaling

# # define samples (train, test, validation)

# # define model

# In[359]:


def sigmoid(X,theta):
    z = np.matmul(X,theta.T)
    return (1+np.exp(-z))**-1


# In[360]:


def hypothesis(X,theta):
    return sigmoid(X, theta)


# # initialize data matrix X

# ### make random input data

# In[361]:


Ndat = 1000
Nfeature_in = 2


# In[362]:


#random feature vectors for Ndat data points
f1 = -1+np.random.rand(Ndat)*2
f2 = -1+np.random.rand(Ndat)*2

# concat input vectors to data matrix
X = np.concatenate((f1,f2)).reshape(Nfeature_in,Ndat).T


# ### add higher order terms to data matrix

# In[363]:


def add_higher_orders(X_in,max_order):
    
    X_out = X_in.copy()
    
    Nfeature_in = X_out.shape[1]

    for n in range(1,max_order):

        exponent = n+1

        for i in range(Nfeature_in):
            Nfeature_tmp = X_out.shape[1]
            Xn = X_out[:,i]**exponent
            X_out = np.insert(X_out,Nfeature_tmp,Xn, axis=1)
            
    return X_out


# In[364]:


max_order = 3


# In[365]:


X = add_higher_orders(X,max_order)


# In[366]:


print('Nfeature =', X.shape[1])


# ### add 1 as first column bias terms

# In[367]:


X = np.insert(X,0,1, axis=1)


# ### examples of descision bounderies for random parameters

# In[368]:


Nx = 5
Ny = 5

patch_size=1

fig,ax = plt.subplots(Ny,Nx, figsize=(patch_size*Nx,patch_size*Ny), sharex=True, sharey=True)


np.random.seed(1234321)
#np.random.seed(103)

for ix in range(Nx):
    for iy in range(Ny):
        
        #mnake random parameter vector in intervall [-1,1]
        Nfeature = X.shape[1]
        
        theta = -1 + np.random.rand(Nfeature)*2
        
        sig = sigmoid(X,theta)
        gal_class = np.zeros(len(sig))
        gal_class[sig>0.5]=1
        
        x = X[:,1]
        y = X[:,2]

        select1 = gal_class==0
        select2 = gal_class==1
        
        #ax[iy,ix].scatter(X[:,1],X[:,2],c=sig, cmap='gnuplot', s=20)
        ax[iy,ix].scatter(x[select1],y[select1],c='navy', s=10)
        ax[iy,ix].scatter(x[select2],y[select2],c='orangered', s=10)
                
        ax[iy,ix].axis('off')

plt.tight_layout()


# In[385]:


#example

theta = -1 + np.random.rand(Nfeature)*2
        
sig = sigmoid(X,theta)
gal_class = np.zeros(len(sig))
gal_class[sig>0.5]=1

Y = gal_class

plt.scatter(X[:,1],X[:,2],c=Y, cmap='bwr', s=20)
plt.show()


# In[386]:


theta_in = theta
print(theta)


# In[387]:


X.shape


# In[388]:


Y.shape


# # define cost function
# 
# ### $J = -\frac{1}{m}\sum \left[ y^{i} log(h_\theta(x^i)) + (1-y^i)log(1-h_\theta(x^i))  \right]$

# In[389]:


def cost_function(theta, X, Y):
    m = X.shape[0]
    h = hypothesis(X,theta)
    return -(1/m)*np.sum(Y*np.log(h) + (1-Y)*np.log(1-h))


# In[390]:


h = hypothesis(X,theta)


# In[391]:


cost_function(theta,X,Y)


# # define gradient

# In[392]:


def gradient(theta, X, Y):
    m = X.shape[0]
    h = hypothesis(X,theta)
    return (1/m) * np.dot(X.T, (h-Y))


# # fit model to training data

# In[393]:


theta = np.ones((X.shape[1], 1))


# In[394]:


from scipy.optimize import minimize,fmin_tnc


# In[395]:


def fit(x, y, theta):
    opt_weights = fmin_tnc(func=cost_function, x0=theta, fprime=gradient, args=(x, y.flatten()))
    return opt_weights[0]


# In[396]:


parameters = fit(X, Y, theta)


# In[397]:


print(parameters)
print(theta_in)


# In[398]:


Nx, Ny = 2,1
patch_size = 4

fig,ax = plt.subplots(Ny,Nx, figsize=(patch_size*Nx,patch_size*Ny), sharex=True, sharey=True)

x = X[:,1]
y = X[:,2]

sig1 = sigmoid(X,theta_in)
sig2 = sigmoid(X,parameters)

ax[0].set_title('truth')
ax[0].scatter(X[:,1],X[:,2],c=Y, cmap='bwr', s=20)

ax[1].set_title('fit')
ax[1].scatter(X[:,1],X[:,2],c=sig2, cmap='bwr', s=20)

#ax[iy,ix].scatter(x[select1],y[select1],c='navy', s=10)
#ax[iy,ix].scatter(x[select2],y[select2],c='orangered', s=10)


# # define class
# - initialize
# - read training data
# - fit model
# - compute cost
# - make prediction
# 
# import logistic_regression as logreg
# 
# model = logreg(order: 3, classifyer: binary)
# model.fit(data: X_train, optimization = 'something')
# mode.parameters()
# model.predict()

# In[ ]:





# In[350]:


class logistic_regression:
    
    def  __init__(self, **kwargs):

        #set default values
        self.Nfeature_in = 0
        self.Nfeature_model = 0
        
        self.poly_order = 1
        self.feature_names = []
    
        self.optimizer = 'scipy_optimize'
    
        for key, value in kwargs.items():
            if key == 'poly_order': self.poly_order = value
            
            
    def add_higher_orders(self,X_in,max_order):

        X_out = X_in.copy()

        Nfeature_in = X_out.shape[1]

        for n in range(1,max_order):

            exponent = n+1

            for i in range(Nfeature_in):
                Nfeature_tmp = X_out.shape[1]
                Xn = X_out[:,i]**exponent
                X_out = np.insert(X_out,Nfeature_tmp,Xn, axis=1)

        return X_out
            
    
    def training_data(self,**kwargs):
        
        for key, value in kwargs.items():
            if key == 'feature_names': self.feature_names = value

        #todo:
        #- read pandas dataframe
        #- make trueth vector
        #- make data matrix
        #- add higher orders
        #- add bias term
        #- adjust Nfeature_in, Nfeature_model
            
            
    def model_summary(self):        
        print('polynomial order: ', self.poly_order)
        print('number of input features: ', self.Nfeature_in)
        print('number of model parameter: ', self.Nfeature_model)
        
        
    def sigmoid(self,X,theta):
        z = np.matmul(X,theta.T)
        return (1+np.exp(-z))**-1

    
    def hypothesis(self,X,theta):
        return sigmoid(X, theta) 
        
        
    def cost_function(theta, X, Y):
        m = X.shape[0]
        h = hypothesis(X,theta)
        return -(1/m)*np.sum(Y*np.log(h) + (1-Y)*np.log(1-h))

    
    def fit_model(self,**kwargs):
        for key, value in kwargs.items():
            if key == 'optimizer': self.optimizer = value
        print(self.optimizer)


    def validate_model(self):
        print('cost_train:')
        print('cost_test:')
    


# In[351]:


logreg = logistic_regression(poly_order=4)
#logreg = logistic_regression()


# In[352]:


names = ['wer', 'wie', 'was', 'wann', 'warum']
logreg.training_data(feature_names = names)


# In[353]:


logreg.model_summary()


# In[354]:


logreg.fit_model(optimizer='something new')


# In[ ]:





# # convergence tests

# # convert notebook to python script and remove this command from script

# In[355]:


