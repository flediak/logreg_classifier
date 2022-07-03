#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Logistic regression for Goofies


# # Todo
# - get training data
# - make convergence tests
# - add mulsti calss classification

# # references:
# - https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148
# - https://towardsdatascience.com/logistic-regression-with-python-using-optimization-function-91bd2aee79b

# In[18]:


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


# # read data

# In[52]:


fname_cosmos = '~/Documents/IA/data/COSMOS/matched_ZEST_L15_ACS-GC_v1.cvs.bz2'

cat = pd.read_csv(fname_cosmos)

# rename columns
cat.rename(columns={"mass_best": "lgM", "photoz": "z"}, inplace=True)

# define colors
cat['ur'] = cat.m_u - cat.m_r
cat['rk'] = cat.m_r - cat.m_k
cat['rj'] = cat.m_r - cat.m_j
cat['nuvr'] = cat.m_nuv - cat.m_r

# apparent axis ratio
cat['q_app'] = cat.b_a_G1
cat['rcut_pix'] = cat.Re_G1

#clean
cat = cat[cat.ip_mag_iso<24]
cat = cat[cat.b_a_G1<=1]
cat = cat[cat.rcut_pix<750]


# # volume limited samples

# In[53]:


z_min, z_max = 0.2, 2.0


# ### intrinsic size cut

# In[54]:


ang_psf = 0.085 # psf angle in arcsec
ang_pix = 0.03 # pixel angle in arcsec

ang_min = ang_psf*1.0 # min angle in arcsec
app_mag_max=24


# In[55]:


from colossus.cosmology import cosmology


# In[57]:


z_rcut = z_max

cosmo = cosmology.setCosmology('planck18')
funit = np.pi/(180*60**2)*(cosmo.H0/100.)*1000.
r_cut_kpc_min = cosmo.angularDiameterDistance(z_rcut)*ang_min*funit
#print(cosmo.H0, cosmo.Om0, cosmo.Ob0, cosmo.sigma8, cosmo.ns)
#print(r_cut_kpc)

#r_cut_kpc = 0.0
cat['rcut_arcsec'] = cat.rcut_pix*ang_pix
cat = cat[cat.rcut_arcsec>0]
cat['rcut_kpc'] = cosmo.angularDiameterDistance(cat.z.values)*cat.rcut_arcsec*funit


# In[58]:


z_min, z_max = 0.2,1.5
lgM_min, lgM_max = 5.0,15.0
mag_min, mag_max = -25,24

print(len(cat))

# intrinsic size
cat = cat[cat.rcut_kpc>r_cut_kpc_min]

# magnitude
cat = cat.loc[cat.m_i > mag_min]
cat = cat.loc[cat.m_i < mag_max]

# redshift
cat = cat.loc[cat.z > z_min]
cat = cat.loc[cat.z < z_max]

print(len(cat))


# # set galaxy types

# ### disks (disk dominated)

# In[71]:


type_ZEST = 2 # late type
bulg1, bulg2 = 2,3 # disk dominated
irre1, irre2 = 9,9 # no values

select_disks_dd = (cat.type_ZEST==type_ZEST )                    & ((cat.bulg == bulg1) | (cat.bulg == bulg2) )                    & ((cat.irre == irre1) | (cat.irre == irre2) )


# ### disks (bulge dominated)

# In[72]:


type_ZEST = 2 # late type
bulg1, bulg2 = 1,2 # bulge dominated
irre1, irre2 = 9,9 # no values
rcut_min_kpc = r_cut_kpc

select_disks_bd = (cat.type_ZEST==type_ZEST )                    & ((cat.bulg == bulg1) | (cat.bulg == bulg2) )                    & ((cat.irre == irre1) | (cat.irre == irre2) )


# ### ellipticals

# In[73]:


type_ZEST = 1 # early type
bulg1, bulg2 = 9,9 # no values
irre1, irre2 = 0,1 # regular
rcut_min_kpc = r_cut_kpc

select_ellis = (cat.type_ZEST==type_ZEST )                    & ((cat.bulg == bulg1) | (cat.bulg == bulg2) )                    & ((cat.irre == irre1) | (cat.irre == irre2) )


# ### iregular

# In[74]:


type_ZEST = 3 # late type
bulg1, bulg2 = 9,9 # no values
irre1, irre2 = 9,9 # no values
rcut_min_kpc = r_cut_kpc

select_irre = (cat.type_ZEST==type_ZEST )                    & ((cat.bulg == bulg1) | (cat.bulg == bulg2) )                    & ((cat.irre == irre1) | (cat.irre == irre2) )


# In[90]:


cat['gal_class'] = 0
cat.loc[select_disks_dd, 'gal_class']=1


# # plot data

# In[96]:


plt.scatter(cat.rj, cat.nuvr, s=0.1, c=cat.gal_class)


# # select random subsets for training, validation and testing

# In[277]:


frac_train, frac_val, frac_test = 0.4,0.4,0.2


# In[278]:


rnd = np.random.random(len(cat))


# In[279]:


cat_train = cat.loc[rnd <=frac_train]
cat_val = cat.loc[((frac_train < rnd) & (rnd<frac_train+frac_val))]
cat_test = cat.loc[frac_train+frac_val <= rnd]


# In[280]:


len(cat_train)/len(cat), len(cat_val)/len(cat), len(cat_test)/len(cat)


# # initialize data matrix X

# In[281]:


features = ['rj', 'nuvr']

X_train = cat[features].values
Y_train = cat['gal_class']

nonan = (np.isnan(X_train[:,0])==False) & (np.isnan(X_train[:,1])==False)

X_train = X_train[nonan]
Y_train = Y_train[nonan]


# # fit model to training data

# # define class

# In[302]:


from scipy.optimize import minimize,fmin_tnc

class logistic_regression:
    
    def  __init__(self, X_train,Y_train ,**kwargs):

        self.X_train = X_train
        self.Y_train = Y_train
        
        #set default values
        self.Nfeature_in = self.X_train.shape[1]
        
        self.poly_order = 1
    
        self.optimizer = 'scipy_optimize'
    
        for key, value in kwargs.items():
            if key == 'poly_order': self.poly_order = value
            
        #re-scale input features
        X_train = X_train - X_train.T.mean(1)
        X_train = X_train / X_train.T.var(1)
        
        ### add higher orders
        self.X_train = self._add_higher_orders(self.X_train,self.poly_order)
        
        ### add 1 as first column bias terms
        self.X_train = np.insert(self.X_train,0,1, axis=1)
        
        self.Nfeature_model = self.X_train.shape[1]
            
        #model parameters
        self.theta_ini = np.full(self.Nfeature_model, 1)
        self.theta_fit = np.full(self.Nfeature_model, np.nan)
        
        #costs
        self.cost_train = np.nan
        self.cost_val = np.nan
        

    def _add_higher_orders(self,X_in,max_order):

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
            
            
    def summary(self):        
        print('polynomial order: ', self.poly_order)
        print('number of input features: ', self.Nfeature_in)
        print('number of model parameter: ', self.Nfeature_model)
        
        
    def sigmoid(self,X,theta):
        z = np.matmul(X,theta.T)
        return (1+np.exp(-z))**-1

    
    def _hypothesis(self,X,theta):
        return sigmoid(X, theta) 
        
        
    def _cost_function(self,theta, X, Y):
        m = X.shape[0]
        h = hypothesis(X,theta)
        return -(1/m)*np.sum(Y*np.log(h) + (1-Y)*np.log(1-h))

    
    def _gradient(self,theta, X, Y):
        m = X.shape[0]
        h = hypothesis(X,theta)
        return (1/m) * np.dot(X.T, (h-Y))
    
    
    def fit(self,**kwargs):
        
        for key, value in kwargs.items():
            if key == 'optimizer': self.optimizer = value
            if key == 'theta_ini': self.theta_ini = value
        #print(self.optimizer)
        #print(self.theta_ini)
        
        
        opt_weights = fmin_tnc(
            func=self._cost_function, x0=self.theta_ini, fprime=self._gradient,
            args=(self.X_train, self.Y_train)
        )
        
        self.theta_fit = opt_weights[0]
        
        
        self.cost_train = self._cost_function(self.theta_fit, self.X_train, self.Y_train)


    def validate_model(self):
        print('cost_train:')
        print('cost_test:')
    


# In[303]:


model = logistic_regression(X_train, Y_train, poly_order=2)
model.summary()


# In[304]:


model.fit(theta_ini=np.array([2,1,2,2,2]))


# In[177]:


model.theta_fit


# In[172]:


model.cost_train


# In[173]:


names = ['wer', 'wie', 'was', 'wann', 'warum']
model.training_data(feature_names = names)


# # convergence tests

# # convert notebook to python script and remove this command from script

# In[64]:


