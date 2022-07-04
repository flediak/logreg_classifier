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
# - https://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/

# In[2]:


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


# # read data

# In[3]:


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

# In[4]:


z_min, z_max = 0.2, 2.0


# ### intrinsic size cut

# In[5]:


ang_psf = 0.085 # psf angle in arcsec
ang_pix = 0.03 # pixel angle in arcsec

ang_min = ang_psf*1.0 # min angle in arcsec
app_mag_max=24


# In[6]:


from colossus.cosmology import cosmology


# In[7]:


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


# In[8]:


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

# In[9]:


type_ZEST = 2 # late type
bulg1, bulg2 = 2,3 # disk dominated
irre1, irre2 = 9,9 # no values

select_disks_dd = (cat.type_ZEST==type_ZEST )                    & ((cat.bulg == bulg1) | (cat.bulg == bulg2) )                    & ((cat.irre == irre1) | (cat.irre == irre2) )


# ### disks (bulge dominated)

# In[10]:


type_ZEST = 2 # late type
bulg1, bulg2 = 1,2 # bulge dominated
irre1, irre2 = 9,9 # no values

select_disks_bd = (cat.type_ZEST==type_ZEST )                    & ((cat.bulg == bulg1) | (cat.bulg == bulg2) )                    & ((cat.irre == irre1) | (cat.irre == irre2) )


# ### ellipticals

# In[11]:


type_ZEST = 1 # early type
bulg1, bulg2 = 9,9 # no values
irre1, irre2 = 0,1 # regular

select_ellis = (cat.type_ZEST==type_ZEST )                    & ((cat.bulg == bulg1) | (cat.bulg == bulg2) )                    & ((cat.irre == irre1) | (cat.irre == irre2) )


# ### iregular

# In[12]:


type_ZEST = 3 # late type
bulg1, bulg2 = 9,9 # no values
irre1, irre2 = 9,9 # no values

select_irre = (cat.type_ZEST==type_ZEST )                    & ((cat.bulg == bulg1) | (cat.bulg == bulg2) )                    & ((cat.irre == irre1) | (cat.irre == irre2) )


# In[13]:


fig,ax = plt.subplots(1,4, figsize=(20,5))

ax[0].scatter(cat[select_disks_dd].rj, cat[select_disks_dd].nuvr, s=0.1, c='k')
ax[1].scatter(cat[select_disks_bd].rj, cat[select_disks_bd].nuvr, s=0.1, c='k')
ax[2].scatter(cat[select_ellis].rj, cat[select_ellis].nuvr, s=0.1, c='k')
ax[3].scatter(cat[select_irre].rj, cat[select_irre].nuvr, s=0.1, c='k')

ax[0].set_title('disks (disk dominated)')
ax[1].set_title('disks (bulge dominated)')
ax[2].set_title('ellipticals)')
ax[3].set_title('iregulars')

ax[0].set_ylabel('$nuv-r$')

for ix in range(4):
    ax[ix].set_xlabel('r-j')

plt.show()


# # fit model to training data

# # define class

# In[49]:


from scipy.optimize import minimize,fmin_tnc, fmin_cg, fmin_bfgs, fmin_l_bfgs_b

class logistic_regression:
    
    def  __init__(self, X_train,Y_train ,**kwargs):

        self.verbose=0
        
        self.X_train = X_train
        self.Y_train = Y_train
        
        #set default values
        self.Nfeature_in = self.X_train.shape[1]
        
        self.poly_order = 1
    
        self.optimizer = 'scipy_optimize'
    
        for key, value in kwargs.items():
            if key == 'poly_order': self.poly_order = value
            if key == 'verbose': self.verbose = value
            
        #re-scale input features
        self.train_mean = self.X_train.T.mean(1)
        self.X_train = self.X_train - self.train_mean
        
        self.train_var = self.X_train.T.var(1)
        self.X_train = self.X_train / self.train_var
        
        ### add higher orders
        self.X_train = self._add_higher_orders(self.X_train,self.poly_order)
        
        ### add 1 as first column bias terms
        self.X_train = np.insert(self.X_train,0,1, axis=1)
        
        self.Nfeature_model = self.X_train.shape[1]
            
        #model parameters
        self.theta_ini = np.full(self.Nfeature_model, 0)
        self.theta_fit = np.full(self.Nfeature_model, np.nan)
        
        #cost
        self.cost_train = np.nan
        self.cost_valid = np.nan

        #performance
        self.precision_train = np.nan
        self.precision_valid = np.nan
        
        self.recall_train = np.nan
        self.recall_valid = np.nan

        self.accuracy_train = np.nan
        self.accuracy_valid = np.nan

        
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
        
        
    def _sigmoid(self,X,theta):
        z = np.matmul(X,theta.T)
        
        #print(theta,z)
        
        return (1+np.exp(-z))**-1

    
    def _hypothesis(self,X,theta):
        return self._sigmoid(X, theta) 
        
        
    def _cost_function(self,theta, X, Y):
        m = X.shape[0]
        h = self._hypothesis(X,theta)
        return -(1/m)*np.sum(Y*np.log(h) + (1-Y)*np.log(1-h))

    
    def _gradient(self,theta, X, Y):
        m = X.shape[0]
        h = self._hypothesis(X,theta)
        return (1/m) * np.dot(X.T, (h-Y))
    
    
    def fit(self,**kwargs):
        
        for key, value in kwargs.items():
            if key == 'optimizer': self.optimizer = value
            if key == 'theta_ini': self.theta_ini = value
                
        opt_weights = fmin_tnc(
            func=self._cost_function, x0=self.theta_ini, fprime=self._gradient,
            args=(self.X_train, self.Y_train)
        )
        
        self.theta_fit = opt_weights[0]
        
        self.cost_train = self._cost_function(self.theta_fit, self.X_train, self.Y_train)

    def predict(self,X_in):
        
        #re-scale input features
        X_in = X_in - self.train_mean
        X_in = X_in / self.train_var
        
        ### add higher orders
        X_in = self._add_higher_orders(X_in,self.poly_order)
        
        ### add 1 as first column bias terms
        X_in = np.insert(X_in,0,1, axis=1)
        
        sig = self._sigmoid(X_in,self.theta_fit)
        Y_out = np.zeros(len(sig))
        Y_out[sig>0.5]=1

        return Y_out

        
    def performance(self, X_in, Y_in, dataset):
        
        Y_model = self.predict(X_in)
        
        true_positive = (Y_in==1) & (Y_model==1)
        false_positive = (Y_in==1) & (Y_model==0)
        true_negative =  (Y_in==0) & (Y_model==0)
        false_negative =  (Y_in==0) & (Y_model==1)

        Ntp = len(true_positive[true_positive==True])
        Nfp = len(true_positive[false_positive==True])
        Ntn = len(true_positive[true_negative==True])
        Nfn = len(true_positive[false_negative==True])

        if self.verbose>0:
            print('')
            if dataset=='training': print('performance on training set:')
            if dataset=='validation': print('performance on validation set:')

            print('')
            print('true positive: ',Ntp)
            print('false positive: ',Nfp)
            print('true negative: ',Ntn)
            print('false negative: ',Nfn)
            

        if dataset=='training':
            if Ntp + Nfp > 0: self.precision_train = Ntp / (Ntp + Nfp)
            if Ntp + Nfn > 0: self.recall_train = Ntp / (Ntp + Nfn)
            if Ntp+Ntn+Nfp+Ntn > 0: self.accuracy_train = (Ntp+Ntn) / (Ntp+Ntn+Nfp+Ntn)        
            if self.verbose>0:
                print('')
                print('precision: ',self.precision_train)
                print('recall: ',self.recall_train)
                print('accuracy: ',self.accuracy_train)

        if dataset=='validation':
            if Ntp + Nfp > 0: self.precision_valid = Ntp / (Ntp + Nfp)
            if Ntp + Nfn > 0:self.recall_valid = Ntp / (Ntp + Nfn)
            if Ntp+Ntn+Nfp+Ntn > 0: self.accuracy_valid = (Ntp+Ntn) / (Ntp+Ntn+Nfp+Ntn)        
            if self.verbose>0:
                print('')
                print('precision: ',self.precision_valid)
                print('recall: ',self.recall_valid)
                print('accuracy: ',self.accuracy_valid)


# # features used for cliassification

# In[50]:


features = ['rj', 'nuvr']
#features = ['rj', 'nuvr', 'rk']
#features = ['rj', 'nuvr', 'rk', 'ur', 'm_r']

cat = cat.dropna(subset = features)


# # define subsets for training, testing, validation

# In[51]:


select_class = select_ellis
#select_class = select_disks_dd
#select_class = select_disks_bd
#select_class = select_irre

cat_class = cat.copy()

cat_class['gal_class'] = 0
cat_class.loc[select_class, 'gal_class']=1


# In[52]:


frac_train, frac_valid, frac_test = 0.5,0.5,0.0

rnd = np.random.random(len(cat_class))

cat_train = cat_class.loc[rnd <=frac_train]
cat_valid = cat_class.loc[((frac_train < rnd) & (rnd<frac_train+frac_valid))]
cat_test = cat_class.loc[frac_train+frac_valid <= rnd]

#len(cat_train)/len(cat), len(cat_val)/len(cat), len(cat_test)/len(cat)


# In[53]:


len(cat_test)


# # initialize data matrix X

# In[54]:


X_train = cat_train[features].values
Y_train = cat_train['gal_class']

X_valid = cat_valid[features].values
Y_valid = cat_valid['gal_class']


# In[ ]:





# In[55]:


theta_ini


# In[56]:


5#for order in range(3):

model = logistic_regression(X_train, Y_train, poly_order=2, verbose=1)
model.summary()
model.fit(theta_ini = -0.3 + 0.3*np.random.rand(model.Nfeature_model))
#model.cost_train
model.performance(X_train, Y_train,'training')
model.performance(X_valid, Y_valid,'validation')


# In[46]:


Ndat = 10000
Nfeature = 2

#random feature vectors for Ndat data points
f1 = -1+np.random.rand(Ndat)*3
f2 = -1+np.random.rand(Ndat)*8

# concat input vectors to data matrix
X_rand = np.concatenate((f1,f2)).reshape(Nfeature,Ndat).T


# In[47]:


Y_rand = model.predict(X_rand)


# In[59]:


plt.scatter(X_rand[:,0], X_rand[:,1], c=Y_rand, s=0.1)
#plt.scatter(cat.rj, cat.nuvr, s=0.1, c='r')
plt.scatter(cat_class[select_class==False].rj, cat_class[select_class==False].nuvr, s=0.1, c='b')
plt.scatter(cat_class[select_class==True].rj, cat_class[select_class==True].nuvr, s=0.1, c='r')
#plt.scatter(cat[select_ellis==False].rj, cat[select_ellis==False].nuvr, s=0.1, c='b')
#plt.scatter(cat[select_disks_dd==False].rj, cat[select_disks_dd==False].nuvr, s=0.1, c='b')


# # convergence tests

# # convert notebook to python script and remove this command from script

# In[24]:


