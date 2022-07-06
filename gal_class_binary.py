#!/usr/bin/env python
# coding: utf-8

# # references:
# - https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148
# - https://towardsdatascience.com/logistic-regression-with-python-using-optimization-function-91bd2aee79b
# - https://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from LogisticRegression import BinaryClass


# # read data

# In[2]:


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

# In[3]:


z_min, z_max = 0.2, 2.0
mag_min, mag_max = -25,24


# ### intrinsic size cut

# In[4]:


ang_psf = 0.085 # psf angle in arcsec
ang_pix = 0.03 # pixel angle in arcsec

ang_min = ang_psf*1.0 # min angle in arcsec
app_mag_max=24


# In[5]:


from colossus.cosmology import cosmology


# In[6]:


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


# In[7]:


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

# In[8]:


type_ZEST = 2 # late type
bulg1, bulg2 = 2,3 # disk dominated
irre1, irre2 = 9,9 # no values

select_disks_dd = (cat.type_ZEST==type_ZEST )                    & ((cat.bulg == bulg1) | (cat.bulg == bulg2) )                    & ((cat.irre == irre1) | (cat.irre == irre2) )


# ### disks (bulge dominated)

# In[9]:


type_ZEST = 2 # late type
bulg1, bulg2 = 0,1 # bulge dominated
irre1, irre2 = 9,9 # no values

select_disks_bd = (cat.type_ZEST==type_ZEST )                    & ((cat.bulg == bulg1) | (cat.bulg == bulg2) )                    & ((cat.irre == irre1) | (cat.irre == irre2) )


# ### ellipticals

# In[10]:


type_ZEST = 1 # early type
bulg1, bulg2 = 9,9 # no values
irre1, irre2 = 0,1 # regular

select_ellis = (cat.type_ZEST==type_ZEST )                    & ((cat.bulg == bulg1) | (cat.bulg == bulg2) )                    & ((cat.irre == irre1) | (cat.irre == irre2) )


# ### iregular

# In[11]:


type_ZEST = 3 # late type
bulg1, bulg2 = 9,9 # no values
irre1, irre2 = 9,9 # no values

select_irre = (cat.type_ZEST==type_ZEST )                    & ((cat.bulg == bulg1) | (cat.bulg == bulg2) )                    & ((cat.irre == irre1) | (cat.irre == irre2) )


# In[12]:


fig,ax = plt.subplots(1,4, figsize=(20,5))

ax[0].scatter(cat[select_disks_dd].rj, cat[select_disks_dd].nuvr, s=0.1, c='k')
ax[1].scatter(cat[select_disks_bd].rj, cat[select_disks_bd].nuvr, s=0.1, c='k')
ax[2].scatter(cat[select_ellis].rj, cat[select_ellis].nuvr, s=0.1, c='k')
ax[3].scatter(cat[select_irre].rj, cat[select_irre].nuvr, s=0.1, c='k')

ax[0].set_title('disks (disk dominated)')
ax[1].set_title('disks (bulge dominated)')
ax[2].set_title('ellipticals')
ax[3].set_title('iregulars')

ax[0].set_ylabel('$nuv-r$')

for ix in range(4):
    ax[ix].set_xlabel('r-j')

plt.show()


# # fit model to training data

# In[13]:


def uniform_class_size(cat_in):
    cat_class0 = cat_in[cat_in.gal_class==0]
    cat_class1 = cat_in[cat_in.gal_class==1]

    Nclass0 = len(cat_class0)
    Nclass1 = len(cat_class1)

    Nclass_min = min(Nclass0, Nclass1)

    if(Nclass0>Nclass_min):
        rnd = np.random.random(Nclass0)
        cat_class0 = cat_class0[rnd <= Nclass_min/Nclass0]

    if(Nclass1>Nclass_min):
        rnd = np.random.random(Nclass1)
        cat_class1 = cat_class1[rnd <= Nclass_min/Nclass1]

    return pd.concat([cat_class0, cat_class1])
    


# In[14]:


def prep_XY(cat, select_class, features, **kwargs):
        
    uni_class = True
    
    for key, value in kwargs.items():
        if key == 'uni_class': uni_class = value

    cat_class = cat.copy()

    #delete rows with nans in feature vector
    cat_class = cat_class.dropna(subset = features)

    cat_class['gal_class'] = 0
    cat_class.loc[select_class, 'gal_class']=1


    # ======== make equal fraction of all classes ========
    if uni_class: cat_class = uniform_class_size(cat_class)


    # ======== devide trainig and test sets ========
    frac_train, frac_valid, frac_test = 0.5,0.5,0.0
    rnd = np.random.random(len(cat_class))
    cat_train = cat_class.loc[rnd <=frac_train]
    cat_valid = cat_class.loc[((frac_train < rnd) & (rnd<frac_train+frac_valid))]
    cat_test = cat_class.loc[frac_train+frac_valid <= rnd]


    # ======== initialize data matrix X ========
    X_train = cat_train[features].values
    Y_train = cat_train['gal_class']

    X_valid = cat_valid[features].values
    Y_valid = cat_valid['gal_class']

    return X_train, Y_train, X_valid, Y_valid


# In[15]:


class results:
    
    def  __init__(self, features, gal_type):

        self.features = features
        
        self.gal_type=gal_type
    
        self.precision_train, self.precision_valid = [],[]
        self.recall_train, self.recall_valid = [],[]
        self.accuracy_train, self.accuracy_valid = [],[]
        self.Nparams = []


# # select elliptical galaxies in color-color plane

# In[16]:


gal_type = 'ellis'
select_class = select_ellis

vfeatures = ['rj', 'nuvr']

vpoly_orders = [1,2,3,4]


# ### fit model and plot decision bounderies

# In[17]:


Ndat = 100000
Nfeature = len(vfeatures)

#random feature vectors for Ndat data points
x_min = cat[vfeatures[0]].min()
x_max = cat[vfeatures[0]].max()
y_min = cat[vfeatures[1]].min()
y_max = cat[vfeatures[1]].max()

f1 = x_min + (x_max-x_min)*np.random.rand(Ndat)
f2 = y_min + (y_max-y_min)*np.random.rand(Ndat)


# concat input vectors to data matrix
X_rand = np.concatenate((f1,f2)).reshape(Nfeature,Ndat).T

len(cat[select_ellis==True]), len(cat[select_ellis==False]), len(cat)


# In[18]:


np.log(10**-6)


# In[19]:


fig,ax = plt.subplots(2,len(vpoly_orders), figsize=(16,8), sharex=True, sharey=True)

fontsize=16

for i in range(len(vpoly_orders)):
    
    o=vpoly_orders[i]
    
    ax[0,i].set_title('polynomial order=  ' + str(o),fontsize=fontsize)

    
    res = results(vfeatures, gal_type)

    X_train, Y_train, X_valid, Y_valid = prep_XY(cat, select_class, vfeatures)

    model = BinaryClass(X_train, Y_train, poly_order=o, verbose=0)
    model.fit(theta_ini = -0.01 + 0.01*np.random.rand(model.Nfeature_model))

    Y_rand = model.predict(X_rand)

    ax[0,i].scatter(X_rand[Y_rand==1][:,0], X_rand[Y_rand==1][:,1], s=5, c='lightgrey', alpha=1)
    ax[1,i].scatter(X_rand[Y_rand==1][:,0], X_rand[Y_rand==1][:,1], s=5, c='lightgrey', alpha=1)

    ax[0,i].scatter(cat[select_ellis==True].rj, cat[select_ellis==True].nuvr, s=0.2, c='r', alpha=0.1)
    ax[1,i].scatter(cat[select_ellis==False].rj, cat[select_ellis==False].nuvr, s=0.2, c='b', alpha=0.1)

    
x_label = x_max*1.05
y_label = y_min + 0.3*(y_max-y_min)

ax[0,-1].text(x_label,y_label,'ellipticals',rotation=-90, fontsize=fontsize);
ax[1,-1].text(x_label,y_label,'no ellipticals',rotation=-90, fontsize=fontsize);

for i in range(len(vpoly_orders)):
    ax[-1,i].set_xlabel(vfeatures[0], fontsize=fontsize)
    
for j in range(2):
    ax[j,0].set_ylabel(vfeatures[1], fontsize=fontsize)
    
for i in range(len(vpoly_orders)):
    for j in range(2):
        ax[j,i].set_xlim(x_min, x_max)
        ax[j,i].set_ylim(y_min, y_max)
        
plt.show()


# ### model performance for 2nd order polynomial

# In[20]:


vfeatures = ['rj', 'nuvr', 'q_app']

X_train, Y_train, X_valid, Y_valid = prep_XY(cat, select_ellis, vfeatures, uni_class=True)

model = BinaryClass(X_train, Y_train, poly_order=2, verbose=1)
model.summary()
model.fit(theta_ini = -0.01 + 0.01*np.random.rand(model.Nfeature_model))

print('\n #### same sample size for all classes (as used in training)####')
X_train, Y_train, X_valid, Y_valid = prep_XY(cat, select_ellis, vfeatures, uni_class=True)
model.performance(X_valid, Y_valid,'validation')

print('\n #### different sample size for different classes (as in input catalogue)####')
X_train, Y_train, X_valid, Y_valid = prep_XY(cat, select_ellis, vfeatures, uni_class=False)
model.performance(X_valid, Y_valid,'validation')


# ### galaxy types that are classified as ellipticals

# In[21]:


ellis = cat[select_ellis].dropna()
disks_dd = cat[select_disks_dd].dropna()
disks_bd = cat[select_disks_bd].dropna()
irre = cat[select_irre].dropna()

X_ellis = ellis[vfeatures].values
X_disks_dd = disks_dd[vfeatures].values
X_disks_bd = disks_bd[vfeatures].values
X_irre = irre[vfeatures].values

Y_ellis = model.predict(X_ellis)
Y_disks_dd = model.predict(X_disks_dd)
Y_disks_bd = model.predict(X_disks_bd)
Y_irre = model.predict(X_irre)

print('from all galaxies classified as ellipticals:\n')
print(len(X_ellis[Y_ellis==1]), 'ellipticals')
print(len(X_disks_dd[Y_disks_dd==1]), 'disks (disk dominated)')
print(len(X_disks_bd[Y_disks_bd==1]), 'disks (bulge dominated)')
print(len(X_irre[Y_irre==1]), 'irregular')


# # test performance for different, galaxy types|, features, polynomial orders

# In[22]:


gal_type1 = 'ellis'
gal_type2 = 'disks_dd'
gal_type3 = 'disks_bd'
gal_type4 = 'irre'

vgal_type = [gal_type1,gal_type2,gal_type3,gal_type4]

features1 = ['rj', 'nuvr']
features2 = ['rj', 'nuvr', 'rk']
features3 = ['rj', 'nuvr', 'rk', 'm_r']
features4 = ['rj', 'nuvr', 'rk', 'm_r', 'q_app']

vfeatures = [features1,features2,features3, features4]

vpoly_orders = [1,2,3,4,5]


# In[23]:


vres_gt = []

for gt in vgal_type:

    print(gt)
    
    if gt == 'ellis': select_class = select_ellis
    if gt == 'disks_dd': select_class = select_disks_dd
    if gt == 'disks_bd': select_class = select_disks_bd
    if gt == 'irre': select_class = select_irre

    vres_ft = []

    for ft in vfeatures:

        res = results(ft, gt)

        X_train, Y_train, X_valid, Y_valid = prep_XY(cat, select_class, ft, uni_class=True)

        for o in vpoly_orders:

            model = BinaryClass(X_train, Y_train, poly_order=o, verbose=0)
            model.fit(theta_ini = -0.01 + 0.01*np.random.rand(model.Nfeature_model))

            res.Nparams.append(len(model.theta_fit))

            model.performance(X_train, Y_train,'training')
            model.performance(X_valid, Y_valid,'validation')

            res.precision_train.append(model.precision_train)
            res.precision_valid.append(model.precision_train)

            res.recall_train.append(model.recall_train)
            res.recall_valid.append(model.recall_valid)

            res.accuracy_train.append(model.accuracy_train)
            res.accuracy_valid.append(model.accuracy_valid)


        vres_ft.append(res)
        
    vres_gt.append(vres_ft)


# ### plot results

# In[24]:


fig, ax = plt.subplots(3,4,figsize=(15,9), sharex='col', sharey='row')

fontsize=16

for vres_ft in vres_gt:

    for i in range(len(vres_ft)):

        res = vres_ft[i]

        ax[0,i].set_title(res.features, fontsize=fontsize)

        if res.gal_type == 'ellis': color = 'r'
        if res.gal_type == 'disks_bd': color = 'purple'
        if res.gal_type == 'disks_dd': color = 'b'
        if res.gal_type == 'irre': color = 'g'
        
        ax[0,i].plot(vpoly_orders, res.accuracy_train, marker='o', ls=':', label=res.gal_type, c=color)
        ax[1,i].plot(vpoly_orders, res.precision_train, marker='o', ls=':', label=res.gal_type, c=color)
        ax[2,i].plot(vpoly_orders, res.recall_train, marker='o', ls=':', label=res.gal_type, c=color)

        ax[0,i].plot(vpoly_orders, res.accuracy_valid, marker='o', c=color)
        ax[1,i].plot(vpoly_orders, res.precision_valid, marker='o', c=color)
        ax[2,i].plot(vpoly_orders, res.recall_valid, marker='o', c=color)

        ax[0,i].grid()

for i in range(4):
    ax[2,i].set_xlabel('polynomial order',fontsize=fontsize)
    
    ax[0,i].axhline(y=0.6, c='k', ls='--')
    ax[1,i].axhline(y=0.9, c='k', ls='--')
    ax[2,i].axhline(y=0.6, c='k', ls='--')
    ax[2,i].axhline(y=0.9, c='k', ls='--')

ax[0,0].set_ylabel('accuracy', fontsize=fontsize)
ax[1,0].set_ylabel('precision',fontsize=fontsize)
ax[2,0].set_ylabel('recall', fontsize=fontsize)

ax[1,0].legend()

plt.show()


# # convert notebook to python script and remove this command from script

# In[25]:


get_ipython().system('jupyter-nbconvert --to script gal_class.ipynb')

with open('gal_class_binary.py', 'r') as f:
    lines = f.readlines()
with open('gal_class_binary.py', 'w') as f:
    for line in lines:
