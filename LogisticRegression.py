from scipy.optimize import minimize,fmin_tnc, fmin_cg, fmin_bfgs, fmin_l_bfgs_b

import numpy as np

class BinaryClass:
    
    def  __init__(self, X_train, Y_train ,**kwargs):

        self.verbose=0
        
        self.X_train = X_train
        self.Y_train = Y_train
        
        #set default values
        self.Nfeature_in = self.X_train.shape[1]
        
        self.poly_order = 1
    
        self.optimizer = 'scipy_optimize'
    
        self.uniform_class_size = True
    
        for key, value in kwargs.items():
            if key == 'poly_order': self.poly_order = value
            if key == 'verbose': self.verbose = value
            if key == 'uniform_class_size': self.uniform_class_size = value
            
        # re-size catalogue to have similar size of examples for each class
        if self.uniform_class_size :
            self.X_train, self.Y_train = self._uniform_class_size(self.X_train, self.Y_train)
        
        #re-scale input features
        self.train_mean = self.X_train.T.mean(1)       
        self.train_var = self.X_train.T.var(1)
        
        self.X_train = (self.X_train - self.train_mean)/ self.train_var
        
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
            
            
    def _uniform_class_size(self,X_in, Y_in):

        Nclass0 = len(Y_in[Y_in==0])
        Nclass1 = len(Y_in[Y_in==1])

        rnd = np.random.random(len(Y_in))

        if Nclass0 > Nclass1:    
            select0 = (Y_in==0) & (rnd < Nclass1/Nclass0)
            select1 = (Y_in==1)
        else:
            select0 = (Y_in==0)
            select1 = (Y_in==1) & (rnd < Nclass0/Nclass1)

        Y0 = Y_in[select0]
        Y1 = Y_in[select1]

        Y_out = np.concatenate([Y_in[select0], Y_in[select1]])
        X_out = np.concatenate([X_in[select0], X_in[select1]])

        return X_out, Y_out
    
            
    def summary(self):        
        print('polynomial order: ', self.poly_order)
        print('number of input features: ', self.Nfeature_in)
        print('number of model parameters: ', self.Nfeature_model)
        
        
    def _sigmoid(self,X,theta):
        
        z = np.matmul(X,theta.T)
        
        y = np.zeros(len(z))

        # for numerical stability
        z_lo = z<-10
        z_hi = z>10
        z_mid = (z>=-10) & (z<=10)

        y[z_lo] = 10**-6
        y[z_hi] = 1-10**-6
        y[z_mid] = (1+np.exp(-z[z_mid]))**-1

        return y
    
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
            disp=0,
            func=self._cost_function, x0=self.theta_ini, fprime=self._gradient,
            args=(self.X_train, self.Y_train)
        )
        
        self.theta_fit = opt_weights[0]
        
        self.cost_train = self._cost_function(self.theta_fit, self.X_train, self.Y_train)

        
    def predict(self, X_in ,**kwargs):
        
        binary = False
        for key, value in kwargs.items():
            if key == 'binary': binary = value

        
        #re-scale input features
        X_in = (X_in - self.train_mean)/ self.train_var
        
        ### add higher orders
        X_in = self._add_higher_orders(X_in,self.poly_order)
        
        ### add 1 as first column bias terms
        X_in = np.insert(X_in,0,1, axis=1)
        
        sig = self._sigmoid(X_in,self.theta_fit)

        if binary:
            Y_out = np.zeros(len(sig))
            Y_out[sig>0.5]=1
        else:
            Y_out = sig

        return Y_out

        
    def performance(self, X_in, Y_in, dataset):
        
        Y_model = self.predict(X_in, binary=True)
        
        true_positive = (Y_in==1) & (Y_model==1)
        false_positive = (Y_in==1) & (Y_model==0)
        true_negative =  (Y_in==0) & (Y_model==0)
        false_negative =  (Y_in==0) & (Y_model==1)

        TP = len(true_positive[true_positive==True])
        FP = len(false_positive[false_positive==True])
        TN = len(true_negative[true_negative==True])
        FN = len(false_negative[false_negative==True])

        if self.verbose>0:
            print('\n')
            if dataset=='training': print('==== performance on training set ====')
            if dataset=='validation': print('==== performance on validation set ====')

            print('')
            print('true positive: ',TP)
            print('false positive: ',FP)
            print('true negative: ',TN)
            print('false negative: ',FN)
            

        if dataset=='training':
            if TP + FP > 0: self.precision_train = TP / (TP + FP)
            if TP + FN > 0: self.recall_train = TP / (TP + FN)
            if TP+TN+FP+TN > 0: self.accuracy_train = (TP+TN) / (TP+TN+FP+TN)        
            if self.verbose>0:
                print('')
                print('precision: ',self.precision_train)
                print('recall: ',self.recall_train)
                print('accuracy: ',self.accuracy_train)

        if dataset=='validation':
            if TP + FP > 0: self.precision_valid = TP / (TP + FP)
            if TP + FN > 0:self.recall_valid = TP / (TP + FN)
            if TP+TN+FP+TN > 0: self.accuracy_valid = (TP+TN) / (TP+TN+FP+TN)        
            if self.verbose>0:
                print('')
                print('precision: ',self.precision_valid)
                print('recall: ',self.recall_valid)
                print('accuracy: ',self.accuracy_valid)
