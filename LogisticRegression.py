from scipy.optimize import minimize,fmin_tnc, fmin_cg, fmin_bfgs, fmin_l_bfgs_b
import numpy as np
from matplotlib import pyplot as plt


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

        self.X_train = (self.X_train - self.train_mean)
        self.X_train[:,self.train_var>0]/self.train_var[self.train_var>0]
        
        
        ### add higher orders
        self.X_train = self._add_higher_orders(self.X_train,self.poly_order)
        
        ### add 1 as first column bias terms
        self.X_train = np.insert(self.X_train,0,1, axis=1)
        
        self.Nfeature_model = self.X_train.shape[1]
            
        #model parameters
        self.theta_ini = np.full(self.Nfeature_model, 0)
        self.theta_fit = np.full(self.Nfeature_model, 0)
        
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
        
        print('size of training sample (class 0+1): ', len(self.Y_train))
        print('size of training sample (class 0): ', len(self.Y_train[self.Y_train==0]))
        print('size of training sample (class 1): ', len(self.Y_train[self.Y_train==1]))
        
        
        
    def _sigmoid(self,X,theta):
        
        z = np.matmul(X,theta.T)
        
        sig = np.zeros(len(z))

        # for numerical stability
        zlim = 10
        lo = z < -zlim
        hi = z > zlim
        mid = (z>=-zlim) & (z<=zlim)

        sig[mid] = (1+np.exp(-z[mid]))**-1 
        #sig[lo] = (1+np.exp(zlim))**-1
        #sig[hi] = (1+np.exp(-zlim))**-1
        sig[lo] = 0.0000453978687024
        sig[hi] = 0.9999546021312976

        return sig
    
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
            if key == 'theta_ini': self.theta_ini = np.array(value)

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
        X_in = (X_in - self.train_mean)
        X_in[:,self.train_var>0]/self.train_var[self.train_var>0]
        
        #X_in = (X_in - self.train_mean)/ self.train_var
        
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
        
        TP = np.sum((Y_in==1) & (Y_model==1))
        FP = np.sum((Y_in==0) & (Y_model==1))
        TN =  np.sum((Y_in==0) & (Y_model==0))
        FN =  np.sum((Y_in==1) & (Y_model==0))

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
            if TP+TN+FP+TN > 0: self.accuracy_train = (TP+TN) / (TP+TN+FP+FN)        
            if self.verbose>0:
                print('')
                print('precision: ',self.precision_train)
                print('recall: ',self.recall_train)
                print('accuracy: ',self.accuracy_train)

        if dataset=='validation':
            if TP + FP > 0: self.precision_valid = TP / (TP + FP)
            if TP + FN > 0:self.recall_valid = TP / (TP + FN)
            if TP+TN+FP+TN > 0: self.accuracy_valid = (TP+TN) / (TP+TN+FP+FN)
            if self.verbose>0:
                print('')
                print('precision: ',self.precision_valid)
                print('recall: ',self.recall_valid)
                print('accuracy: ',self.accuracy_valid)





class MultiClass:
    
    def  __init__(self, X_train, Y_train ,**kwargs):
        
        self.verbose=0
        self.poly_order = 1
        self.uniform_class_size = True
        
        for key, value in kwargs.items():
            if key == 'poly_order': self.poly_order = value
            if key == 'verbose': self.verbose = value
            if key == 'uniform_class_size': self.uniform_class_size = value

        self.X_train = X_train
        self.Y_train = Y_train
        
        self.Nfeature_in = self.X_train.shape[1]
        
        self.Nclass = self.Y_train.max()+1#add +1 because classes are counted starting from zero

        self.optimizer = 'scipy_optimize'
    
        self.model_vec = []# for storing binary classification models for each class from One-vs-All approach

        
        # confusion matrix
        self.CoMa = np.zeros(self.Nclass**2).reshape(self.Nclass,self.Nclass)



    def summary(self):        
        
        
        print('polynomial order: ', self.poly_order)
        print('number of input features: ', self.Nfeature_in)
        print('number of classes: ', self.Nclass)
        
        for i in range(self.Nclass):
            print('size of training sample class', i,': ', len(self.Y_train[self.Y_train==i]))
        

    def fit(self,**kwargs):
        
        for key, value in kwargs.items():
            if key == 'optimizer': self.optimizer = value
            if key == 'theta_ini': self.theta_ini = value

        for i in range(self.Nclass):
            
            if self.verbose > 0:
                print('training on class ', i)
            
            Y_train_i = np.where(self.Y_train==i, 1, 0)
            model = BinaryClass(
                self.X_train, Y_train_i,
                poly_order=self.poly_order,
                verbose=self.verbose,
                uniform_class_size=self.uniform_class_size
            )
            
            model.fit(theta_ini = -0.1 + 0.1*np.random.rand(model.Nfeature_model))

            self.model_vec.append(model)
            
            
    def predict(self, X_in):
        
        Nrow = X_in.shape[0]

        sigmoid_max = np.zeros(Nrow)
        Y_out = np.zeros(Nrow)

        for iclass in range(len(self.model_vec)):
            
            sigmoid = self.model_vec[iclass].predict(X_in, binary=False)
            select = sigmoid > sigmoid_max    
            sigmoid_max[select] = sigmoid[select]
            Y_out[select] = iclass
        
        return Y_out
    
    def _get_coma(self, X_in, Y_in):

        Y_model = self.predict(X_in)

        self.CoMa = np.zeros(self.Nclass**2).reshape(self.Nclass,self.Nclass)

        for iclass in range(self.CoMa.shape[0]):
            for jclass in range(self.CoMa.shape[1]):
                self.CoMa[iclass, jclass] = np.sum((Y_in == iclass) & (Y_model == jclass))
        
    def plot_confusion_matrix(self, X_in, Y_in):
        
        self._get_coma(X_in, Y_in)
        
        plt.imshow(self.CoMa, interpolation=None, cmap='jet')
        plt.colorbar()
        plt.xlabel('predicted class')
        plt.ylabel('true class')
        plt.show()
    
        
    def performance(self, X_in, Y_in):
        
        Y_model = self.predict(X_in)

        for iclass in range(self.CoMa.shape[0]):

            TP = np.sum((Y_in==iclass) & (Y_model==iclass))
            FP = np.sum((Y_in!=iclass) & (Y_model==iclass))
            TN =  np.sum((Y_in!=iclass) & (Y_model!=iclass))
            FN =  np.sum((Y_in==iclass) & (Y_model!=iclass))

            precision = np.nan
            recall = np.nan
            accuracy = np.nan
            
            if TP + FP > 0: precision = TP / (TP + FP)
            if TP + FN > 0: recall = TP / (TP + FN)
            if TP+TN+FP+TN > 0: accuracy = (TP+TN) / (TP+TN+FP+TN)        
            
            print('class:',iclass)
            print('precision:\t',np.round(precision,3))
            print('recall:   \t',np.round(recall,3))
            print('accuracy: \t',np.round(accuracy,3))
            print('')
        











