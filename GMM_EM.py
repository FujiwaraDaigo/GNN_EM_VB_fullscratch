#!/usr/bin/env python
# coding: utf-8

# # GMM

# ## EM algorythm

# $$
# \pi_{k}^{\mathrm{new}}=\frac{N_{k}}{N}\\
# \boldsymbol{\mu}_{k}^{\text { new }}=\frac{1}{N_{k}} \sum_{n=1}^{N} \gamma_{n k} \boldsymbol{x}_{n}\\
# \boldsymbol\Lambda_{k}^{-1 \mathrm{new}}=\frac{1}{N_{k}} \sum_{n=1}^{N} \gamma_{n k} \boldsymbol{x}_{n} \boldsymbol{x}_{n}^{T}-\boldsymbol{\mu}_{k}^{\mathrm{new}} \boldsymbol{\mu}_{k}^{\mathrm{new} \mathrm{T}}
# $$
# where,
# $$
# N_{k}=\sum_{n=1}^{N} \gamma_{n k}
# $$
# $$
# \gamma_{n k}=\frac{\pi_{k} N\left(x_{n} | \boldsymbol{\mu}_{k}, \mathbf{\Lambda}_{k}^{-1}\right)}{\sum_{k^{\prime}=1}^{K} \pi_{k^{\prime}} N\left(\boldsymbol{x}_{n} | \boldsymbol{\mu}_{k^{\prime}}, \boldsymbol{\Lambda}_{k^{\prime}}^{-1}\right)}
# $$

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import multivariate_normal as multi_gauss
from mpl_toolkits.mplot3d import Axes3D


# In[17]:


class EM_GMM:
    def __self__(self,K):
        self.K=K
        
    def E_step(self,X,pi,mu,Sigma):
        gamma=np.zeros((self.N,self.K))
        
        for n in range(self.N):
            for k in range(self.K):
                gamma[n,k]= pi[k]*self.gauss(X[n].reshape(self.D, 1), mu[k].reshape(self.D, 1), Sigma[k])
                
        gamma=gamma/np.sum(gamma, axis=1, keepdims=True)
        
        return gamma
        
    def M_step(self,X,gamma):
        Nk=np.sum(gamma,axis=0)
        pi=Nk/N
        mu=np.zeros((self.K,self.D))
        Sigma=np.zeros((self.K,self.D,self,D))
        temp=np.zeros((self.N,self.D,self.D))
        for n in range(self.N):
            temp[n]=np.dot(X[n].reshape(self.D,1),X[n].reshape(1,self.D))
        for k in range(self.K):
            mu[k]=np.average(X,axis=0,weights=gamma[:,k])
            Sigma[k]=np.average(temp,axis=0,weights=gamma[:,k])-np.dot(mu[k].reshape(self.D,1),mu[k].reshape(1,self.D))
        
        return pi,mu,Sigma
        
    def fit(self,X,T=100):
        #initialize
        self.D = len(X[0])
        self.N=X.shape[0]
        pi=np.ones(self.K)/self.K
        mu0 = np.mean(X,axis=0)
        mu=np.zeros((self.K,self.D))
        for i in range(self.K):
            muk[i]=mu0
        Sigma=np.zeros((self.K,self.D,self.D))
        temp=np.dot(X.T,X)/N
        for k in range(self.K):
            Sigma[k]=temp
        
        #EM algorythm
        record=[]
        for step in range(T):
            gamma = self.E_step(X,pi,mu,Sigma)
            pi,mu,Sigma=self.M_step(X,gamma)
            
            logL = self.loglikelihood(X,pi,mu,Sigma)
            print("iter: %d, log likelihood: %f" % (step, logL))
            record.append([step, logL])
            if step == 0:
                oldL = logL
            else:
                if logL - oldL < 1e-5:
                    break
                else:
                    oldL = logL
        return np.array(record)
    
        def gauss(self, x, mu, Sigma):
            x=x.reshape(self.D,1)
            mu=mu.reshape(self.D,1)
            return   np.exp(-0.5*(x-mu).T @ np.linalg.inv(Sigma)@(x-mu))/(np.linalg.det(Sigma) * np.sqrt(2 *np.pi)**self.D)
        
        def loglikelihood(self, X,pi,mu,Sigma):
        # compute log likelohood
            logL = 0
            for n in range(self.N):
                L = 0
                for k in range(self.n_cluster):
                    L += pi[k] * self.gauss(X[n].reshape(self.D, 1), self.mu[k].reshape(self.D, 1), self.Sigma[k])
                logL += np.log(L)
            return logL
        
        def classify(self, X):
            gamma = np.zeros((self.N,self.K)
            for n in range(N):
                for k in range(self.K):
                    gamma[n, k] = self.pi[k] * self._gauss(X[n].reshape(self.D, 1), self.means[k].reshape(self.D, 1), self.cov[k])
            return posterior/np.sum(posterior, axis=1, keepdims=True), np.argmax(posterior, axis=1)


# In[16]:


a=np.array([[1,2],[3,4]])
np.average(a,axis=0,)


# In[11]:


X = np.loadtxt("/Users/daigofujiwara/Documents/授業資料/パターン認識特論/Report4/x.csv", delimiter=",")
X.shape[0]


# In[ ]:




class EM_GMM:
    def __self__(self,K):
        self.K=K
        
    def E_step(self,X,pi,mu,Sigma):
        gamma=np.zeros((self.N,self.K))
        
        for n in range(self.N):
            for k in range(self.K):
                gamma[n,k]= pi[k]*self.gauss(X[n].reshape(self.D, 1), mu[k].reshape(self.D, 1), Sigma[k])
                
        gamma=gamma/np.sum(gamma, axis=1, keepdims=True)
        
        return gamma
        
    def M_step(self,X,gamma):
        Nk=np.sum(gamma,axis=0)
        pi=Nk/N
        mu=np.zeros((self.K,self.D))
        Sigma=np.zeros((self.K,self.D,self,D))
        temp=np.zeros((self.N,self.D,self.D))
        for n in range(self.N):
            temp[n]=np.dot(X[n].reshape(self.D,1),X[n].reshape(1,self.D))
        for k in range(self.K):
            mu[k]=np.average(X,axis=0,weights=gamma[:,k])
            Sigma[k]=np.average(temp,axis=0,weights=gamma[:,k])-np.dot(mu[k].reshape(self.D,1),mu[k].reshape(1,self.D))
        
        return pi,mu,Sigma
        
    def fit(self,X,T=100):
        #initialize
        self.D = len(X[0])
        self.N=X.shape[0]
        pi=np.ones(self.K)/self.K
        mu0 = np.mean(X,axis=0)
        mu=np.zeros((self.K,self.D))
        for i in range(self.K):
            muk[i]=mu0
        Sigma=np.zeros((self.K,self.D,self.D))
        temp=np.dot(X.T,X)/N
        for k in range(self.K):
            Sigma[k]=temp
        
        #EM algorythm
        record=[]
        for step in range(T):
            gamma = self.E_step(X,pi,mu,Sigma)
            pi,mu,Sigma=self.M_step(X,gamma)
            
            logL = self.loglikelihood(X,pi,mu,Sigma)
            print("iter: %d, log likelihood: %f" % (step, logL))
            record.append([step, logL])
            if step == 0:
                oldL = logL
            else:
                if logL - oldL < 1e-5:
                    break
                else:
                    oldL = logL
        return np.array(record)
    
        def gauss(self, x, mu, Sigma):
            x=x.reshape(self.D,1)
            mu=mu.reshape(self.D,1)
            return   np.exp(-0.5*(x-mu).T @ np.linalg.inv(Sigma)@(x-mu))/(np.linalg.det(Sigma) * np.sqrt(2 *np.pi)**self.D)
        
        def loglikelihood(self, X,pi,mu,Sigma):
        # compute log likelohood
            logL = 0
            for n in range(self.N):
                L = 0
                for k in range(self.n_cluster):
                    L += pi[k] * self.gauss(X[n].reshape(self.D, 1), self.mu[k].reshape(self.D, 1), self.Sigma[k])
                logL += np.log(L)
            return logL
        
        def classify(self, X,pi,mu,Sigma):
            gamma = np.zeros((self.N,self.K)
            for n in range(self.N):
                for k in range(self.K):
                    gamma[n, k] = pi[k] * self.gauss(X[n].reshape(self.D, 1), self.mu[k].reshape(self.D, 1), self.Sigma[k])
            gamma=gamma/np.sum(gamma, axis=1, keepdims=True)
                             
            return gamma, np.argmax(gamma, axis=1)