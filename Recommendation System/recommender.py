'''
Created on May 4, 2018

@author: nishant.sethi
'''

import numpy as np
import pandas as pd



data=pd.read_csv("movie_rating.csv")
data_rated=pd.read_csv("who_rated.csv")

'''cost function'''
def compute_cost_function(nm,nu,theta, x,y,r):
    p=np.zeros(y.shape) 
    for i in range(0,nm):
        for j in range(0,nu):
            p[i][j]=theta[j].T.dot(x[i])
    error=0
    for i in range(0,nm):
        for j in range(0,nu):
            if r[i][j]==1:
                error+=(p[i][j]-y[i][j])**2
    error=1/2 *error
    return error       

'''gradient descent'''
def gradient_descent(alpha,ep=0.0001, max_iter=1500):
    converged = False
#     m = x.shape # number of samples
    '''initial theta'''
    nm,nu=data.shape
    nm=5
    nu=4
    #nu=4
    n=3
    theta=np.random.randint(1,6,(nu,n))
    print("Theta values")
    print(theta)
    temp=np.zeros((nu,n))
    grad=np.zeros((nu,n))
    x=np.random.random((nm,n))
    print("X values")
    print(x)
    y=np.random.randint(1,6,(nm,nu))
    #y=np.array(data)
    r=np.random.randint(0,2,(nm,nu))
    for i in range(0,nm):
            for j in range(0,nu):
                if r[i][j]!=1:
                    y[i][j]=0
    #r=np.array(data_rated)
#     m = y.shape
#     p=x.T.dot(theta)
    #print(p.shape)
    '''total error, J(theta)'''
    J = compute_cost_function(nm,nu,theta, x, y,r)
    print('J=', J);
    '''Iterate Loop'''
    num_iter = 0
    while not converged:
        '''for each training sample, compute the gradient (d/d_theta j(theta))'''
#         grad0 = sum([(t0 + t1*np.asarray([x[i]]) - y[i]) for i in range(m)]) 
#         grad1 = sum([(t0 + t1*np.asarray([x[i]]) - y[i])*np.asarray([x[i]]) for i in range(m)])
        a=0
        for k in range(0,n):
            for i in range(0,nm):
                for j in range(0,nu):
                    if r[i][j]==1:
                        theta[j][k]-=alpha*((theta[j].T.dot(x[i])-y[i][j])*x[i][k]) 
                        theta[j][k]=round(theta[j][k],3) 
        
        
        for k in range(0,n):
            for i in range(0,nm):
                for j in range(0,nu):
                    if r[i][j]==1:
                        x[i][k]-=alpha*((theta[j].T.dot(x[i])-y[i][j])*theta[j][k])
                        x[i][k]=round(x[i][k],3)
                        if x[i][k]>1:
                            x[i][k]=1
                        if x[i][k]<0:
                            x[i][k]=0
                    
                    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#         a=0
#         for i in range(0,m[0]):
#             for j in range(0,m[1]):
# #                 print(theta[i])
# #                 print(p[i][j])
# #                 print(y[i][j])
#                 a=theta[0][j]*x[0][j]+theta[1][j]*x[1][j]
#                 a=a-y[i][j]
#                 #a=sum(theta[i]*(p[i][j]-y[i][j]))
#                 grad[i][j]=a
#         
#         '''update the theta_temp'''
# #         temp0 = t0 - alpha * grad0
# #         temp1 = t1 - alpha * grad1
#         for i in range(0,m[0]):
#             for j in range(0,m[1]):
#                 temp[i][j]=theta[i][j]-alpha*grad[i][j]   
#         '''update theta'''
# #         t0 = temp0
# #         t1 = temp1
        theta=temp
        '''mean squared error'''
        e = compute_cost_function(nm,nu, theta, x, y,r)
        print ('J = ', e)
        J = e   # update error 
        num_iter += 1  # update iter   
        if num_iter==max_iter:
            print ('Max interactions exceeded!')
            converged = True
    return (theta,x,y,r)

a,b,c,d=gradient_descent(0.01)
print(a.shape,b.shape)
print(a)
print(b)
print(c)
print(d)
predicted=np.zeros(c.shape)
for i in range(0,c.shape[0]):
    for j in range(0,c.shape[1]):
        predicted[i][j]=round(a[j].T.dot(b[i]),3)
print(predicted)
# p=pd.DataFrame(b.dot(a.T))
# print(p.head())
# print(pd.DataFrame(c).head())



def recommender():
    y=pd.read_csv("movie_rating.csv")
    a,b=y.shape
    r=pd.read_csv("who_rated.csv")
    x=np.random.random((2,1682))
    theta=np.random.random((2,944))
    predicted=pd.DataFrame(round(x.T.dot(theta),2))
    print(predicted.shape)
#     err=0
#     err=cost(predicted,y,r,theta,a,b)
#     res = minimize(cost,x,args = [predicted,y,r,theta,a,b])
#    print(res.x)

#recommender()