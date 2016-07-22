import numpy as np
import matplotlib.pyplot as plt
import emcee
def my_model(lam,center1,height1,width1,center2,height2,width2):
    model_1 =  height1 * (width1/2.)**2 / ((lam - center1)**2 + (width1/2.)**2)
    model_2 =  height2 * (width2/2.)**2 / ((lam - center2)**2 + (width2/2.)**2)
    return model_1 + model_2

#Set up our fake dataset
my_lam = np.arange(4500,5500,5)#Angstroms
NOISE = 5.0
my_data = my_model(my_lam,5002,100,30,5052,50,40) + np.random.normal(0, NOISE,len(my_lam))
my_sigmas = np.zeros(len(my_lam)) + NOISE
plt.plot(my_lam,my_data)
plt.xlabel(r'$\lambda$')
plt.ylabel('Flux')
plt.title('Beautiful Data')
plt.show()

#Let's make a final METRIC, Log Likelihood
def metric_log_likelihood(theta,lam,data,sigmas):
    c1, h1, w1,c2,h2,w2 = theta
    return -0.5*(np.sum((data-my_model(lam,c1,h1,w1,c2,h2,w2))**2/sigmas**2 + np.log(sigmas**2)))

def metric_chi2(theta,lam,data,sigmas):
    c1, h1, w1,c2,h2,w2 = theta
    return np.sum((my_model(lam,c1,h1,w1,c2,h2,w2)-data)**2/(sigmas)**2)

#Run "emcee"
ndim, nwalkers = 6, 100
p0 = [5000,90,20,5100,40,20]
pos = [p0 + .1*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, metric_log_likelihood, args=(my_lam, my_data, my_sigmas))
sampler.run_mcmc(pos, 1000)
samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
plt.plot(samples[:,0])
plt.xlabel('Iteration Number')
plt.ylabel('Height')
plt.title('A Sample Chain')
plt.show()

bc1 = np.percentile(samples[:,0],50)
bh1 = np.percentile(samples[:,1],50)
bw1 = np.percentile(samples[:,2],50)
bc2 = np.percentile(samples[:,3],50)
bh2 = np.percentile(samples[:,4],50)
bw2 = np.percentile(samples[:,5],50)


plt.plot(my_lam,my_model(my_lam,bc1,bh1,bw1,bc2,bh2,bw2),'r')
plt.plot(my_lam,my_data,'b')
plt.show()

print metric_chi2([bc1,bh1,bw1,bc2,bh2,bw2],my_lam,my_data,my_sigmas)/len(my_sigmas)

np.savetxt('test.dat', np.c_[my_lam,my_data,my_sigmas])