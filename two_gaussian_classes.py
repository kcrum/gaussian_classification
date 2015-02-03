import numpy as np
import scipy.stats as st
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

'''
This code will draw events from two classes, each a function of some number of 
covariates chosen by the user. These events will be plotted on the two axes
chosen by the user. The Bayes decision boundary will also be drawn on this 
plot.
'''

# Make your plots prettier
from mpltools import style
style.use('ggplot')

# Set random seed
np.random.seed(4)

# Number of events pulled from classes
sizes = np.array([500,1000])
ntotal = sum(sizes)

# Classes well-separated along x and y axes, not so much along z.
means = np.array([[1,1],[-1,-1]])
#means = np.array([[1,1,1,1],[-1,-1,-1,-1]])

# Ensure means are same dimensionality
if len(means[0]) != len(means[1]):
    print 'Both classes must have same sized feature space. Check the means!'
    raise SystemExit

# Covariance matrices
covmat0 = np.array([[1.7, 0.5],
                    [0.5, 1.7]])
covmat1 = np.array([[2, -0.7],
                    [-0.7, 2]])

#covmat0 = np.array([[1.7, 0.5, -0.1, 0.2],
#                    [0.5, 1.1, 0.1, -0.1],
#                    [-0.1, 0.1, 1.4, 0.6],
#                    [0.2, -0.1, 0.6, 1.3]])
#
#covmat1 = np.array([[1.2, -0.5, 0.1, 0.2],
#                    [-0.5, 1.1, -0.1, 0.1],
#                    [0.1, -0.1, 0.9, 0.2],
#                    [0.2, -0.1, 0.2, 1.]])

covmats = [covmat0, covmat1]

# Ensure covmats are same dimensionality as each other and means.
if covmats[0].shape !=  covmats[1].shape:
    print 'Covariance matrices must have the same shape. Exiting!'
    raise SystemExit
if covmats[0].shape[0] != means[0].size or \
   covmats[0].shape[1] != means[0].size:
    print 'Covariance matrices must have shape that is square of the dimensionality of the means. Exiting!'
    raise SystemExit

# Ensure covmat is pos. def.
for cmat in covmats:
    if not np.all(np.linalg.eigvals(cmat) > 0):
        print 'Covariance matrix not positive definite. Exiting!'
        raise SystemExit

# Create PDFs
pdfs = [st.multivariate_normal(mean=m, cov=c) for m,c in zip(means,covmats)]

# Pull sample from each class
samples = [p.rvs(size=s) for p,s in zip(pdfs,sizes)]


def plot_samples(samples=samples, xind=0, yind=1, plotEigenvecs = True):
    '''
    "xind" and "yind" are the indices of the variables plotted on the x- and 
    y-axis, respectively.
    '''
    if xind == yind:
        print "You need to pass different indices to 'plot_samples'; exiting."
        raise SystemExit

    # Plot sample
    plt.scatter(samples[1][:,xind], samples[1][:,yind], c='b')
    plt.scatter(samples[0][:,xind], samples[0][:,yind], c='r')

    # Plot eigenvectors, if desired
    if plotEigenvecs:
        plot_eigenvectors(samples[0], xind, yind, c='yellow', label='0')
        plot_eigenvectors(samples[1], xind, yind, c='yellow', label='1')

    # Get axis limits, create contour mesh. Pair x and y grid coordinates into 
    # a list of coordinate pairs.
    axlims = plt.axis()
    xmesh, ymesh = np.mgrid[axlims[0]:axlims[1]:500j, axlims[2]:axlims[3]:500j]
    coords = np.empty(xmesh.shape + (2,))
    coords[:, :, 0] = xmesh; coords[:, :, 1] = ymesh; 

    # Create 2-d cov. mats., means, and class pdfs
    reducedcovmats = [np.array([[cmat[xind,xind],cmat[xind,yind]],
                                [cmat[yind,xind],cmat[yind,yind]]])
                      for cmat in covmats]
    reducedmeans = np.array([[means[0][xind],means[0][yind]],
                             [means[1][xind],means[1][yind]]])    
    classpdfs = [st.multivariate_normal(mean=reducedmeans[i], 
                                        cov=reducedcovmats[i]) 
                 for i in xrange(2) ]                        

    # Plot Bayes decision boundary, set label. Recall Bayes boundary is
    # {x | P(c0|x) = P(c1|x)}. Using Bayes thm, P(c|x) ~ P(x|c)*P(c). 
    # Plugging this in (denom. cancels on both sides) gives below expression.
    cntr = plt.contour(xmesh, ymesh, 
                       (float(sizes[0])/ntotal)*classpdfs[0].pdf(coords) -
                       (float(sizes[1])/ntotal)*classpdfs[1].pdf(coords), 
                       levels=[0.], colors=['g'], linewidths=4, 
                       linestyles='dashed')

    plt.clabel(cntr, fontsize=12, fmt={0.:''}) # 'fmt' dict ensures no label is
    # written through the line on the plot itself.
    cntr.collections[0].set_label('Bayes Boundary')
    # Output Bayes rate
    bayes_rate(pdfs, samples)

    trainx = np.concatenate( (samples[0][:,(xind,yind)],
                              samples[1][:,(xind,yind)]) )
    trainy = np.concatenate((np.zeros(sizes[0]),np.ones(sizes[1])))

    # Create and train logistic regression classifier
    clf = LogisticRegression()
    clf.fit(trainx,trainy)
    # Output logistic regression score
    print 'Logistic regression class 0 training score (2-D only): %.2f' % \
        clf.score(samples[0][:,(xind,yind)],np.zeros(sizes[0]))
    print 'Logistic regression class 1 training score (2-D only): %.2f' % \
        clf.score(samples[1][:,(xind,yind)],np.ones(sizes[1]))
    print 'Logistic regression total training score (2-D only): %.2f' % \
        clf.score(trainx, trainy)
    # Plot QDA decision boundary, set label. 
    cntr = decision_boundary(clf, axlims, color='purple')
    plt.clabel(cntr, fontsize=12, fmt={0.:''}) # 'fmt' dict ensures no label is
    # written through the line on the plot itself.
    cntr.collections[0].set_label('Logistic Reg.')    

    # Create and train LDA classifier
    clf = LDA()
    clf.fit(trainx,trainy)
    # Output LDA score
    print 'LDA class 0 training score (2-D only): %.2f' % \
        clf.score(samples[0][:,(xind,yind)],np.zeros(sizes[0]))
    print 'LDA class 1 training score (2-D only): %.2f' % \
        clf.score(samples[1][:,(xind,yind)],np.ones(sizes[1]))
    print 'LDA total training score (2-D only): %.2f' % \
        clf.score(trainx, trainy)
    # Plot LDA decision boundary, set label. 
    cntr = decision_boundary(clf, axlims, color='orange')
    plt.clabel(cntr, fontsize=12, fmt={0.:''}) # 'fmt' dict ensures no label is
    # written through the line on the plot itself.
    cntr.collections[0].set_label('LDA Boundary')    

    # Create and train QDA classifier
    clf = QDA()
    clf.fit(trainx,trainy)
    # Output QDA score
    print 'QDA class 0 training score (2-D only): %.2f' % \
        clf.score(samples[0][:,(xind,yind)],np.zeros(sizes[0]))
    print 'QDA class 1 training score (2-D only): %.2f' % \
        clf.score(samples[1][:,(xind,yind)],np.ones(sizes[1]))
    print 'QDA total training score (2-D only): %.2f' % \
        clf.score(trainx, trainy)
    # Plot QDA decision boundary, set label. 
    cntr = decision_boundary(clf, axlims, color='magenta')
    plt.clabel(cntr, fontsize=12, fmt={0.:''}) # 'fmt' dict ensures no label is
    # written through the line on the plot itself.
    cntr.collections[0].set_label('QDA Boundary')    

    plt.legend()
    plt.show()
    

def decision_boundary(clf, axlims, ax=None, threshold=0., color='cyan'):
    '''
    Plot decision boundary of classifier 'clf'.
    '''
    xmin, xmax, ymin, ymax = axlims

    # Make mesh grid for decision contour plot
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, num=500),
                         np.linspace(ymin, ymax, num=500))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    zz = z.reshape(xx.shape)

    if ax is None:
        ax = plt.gca()

    return ax.contour(xx, yy, zz, levels=[threshold], colors=color, 
                      linewidths=4)


def plot_eigenvectors(sample, xind, yind, label=None, ax=None, c='brown'):
    '''
    Plot eigenvectors of data samples.
    '''
    sampmean = sample[:,(xind,yind)].mean(axis=0)
    sampcov = np.cov(sample[:,(xind,yind)].T)
    eigvals, eigvecs = np.linalg.eig(sampcov) # Each column of eigvecs is a
    # right eigenvector.

    if ax is None:
        ax = plt.gca()

    for eigval, eigvec in zip(eigvals, eigvecs.T):
        ax.arrow(sampmean[0], sampmean[1], eigval*eigvec[0], eigval*eigvec[1],
                 head_width=0.13, head_length=0.2, width=0.07, fc=c, ec=c,
                 label=label)


def bayes_rate(pdfs, samples):
    '''
    Find Bayes prediction rate on samples. Can you make this find the more 
    general Bayes rate over whole Real^3 space?
    '''
    predrate = np.empty(len(samples), dtype=float)

    for ind in xrange(len(samples)):
        for point in samples[ind]:
            # Find pdf with maximum prob. for each point. 
            maxprob = max([p.pdf(point) for p in pdfs])
            if pdfs[ind].pdf(point) == maxprob:
                predrate[ind] += 1
        print 'Bayes rate for class %s training data (all dimensions): %.2f' %\
            (ind, predrate[ind]/sizes[ind])
    print  'Total Bayes rate for training data (all dimensions): %.2f' %\
        (sum(predrate)/sum(sizes))


if __name__=='__main__':
    plot_samples(xind=0, yind=1, plotEigenvecs=False)
        
