import numpy as np
import scipy.stats as st
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

'''
This code will draw events from two classes, each a function of three 
covariates. These events will be plotted on the two axes chosen by the user.
The Bayes decision boundary will also be drawn on this plot.
'''

# Make your plots prettier
from mpltools import style
style.use('ggplot')

# Set random seed
np.random.seed(4)

# Number of events pulled from classes
sizes = np.array([60,300])
ntotal = sum(sizes)
# Classes well-separated along x and y axes, not so much along z.
means = np.array([[-1,1,0],[1,0,0.1]])

# Covariance matrices
covmat0 = np.array([[1.7, 0.5, -0.3],
                    [0.5, 1.1, 0.1],
                    [-0.3, 0.1, 1.5]])
covmat1 = np.array([[1, 0.1, -0.5],
                    [0.1, 1, 0.3],
                    [-0.5, 0.3, 1]])
covmats = [covmat0, covmat1]

# Ensure covmat is pos. def.
for cmat in covmats:
    if not np.all(np.linalg.eigvals(cmat) > 0):
        print 'Covariance matrix not positive definite. Exiting!'
        raise SystemExit

# Pull sample from each class
samples = []
samples.append(st.multivariate_normal.rvs(mean=means[0], cov=covmats[0], 
                                          size=sizes[0]))
samples.append(st.multivariate_normal.rvs(mean=means[1], cov=covmats[1], 
                                          size=sizes[1]))

def plot_samples(samples=samples, xind=0, yind=1):
    '''
    "xind" and "yind" are the indices of the variables plotted on the x- and 
    y-axis, respectively.
    '''
    if xind == yind:
        print "You need to pass different indices to 'plot_samples'; exiting."
        raise SystemExit

    # Plot sample
    plt.scatter(samples[0][:,xind], samples[0][:,yind], c='r')
    plt.scatter(samples[1][:,xind], samples[1][:,yind], c='b')

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

    # Create and train LDA classifier
    clf = LDA()
    trainx = np.concatenate( (samples[0][:,(xind,yind)],
                              samples[1][:,(xind,yind)]) )
    trainy = np.concatenate((np.zeros(sizes[0]),np.ones(sizes[1])))
    clf.fit(trainx,trainy)
    # Plot LDA decision boundary, set label. 
    cntr = decision_boundary(clf, axlims, color='cyan')
    plt.clabel(cntr, fontsize=12, fmt={0.:''}) # 'fmt' dict ensures no label is
    # written through the line on the plot itself.
    cntr.collections[0].set_label('LDA Boundary')    

    # Create and train QDA classifier
    clf = QDA()
    clf.fit(trainx,trainy)
    # Plot QDA decision boundary, set label. 
    cntr = decision_boundary(clf, axlims, color='magenta')
    plt.clabel(cntr, fontsize=12, fmt={0.:''}) # 'fmt' dict ensures no label is
    # written through the line on the plot itself.
    cntr.collections[0].set_label('QDA Boundary')    

    # Create and train logistic regression classifier
    clf = LogisticRegression()
    clf.fit(trainx,trainy)
    # Plot QDA decision boundary, set label. 
    cntr = decision_boundary(clf, axlims, color='purple')
    plt.clabel(cntr, fontsize=12, fmt={0.:''}) # 'fmt' dict ensures no label is
    # written through the line on the plot itself.
    cntr.collections[0].set_label('Logistic Reg.')    

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

if __name__=='__main__':
    plot_samples(xind=0,yind=1)
        
