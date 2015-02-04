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

A number of changes (mostly minor) would be necessary to make this code more 
general for N classes. Many statements implicitly assume arrays of length 2. 
Finding the Bayes boundaries between N classes would probably be the least 
trivial change, though it's certainly doable. 
'''

# Make your plots prettier
from mpltools import style
style.use('ggplot')


def create_pdfs(means, covmats):
    '''
    Checks that means and covariance matrices entered by user make sense, then 
    returns Gaussian PDFs.
    '''
    # Ensure means are same dimensionality
    if len(means[0]) != len(means[1]):
        print 'Both classes must have same sized feature space. Check the \
        means array. Exiting!'
        raise SystemExit

    # Ensure covmats are same dimensionality as each other and means.
    if covmats[0].shape !=  covmats[1].shape:
        print 'Covariance matrices must have the same shape. Exiting!'
        raise SystemExit
    if covmats[0].shape[0] != means[0].size or \
       covmats[0].shape[1] != means[0].size:
        print 'Covariance matrices must have shape that is square of the \
        dimensionality of the means. Exiting!'
        raise SystemExit

    # Ensure covmat is pos. def.
    for cmat in covmats:
        if not np.all(np.linalg.eigvals(cmat) > 0):
            print 'Covariance matrix not positive definite. Exiting!'
            raise SystemExit

    # Note if cov. mats equal each other (LDA assumptions not violated)
    if np.array_equal(covmats[0],covmats[1]):
        print 'NOTE: Covariance matrices are equal. LDA assumptions valid.'

    # Return PDFs
    return [st.multivariate_normal(mean=m,cov=c) for m,c in zip(means,covmats)]


def sample_and_fit(clfs, pdfs, sizes, xyind):
    '''
    Pull samples from pdfs and fit classifiers.
    '''
    # Pull sample from each class
    samples = [p.rvs(size=s) for p,s in zip(pdfs,sizes)]

    # Put data into sklearn-friendly arrays.
    trainx = np.concatenate( (samples[0][:,xyind],
                              samples[1][:,xyind]) )
    trainy = np.concatenate((np.zeros(sizes[0]),np.ones(sizes[1])))

    for k in clfs:
        clfs[k].fit(trainx,trainy)
    return clfs, samples


def plot_samples(pdfs, clfs, samples, xyind=(0,1), plotEigenvecs=True,
                 nOutOfSample=0, trainingScores=False):
    '''
    Plot samples along with the Bayes boundary of the pdfs from which the
    samples were pulled. Also plot the boundaries of the classifiers in 'clfs'.
    You can optionally plot the sample eigenvectors as well. "xyind" is a 
    2-tuple containing the indices of the variables to be plotted on the x- and
    y-axis, respectively. 

    Fit diagnostics of the classifiers are evaluated for some number of out of
    sample data sets ('nOutOfSample') specified by the user; the default of 0 
    means no out of sample scores are evaluated. The scores of the classifiers 
    on the training data with which they were fit are optionally reported to 
    stdout.
    '''
    if xyind[0] == xyind[1]:
        print "You need to pass different indices to 'plot_samples'; exiting."
        raise SystemExit

    # Numbers of events in both classes
    sizes = [len(samp) for samp in samples]
    ntotal = sum(sizes)

    # Perform out sample evaluations of classifiers
    if nOutOfSample:
        out_of_sample_scores(pdfs, clfs, sizes, xyind, nOutOfSample)

    # Plot sample
    plt.scatter(samples[1][:,xyind[0]], samples[1][:,xyind[1]], c='b')
    plt.scatter(samples[0][:,xyind[0]], samples[0][:,xyind[1]], c='r')

    # Plot eigenvectors, if desired
    if plotEigenvecs:
        for samp in samples:
            plot_eigenvectors(samp, xyind, c='yellow', label='0')

    # Plot Bayes boundary
    axlims = plt.axis()
    plot_bayes_boundary(pdfs, axlims, xyind, sizes)

    # Fit and plot decision boundary for each classifier
    for k,col in zip(clfs,['purple','orange','magenta']):
        # Plot decision boundary, set label. 
        cntr = plot_decision_boundary(clfs[k], axlims, color=col)
        plt.clabel(cntr, fontsize=12, fmt={0.:''}) # 'fmt' dict ensures no 
        # label is written through the line on the plot itself.
        cntr.collections[0].set_label('%s Boundary' % k)    

    # Output Bayes rate and training scores, if desired
    if trainingScores:
        print 30*'-', ' Training scores ', 30*'-'
        # Bayes rate
        predrate = bayes_rate(pdfs, samples)
        for ind in xrange(len(samples)):
            print 'Bayes rate for class %s training data: %.2f' % \
                (ind, predrate[ind]/sizes[ind])
        print  'Total Bayes rate for training data: %.2f' %\
            (sum(predrate)/sum(sizes))
        
        # Classifier training rates
        for name in ['LogReg','LDA','QDA']:
            training_score(clfs[name], name, [samples[0][:,xyind],
                                              samples[1][:,xyind]])
            
    plt.legend()
    plt.show()


def plot_decision_boundary(clf, axlims, ax=None, threshold=0., color='cyan'):
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


def plot_bayes_boundary(pdfs, axlims, xyind, sizes, ax=None):
    '''
    Plot Bayes decision boundary.
    '''
    # Unpack xyind
    xind, yind = xyind
    # Create contour mesh. Pair x and y grid coordinates into a list of
    # coordinate pairs.
    xmesh, ymesh = np.mgrid[axlims[0]:axlims[1]:500j, axlims[2]:axlims[3]:500j]
    coords = np.empty(xmesh.shape + (2,))
    coords[:, :, 0] = xmesh; coords[:, :, 1] = ymesh; 

    # Get means and covariance matrices
    means, covmats = [p.mean for p in pdfs], [p.cov for p in pdfs]

    # Create 2-d cov. mats., means, and class pdfs
    reducedcovmats = [np.array([[cmat[xind,xind],cmat[xind,yind]],
                                [cmat[yind,xind],cmat[yind,yind]]])
                      for cmat in covmats]
    reducedmeans = np.array([[m[xind],m[yind]] for m in means])
    classpdfs = [st.multivariate_normal(mean=reducedmeans[i], 
                                        cov=reducedcovmats[i]) 
                 for i in xrange(2) ]                        

    if ax is None:
        ax = plt.gca()

    # Plot Bayes decision boundary, set label. Recall Bayes boundary is
    # {x | P(c0|x) = P(c1|x)}. Using Bayes thm, P(c|x) ~ P(x|c)*P(c). 
    # Plugging this in (denom. cancels on both sides) gives below expression.
    ntotal = sum(sizes)
    cntr = ax.contour(xmesh, ymesh, 
                       (float(sizes[0])/ntotal)*classpdfs[0].pdf(coords) -
                       (float(sizes[1])/ntotal)*classpdfs[1].pdf(coords), 
                       levels=[0.], colors=['g'], linewidths=4, 
                       linestyles='dashed')

    plt.clabel(cntr, fontsize=12, fmt={0.:''}) # 'fmt' dict ensures no label is
    # written through the line on the plot itself.
    cntr.collections[0].set_label('Bayes Boundary')


def plot_eigenvectors(sample, xyind, label=None, ax=None, c='brown'):
    '''
    Plot eigenvectors of data samples.
    '''
    sampmean = sample[:,xyind].mean(axis=0)
    sampcov = np.cov(sample[:,xyind].T)
    eigvals, eigvecs = np.linalg.eig(sampcov) # Each column of eigvecs is a
    # right eigenvector.

    if ax is None:
        ax = plt.gca()

    for eigval, eigvec in zip(eigvals, eigvecs.T):
        ax.arrow(sampmean[0], sampmean[1], eigval*eigvec[0], eigval*eigvec[1],
                 head_width=0.13, head_length=0.2, width=0.07, fc=c, ec=c,
                 label=label)


def training_score(clf, clfname, samples, sizes=0):
    '''
    Output training score for classifier 'clf' with name 'clfname', which was
    trained on 'sample'.
    '''
    if not sizes:
        sizes = [len(samp) for samp in samples]

    print '%s class 0 training score (2-D only): %.2f' % \
        (clfname, clf.score(samples[0][:],np.zeros(sizes[0])))
    print '%s class 1 training score (2-D only): %.2f' % \
        (clfname, clf.score(samples[1][:],np.ones(sizes[1])))

    trainx = np.concatenate( (samples[0][:], samples[1][:]) )
    trainy = np.concatenate((np.zeros(sizes[0]),np.ones(sizes[1])))

    print '%s total training score (2-D only): %.2f' % \
        (clfname, clf.score(trainx, trainy))


def bayes_rate(pdfs, samples):
    '''
    Find Bayes prediction rate on samples. Can you make this find the more 
    general Bayes rate over whole Real^3 space?
    '''
    # Calculate class priors
    priors = np.array([len(samp) for samp in samples], dtype=float)
    priors /= sum(sizes)
    predrate = np.zeros(len(samples), dtype=float)

    for ind in xrange(len(samples)):
        for x in samples[ind]:
            # Find pdf with maximum prob. for each point. 
            if np.argmax([p.pdf(x)*pri for p,pri in zip(pdfs,priors)]) == ind:
                predrate[ind] += 1
            #maxprob = max([p.pdf(point) for p in pdfs])
            #if pdfs[ind].pdf(point) == maxprob:
            #    predrate[ind] += 1
    return predrate


def out_of_sample_scores(pdfs, clfs, sizes, xyind, nOutOfSample):
    '''
    Check score of each classifier in 'clfs' on some number ('nOutOfSample') of
    new data sets pulled from 'pdfs'. 
    '''
    ntotal = sum(sizes)
    # Build empty score dict
    scores = {'Bayes':0}
    for name in clfs:
        scores[name] = 0

    for _ in xrange(nOutOfSample):
        # Pull sample from each class
        samples = [p.rvs(size=s) for p,s in zip(pdfs,sizes)]

        # Put data into sklearn-friendly arrays.
        testx = np.concatenate( (samples[0][:,xyind],
                                  samples[1][:,xyind]) )
        testy = np.concatenate((np.zeros(sizes[0]),np.ones(sizes[1])))
        
        # Get scores for classifiers and Bayes boundary
        for name in scores:
            if name == 'Bayes':
                scores[name] += (sum(bayes_rate(pdfs, samples))/ntotal)
            else:
                scores[name] += clfs[name].score(testx, testy)    

    print 'Over %s out-of-sample data sets, the mean classification rate was:'\
        % nOutOfSample
    for name in scores:
        print 'mean %s rate: %.5f %%' % (name,scores[name]/float(nOutOfSample))


if __name__=='__main__':

    # Set random seed
    np.random.seed(3)

    # Number of events pulled from classes
    sizes = np.array([500,1000])

    # Classes well-separated along x and y axes, not so much along z.
    means = np.array([[1,1],[-1,-1]])
    #means = np.array([[1,1,1,1],[-1,-1,-1,-1]])

    # Covariance matrices
    covmat0 = np.array([[3, 1.5],
                        [1.5, 2]])
    covmat1 = np.array([[2, -0.7],
                        [-0.7, 2]])

    #covmat0 = np.array([[1.7, 0.5, -0.1, 0.2],
    #                    [0.5, 1.1, 0.1, -0.1],
    #                    [-0.1, 0.1, 1.4, 0.6],
    #                    [0.2, -0.1, 0.6, 1.3]])
    
    #covmat1 = np.array([[1.2, -0.5, 0.1, 0.2],
    #                    [-0.5, 1.1, -0.1, 0.1],
    #                    [0.1, -0.1, 0.9, 0.2],
    #                    [0.2, -0.1, 0.2, 1.]])

    covmats = [covmat0, covmat0]

    pdfs = create_pdfs(means, covmats)

    # Create dict of classifiers
    clfs = {}
    clfs['LogReg'] = LogisticRegression()
    clfs['LDA'] = LDA()
    clfs['QDA'] = QDA()

    # Which dimensions of your n-dimensional gaussians would you like to plot?
    xyind = (0, 1)

    # Pull sample and fit classifiers
    clfs, samples = sample_and_fit(clfs, pdfs, sizes, xyind)
    
    # Plot samples, evaluate classifiers on training data
    plot_samples(pdfs, clfs, samples, xyind, plotEigenvecs=False,
                 trainingScores=True, nOutOfSample=100)
        
