# Gaussian Classification
The code in "two_gaussian_classes.py" draws points from two classes, each specified by an n-dimensional Gaussian, where the user can choose 'n'. The number of points pulled from each class is set by the 'sizes' array. The means and covariance matrices of the Gaussians are specified by the 'means' and 'covmats' variables towards the bottom of the file.

Calling "two_gaussian_classes.py" from the command line as follows:
```
python two_gaussian_classes.py 
```
will plot the randomly drawn points on the two axes specified by 'xind' and 'yind' (where 'xind' and 'yind' are distinct integers in the set {0,...,n-1}), the Bayes optimally decision boundary, a boundary determined by linear discriminant analysis (LDA), a boundary determined by quadratic discriminant analysis (QDA), a boundary determined by logistic regression, and the eigenvectors of the two classes in the 'xind-yind' plane.

The user can test the fitted classifiers against several out of sample data sets, if he or she so chooses. The number of data sets is specified by the 'nOutOfSample' argument to the 'plot_samples()' method. Mean classification rates are evaluated and output to std out.

Several classification rates on training data are optionally output in the terminal window (the user must 'trainingScores=True' to 'plot_samples()'). The empirical Bayes rate for each class and for all points is calculated using all three dimensions of the sample data. The classification rates for the LDA, QDA, and logistic regression estimators are also output to std out.

By changing what is passed to the 'plot_samples(...)' method, the user can specify which axes get plotted and whether or not eigenvectors get plotted. 
