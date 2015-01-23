# Gaussian Classification
The code in "two_class_3preds.py" draws points from two classes, each specified by a 3-dimensional Gaussian. The number of points pulled from each class is set by the 'sizes' array. The means and covariance matrices of the Gaussians are specified by the 'means' and 'covmats' variables towards the top of the file.

Calling "two_class_3preds.py" from the command line as follows:
```
python two_class_3preds.py 
```
will plot the randomly drawn points on the two axes specified by 'xind' and 'yind,' the Bayes decisions boundary, a boundary determined by linear discriminant analysis (LDA), a boundary determined by quadratic discriminant analysis (QDA), a boundary determined by logistic regression, and the eigenvectors of the two classes in the 'xind-yind' plane.

Several classification rates on training data are output in the terminal window. The empirical Bayes rate for each class and for all points is calculated using all three dimensions of the sample data. The classification rates for the LDA, QDA, and logistic regression estimators are also ouput in the terminal.

By changing what is passed to the 'plot_samples(...)' method, the user can specify which axes get plotted and whether or not eigenvectors get plotted. 
