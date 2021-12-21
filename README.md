# Support-Vector-Machines from Scratch

**This repository is for educational purposes only!**

It should demonstrate how Support-Vector-Machines can be implemented from scratch.
Here, SMVs are regarded as Lagrange optimization problems (convex problem with constraint).

### General usage:
````python
import svm

model = svm.RbfSupportVectorClassifier(c=1)
model.train(X, y)
````

For an example of usage, see [this Jupyter-Notebook](inspecting_svm.ipynb)

The mathematical problem is solved with [CVXPY](https://www.cvxpy.org/). 
If you have problems with the installation, have a look at [this link](http://man.hubwiz.com/docset/cvxpy.docset/Contents/Resources/Documents/install/index.html)