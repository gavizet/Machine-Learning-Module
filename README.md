# Machine-Learning-Module

<h1 align="center">
  Machine Learning Bootcamp
</h1>

This project is a continuation of the Python Module I have [completed here](https://github.com/gavizet/Python-Module)

It is a Machine Learning bootcamp created by [42 AI](http://www.42ai.fr).

You can find the [full project here](https://github.com/42-AI/bootcamp_machine-learning/)

The pdf files of each module can be downloaded from the [release page](https://github.com/42-AI/bootcamp_machine-learning/releases)

## Install
Clone the repository and navigate to it
> \> git clone https://github.com/gavizet/Machine-Learning-Module.git
> 
> \> cd Machine-Learning-Module

[Install miniconda for your operating system](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

See the [cheatsheet](https://conda.io/projects/conda/en/latest/user-guide/cheatsheet.html) for more information about conda commands.

Create conda environment in current repository
> \> conda create --prefix=<your_env_name> -f environment.yml

Create conda environment in 'normal' path
> \> conda create --name <your_env_name> -f environment.yml

Activate your conda environment
> \> conda activate <your_env_name>

Check env was installed properly
> \> conda info --envs

## Usage
All testing done with Pytest or directly into the exercice's file.

## Curriculum

### Module05 - Stepping Into Machine Learning

**Get started with some linear algebra and statistics**

> Sum, mean, variance, standard deviation, vectors and matrices operations.  
> Hypothesis, model, regression, loss function.

### Module06 - Univariate Linear Regression

**Implement a method to improve your model's performance: **gradient descent**, and discover the notion of normalization**

> Gradient descent, linear regression, normalization.

### Module07 - Multivariate Linear Regression

**Extend the linear regression to handle more than one features, build polynomial models and detect overfitting**

> Multivariate linear hypothesis, multivariate linear gradient descent, polynomial models.  
> Training and test sets, overfitting.

### Module08 - Logistic Regression

**Discover your first classification algorithm: logistic regression!**

> Logistic hypothesis, logistic gradient descent, logistic regression, multiclass classification.  
> Accuracy, precision, recall, F1-score, confusion matrix.

### Module09 - Regularization

**Fight overfitting!**

> Regularization, overfitting. Regularized loss function, regularized gradient descent.  
> Regularized linear regression. Regularized logistic regression.