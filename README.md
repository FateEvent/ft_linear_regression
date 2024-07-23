# ft_linear_regression

__Linear regression__ is one of the first algorithms that machine learning students learn, due to its fundamental nature.
It is a linear model, since it can only fit linear data points.

__Linear regression__ may be _simple_ and _multiple_, but on this occasion we will limit ourselves to analyze __simple linear regression__.

### Simple Linear Regression

__Simple linear regression__ is useful for finding the relationship between two continuous variables, an independent variable and a dependent or predictable one.

Given as input a set of data points, or vectors, containing a series of features, the goal is to obtain a line that best _fits_ the data. The best fit line is the one for which total prediction error is as small as possible. The error represents the vertical distance between each point and its predicted value on the __regression line__ or, in other words, the difference between actual and predicted values.

In the case of simple linear regression, the __regression line__ is represented as:
y = ax + b
where _x_ is the independent feature, _y_ is the dependent feature, _a_ is the slope and _b_ the y-intercept, or constant.

![A model representation](varia/img/model_representation.png)

In this context, "to fit" means to find the parameters of a model (in this case, the regression line) that best represent the relationship between the input features (independent variables) and the output (dependent variable) in the data. Specifically, it involves determining the slope and intercept of the line that minimize the discrepancies between the observed values and the values predicted by the model.

![An example of simple linear regression](varia/img/slr.png)

The whole concept of __linear regression__ is based on the equation of a line:

![The model](varia/img/model.png)

The goal of the algorithm is to learn _theta0_, _theta1_, and _h(x)_.

### The Cost Function

The __cost function__ aims at evaluating the performance of the model by computing a single scalar value that represents the total error across all data points. In __linear regression__, the most commonly used cost functions are the _Mean Squared Error (MSE)_ or the _Sum of Squared Errors (SSE)_.

![alt text](varia/img/mse.png)

![alt text](varia/img/sse.png)

![The cost function](varia/img/cost_function.png)

Here we use the _Mean Squared Error_ function, whose (optimized) formula may be written as follows:

![The cost function formula](varia/img/cost_function_formula.png)

or, to simplify:

![The cost function formula simplified](varia/img/cost_function_simplified_formula.png)

where _m_ represents the total number of examples in the dataset.

A __cost function__ has to be:
- differentiable,
- convex.

A function is __differentiable__ if it has a derivative for each point in its domain as the following examples:

![Examples of differentiable functions](varia/img/differentiable.png)

While functions which have a cusp or a discontinuity are non-differentiable:

![Examples of undifferentiable functions](varia/img/non-differentiable.png)

On the other hand, a univariate function is convex if the line segment connecting two function’s points lays on or above its curve (it does not cross it). If it does it means that it has a local minimum which is not a global one.

![alt text](varia/img/convex.png)

### The Gradient

Intuitively, a __gradient__ is a slope of a curve at a given point in a specified direction.

In the case of a univariate function, it is simply the first derivative at a selected point. In the case of a multivariate function, it is a vector of derivatives in each main direction (along variable axes). Because we are interested only in a slope along one axis and we don’t care about others these derivatives are called partial derivatives.

### Gradient Descent

__Gradient Descent__ is an algorithm used in machine learning to find the lowest point of the convex __cost function__.

![Gradient Descent on cost function](varia/img/gd_example.png)

For ease, let’s take a simple linear model.

    Error = Y(Predicted) - Y(Actual)

A machine learning model always wants low error with maximum accuracy, in order to decrease error we will intuit our algorithm that you’re doing something wrong that is needed to be rectified, that would be done through Gradient Descent.

We need to minimize our error, in order to get pointer to minima we need to walk some steps that are known as alpha(learning rate).
Steps to implement Gradient Descent

    Randomly initialize values.
    Update values.

3. Repeat until slope = 0

A derivative is a term that comes from calculus and is calculated as the slope of the graph at a particular point. The slope is described by drawing a tangent line to the graph at the point. So, if we are able to compute this tangent line, we might be able to compute the desired direction to reach the minima.

Learning rate must be chosen wisely as:
1. if it is too small, then the model will take some time to learn.
2. if it is too large, model will converge as our pointer will shoot and we’ll not be able to get to minima.

![Machine learning formulas I used](varia/img/mlearnia_formulas.png)

### Bibliography

I started my learning process from a playlist [by Machine Lernia](https://www.youtube.com/watch?v=EUD07IiviJg&list=PLO_fdPEVlfKqUF5BPKjGSh7aV9aBshrpY) initiating to machine learning (in French).

The foundations of my work are inspired by Sindhu Seelam's article ["Linear Regression From Scratch in Python WITHOUT Scikit-learn"](https://medium.com/geekculture/linear-regression-from-scratch-in-python-without-scikit-learn-a06efe5dedb6) published on Medium.

Other articles I used for the explanations and the present redaction of the README file are:
- Daksh Trehan's articles ["Linear Regression Explained"](https://pub.towardsai.net/linear-regression-explained-f5cc85ae2c5c) and ["Gradient Descent Explained"](https://towardsdatascience.com/gradient-descent-explained-9b953fc0d2c);
- Robert Kwiatkowski's article ["Gradient Descent Algorithm — a deep dive"](https://medium.com/towards-data-science/gradient-descent-algorithm-a-deep-dive-cf04e8115f21);
- Jatin Mehra's article [Understanding Gradient Descent: A Beginner’s Guide](https://medium.com/@jatinmehra119/understanding-gradient-descent-a-beginners-guide-ad1f948b4b0a).

I used [Desmos Graphing Calculator](https://www.desmos.com/calculator) to graphically display the functions for my examples.

To normalize the values of my arrays of mileage and prices I followed the tip of Cina on [StackOverflow](https://stackoverflow.com/a/41532180).

[Managing arguments in Python with argparse](https://stackoverflow.com/a/11618620)
[Managing boolean arguments](https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse)

### Appendix A

#### A. Fuchs' Notes

En fait le but du jeu c’est de trouver les paramètres a et b d’une fonction linéaire f(x) = ax + b.
Pour ça tu considères une fonction T(a, b) dont la valeur représente une sorte d’erreur moyenne entre la valeur d’estimation de f(x) et les valeurs réelles (plutôt une variance en fait), et tu cherches à minimiser cette erreur.
Le procédé utilisé pour trouver le minimum de T(a, b) s’appelle l’algorithme de descente du gradient. Si tu veux entrer dans les détails et que tu calcules le gradient de T(a, b) tu vas tomber pile sur les formules données dans le sujet: si tu prends T(a, b) = (1 / N) * Somme( ( p_i - a * k_i + b ) ^ 2) où N est le nombre de couple prix / kilomètre (p_i, k_i) que tu récupères dans le fichier .csv et a et b sont les paramètres de T.
Dans l’idée, de façon plus abstraite, T est juste une fonction surface, et tu cherches les coordonnées du point où l’altitude de la surface est la plus basse.
Tu choisis donc un point de départ (dans le sujet c’est a_0 = 0, b_0 = 0), et tu calcules le gradient en ce point, et de mémoire c’est ce que représentent les deux formules du sujet si tu ne prends pas en compte le coefficient de convergence.
Le gradient est donc un vecteur, une direction de plus forte pente, c'est-à-dire que, sur le plan, c’est la direction où la croissance de la fonction est la plus forte. Mais nous, ce qu’on veut, c’est minimiser, donc on prend l’opposé du gradient pour obtenir une descente (au lieu d’une montée). Donc si ton gradient c’est (x_0, y_0) la direction à prendre c’est (-x_0, -y_0).
A partir de là, tu as un point de départ (a_0, b_0) et une direction (-x_0, -y_0) et ben ! il suffit de partir du point de départ dans le sens opposé du gradient et tu vas trouver un nouveau point: (a_0, b_0) + coef * (-x_0, -y_0) = (a_1, b_1).
A partir de là, tu as complété une première itération de l’algorithme. Le point (a_1, b_1) donne une altitude plus basse que (a_0, b_0) c'est-à-dire que T(a_1, b_1) < T(a_0, b_0).
Et pour continuer tu répètes la même chose en remplaçant (a_0, b_0) par (a_1, b_1), et tu continues jusqu’à ce que la différence entre T(a_n, b_n) et T(a_(n+1), b_(n+1)) est assez petite “à ton goût”.

Le coefficient dont je parle dans la dernière formule, c’est le coefficient de convergence. Il y a une manière de trouver le plus optimisé mais c’est un tout autre sujet, que je pourrais t’expliquer mais ce serait mieux avec un tableau en présentiel.
Normalement avec une valeur suffisamment petite l’algo fonctionnera mais il ne sera pas le plus rapide possible.

[...]

Coef c’est bien le learning rate, tandis que T(a, b) dans mon explication c’est la mesure de l’erreur entre la fonction coût et les données réelles.
Et quand tu calcules le gradient de cette fonction T(a, b) tu obtiens les formules données dans le sujet.
Concrètement, dans la première itération tu calcules les deux formules avec a = 0 et b = 0 et tu obtiens a_temp et b_temp.
Tu fais ensuite a1 = learning_rate * a_temp + a0 et b1 = learning_rate * b_temp + b0.
Et tu recommences, tu calcules les deux formules avec cette fois ci a = a1 et b = b1 et tu obtiens de nouveau a_temp et b_temp, puis a2 = learning_rate * a_temp + a1 et b2 = learning_rate * b_temp + b1. Et ainsi de suite.
Après il te faut une condition d’arrêt, parce que l’algorithme va converger vers la solution du minimum, donc soit il l’atteint et ça va tourner en boucle sur la même valeur, soit il ne l’atteint jamais mais s’en approchera infiniment, et donc ça tournera aussi en boucle infinie.

Concernant T(a, b), ce que fait cette fonction c’est comparer le prix pour un kilométrage avec la fonction coût:
f(x) = a * x + b
c’est la fonction que tu veux trouver à l’issue de l'entraînement de l’algorithme. Donc f(kilometre_réel) = prix_estimé.
Et pour comparer tu fais:
prix_réel - f(kilomètre_réel)

Comme on a plusieurs données, on fait alors la moyenne des différences, c’est à dire:
(1 / Nombre_de_data) * Somme(prix_réel - f(kilomètre_réel))

Et comme on ne veut pas de valeures négatives dans cette mesure, on met aussi au carré chaque membre de la somme et ça donne finalement:
(1 / Nombre_de_data) * Somme( (prix_réel - f(kilomètre_réel))^2 )

Si tu développes f dans cette formule tu obtiens:
(1 / Nombre_de_data) * Somme( (prix_réel - (a * kilomètre_réel + b))^2 )

Et cette dernière formule c’est T(a, b). Elle te donne en fait l’erreur moyenne entre la fonction a*x + b et les données réelles.
Trouver le minimum de T(a, b) permet donc de trouver a et b tels que la droite a*x + b passe au plus près de tous les points des données réelles.

Le reste, en particulier l’algorithme de descente du gradient, c’est juste une technique mathématique pour trouver un minimum local sur une fonction “quelconque” (elle doit vérifier quand même quelques propriétés mais ce n'est pas le sujet).
