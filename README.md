# ft_linear_regression





[Why Linear regression for Machine Learning?](https://www.youtube.com/watch?v=qxo8p8PtFeA)

A playlist [by Machine Lernia](https://www.youtube.com/watch?v=EUD07IiviJg&list=PLO_fdPEVlfKqUF5BPKjGSh7aV9aBshrpY) initiating to machine learning (in French).

![formulas](varia/img/mlearnia_formulas.png)

The foundations of my work are inspired by Sindhu Seelam's article ["Linear Regression From Scratch in Python WITHOUT Scikit-learn"](https://medium.com/geekculture/linear-regression-from-scratch-in-python-without-scikit-learn-a06efe5dedb6) published on Medium.

To normalize the values of my arrays of mileage and prices I followed the tip of Cina on [StackOverflow](https://stackoverflow.com/a/41532180).

### Bibliography

Daksh Trehan's articles ["Linear Regression Explained"](https://pub.towardsai.net/linear-regression-explained-f5cc85ae2c5c) and ["Gradient Descent Explained"](https://towardsdatascience.com/gradient-descent-explained-9b953fc0d2c)
Robert Kwiatkowski's article ["Gradient Descent Algorithm — a deep dive"](https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f210)
Jatin Mehra's article [Understanding Gradient Descent: A Beginner’s Guide](https://medium.com/@jatinmehra119/understanding-gradient-descent-a-beginners-guide-ad1f948b4b0a)

[Managing arguments in Python with argparse](https://stackoverflow.com/a/11618620)
[Managing boolean arguments](https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse)

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
