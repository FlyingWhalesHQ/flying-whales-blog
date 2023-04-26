---
layout: post
title:  "Optimization"
date:   2023-04-26 10:14:54 +0700
categories: Mathematics
---

# Introduction

Convex optimization is a subfield of mathematical optimization that deals with the minimization of convex functions over convex sets. Convex optimization problems have a unique global minimum, which is a highly desirable property for optimization algorithms. The field has many applications in areas such as machine learning, signal processing, statistics, finance, and operations research.

Usually the problem is to find an x that minimize a function $$ f_0(x) $$ with the constraints of $$ f_i(x) \leq 0, i = 1,...m $$ and $$ h_i(x) = 0, i = 1,..p $$. Mathematically, the optimization problem is written as follows:

minimize $$ f_0(x) $$
subject to $$ f_i(x) \leq 0, i = 1,..m $$
and $$ h_i(x) = 0, i = 1,...p $$

We call $$ x \in R^n $$ the optimization variable and the function $$ f_0: R^n \to R $$ the objective function. The inequalities $$ f_i(x) \leq 0 $$ are the inequality constraints and the corresponding functions $$ f_i: R^n \to R $$ are the inequality constraint functions. Without the constraint (m = p = 0) the problem is unconstraint.

The set of points for which all the functions are defined is called the domain. A point $$ x \in D $$ is feasible if it satisfies the constraints. If there is at least one feasible point, the problem is feasible, and infeasible otherwise. The set of all feasible points is the feasible set or constraint set. The optimal value $$ p^* $$ is defined as:

$$ p^* = inf\{f_0(x) \mid f_i(x) \leq 0, i=1...m,h_i(x) = 0, i=1...p $$

If $$ p^* = - \infty $$, the problem is unbounded below. We say $$ x^* $$ is an optimal point if $$ x^* $$ is feasible and $$ f_0(x^*) = p^* $$. The set of all optimal points is denoted:

$$ X_opt = \{ x \mid f_0(x) = p^*, f_i(x) \leq 0, i = 1,...m, h_i(x) = 0, i=1..p \} $$

If the optimal point exists, the problem is solvable. The unbounded below problem is not solvable. A feasible point x with $$ f_0(x) \leq p^* + \epsilon, \epsilon > 0 $$ is called $$ \epsilon-$$suboptimal and the set is the $$ \epsilon-$$suboptimal set. 

A local optimal point is a feasible point that solves the minimum problem $$ f_0 $$ over nearby points in the feasible set:

$$ f_0(x) = inf \{f_0(z) \mid f_i(z) \leq 0, i=1..m, h_i(z) = 0, i=1..p, \mid \mid z - x \mid\mid_2 \leq R \}, R > 0 $$

In other words, x minimize $$ f_0(z) $$ subject to $$ f_i(z) \leq 0, i=1..m, h_i(z) = 0, i=1...p, \mid\mid z - x \mid\mid_2 \leq R $$.

The standard form of an optimization is when you arrange the functions and inequalities so that the right hand size is 0. For example, for an equality constraint $$ f_i(x) = g_i(x) $$, we can turn it into $$ h_i(x) = 0 $$ where $$ h_i(x) = f_i(x) - g_i(x) $$. Similarly, $$ f_i(x) \geq 0 $$ can be written as $$ -f_i(x) \leq 0 $$. The box constraints $$ m_i \leq x_i \leq n_i $$ can be expressed as $$ m_i - x_i \leq 0, x_i - n_i \leq 0 $$. The maximization problem $$ f_0(x) $$ can be written as minimizing $$ -f_0 $$.

# Equivalence

The two problems are equivalent if from the solution of one, the solution of the other is easily found and vice versa. One way to make an equivalent problem is to scale the functions and constraints:

minimize $$ f(x) = \alpha_0 f_0(x) $$
subject to $$ f_i(x) = \alpha_i f_i(x) \leq 0, i = 1..m $$,
$$ h_i(x)=\beta_i h_i(x) = 0, i = 1...p $$
where $$ \alpha_i > 0, i=0..m, \beta_i \neq 0, i=1...p$$

As a result of the scaling, the feasible sets of the problem and the original one is the same. There are some other ways of transformation that make equivalent problems.

- Change of variables: when we substitute variable $$ x = \phi(z) $$ the problem becomes an equivalent one:

minimize $$ f_0(z) $$
subject to $$ f_i(z) \leq 0, i=1..m $$,
$$ h_i(z) = 0, i=1..p $$

If x solves the problem, then $$ z = \phi^{-1} (x) $$ solves the problem and vice versa.

- Transformation of objective and constraint functions: let $$ f_i(x) = \phi_i(f_i(x)), i=0..m, h_i(x) = \phi_i(h_i(x)), i=1..p$$. We come up with an equivalent problem:

minimize $$ f_0(x) $$ subject to $$ f_i(x) \leq 0, i=1..m, h_i(x) = 0, i=1...p $$.

For example the unconstrained Euclidean norm minimization problem $$ \mid\mid Ax - b \mid\mid_2 $$ is equivalent to $$ (Ax - b)^T(Ax - b) $$. This makes it easier to solve since the latter objective is differential for all x (since it is quadratic) and the latter is not differential at $$ Ax - b = 0 $$. 

- Adding slack variables so that the inequalities become equalities. We have: minimize $$ f_0(x) $$ subject to $$ s_i \geq 0, i=1,...m, f_i(x) + s_i = 0, i =1...m, h_i(x) = 0, i=1...p $$. 

- Transform the equality constraints into $$ x = \phi(z) $$ so that the problem becomes: minimize $$ f_0(x) = f_0(\phi(z)) $$ subject to $$ f_i(z) = f_i(\phi(z)) \leq 0, i = 1..m $$

- Introduce new equality constraint $$ y_i = A_i x + b_i, i = 0...m $$: minimize $$ f_0(y_0) $$ subject to $$ f_i(y_i) \leq 0, i = 1...m, y_i = A_i x + b_i, i = 0...m, h_i(x) = 0, i=1...p $$

- Optimizing sequentially over variables. For example, we can optimize over y first, then x: $$ inf_{x,y}f(x,y) = inf_x g(x) $$ with $$ g(x) = inf_y f(x,y) $$.

- By changing into epigraph form: minimize t subject to $$ f_0(x) - t \leq 0, f_i(x) \leq 0, i=1...m, h_i(x) = 0, i=1...p $$

- By restricting the domain of the new unconstraint problem, For example we have a new unconstraint objective: to minimize F(x) but with the feasible set restraint by previous inequalities.

# Convex optimization

From the standard optimization problem, the convex optimization problem requires extra conditions:

- The objective function must be convex

- The inequality constraint functions must be convex

- The equality constrain functions $$ h_i(x) = a_i^T x - b_i $$ must be affine




```python

```
