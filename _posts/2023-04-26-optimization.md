---
layout: post
title:  "Optimization"
date:   2023-04-26 10:14:54 +0700
categories: Mathematics
---

# TOC
- [Introduction](#intro)
- [Equivalence](#equi)
- [Convex optimization](#convex)
- [Linear program](#linear)
- [Quadratic program](#quadratic)
- [Geometric program](#geometric)


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

Since the domain of the problem is convex, the feasible set of a convex optimization problem is also convex. 

For the convex optimization problem, any local optimal point is also global. We prove this by contradictory. If x is feasible and locally optimal, then $$ f_0(x) = inf\{f_0(z) \mid z \text{feasible}, \mid\mid z - x \mid\mid_2 \leq R \}, R \geq 0 $$. Suppose that x is not globally optimal, there would exists a feasible y such that $$ f_0(y) < f_0(x) $$. This would make $$ \mid \mid y - x \mid\mid_2 > R $$ since otherwise $$ f_0(x) \leq f_0(y) $$. Consider a point $$ z = (1 - \theta) x + \theta y, \theta = \frac{R}{2\mid\mid y - x \mid\mid_2} $$. Then $$ \mid\mid z - x \mid\mid_2 = R/2 < R $$. By convexity of the feasible set, z is feasible. By convexity of $$ f_0 $$: $$ f_0(z) \leq (1 - \theta) f_0 (x) + \theta f_0(y) < f_0(x) $$. This is contradictory, so there is no other feasible y with $$ f_0(y) < f_0(x) $$ so x is indeed global.

# Linear optimization

When the objective and constraint functions are all affine, the problem is called a linear program. A general linear program has the form:

minimize $$ c^T x + d $$ subject to $$ Gx \leq h, Ax = b $$ where $$ G \in R^{mxn}, A \in R^{pxn} $$. Notice that the constant d is not necessary. 

<img width="366" alt="Screen Shot 2023-04-26 at 16 19 00" src="https://user-images.githubusercontent.com/7457301/234531058-dba8f50d-2ced-4044-8e62-c84b6a01ad82.png">

Image: geometric interpretation of an LP. The point $$ x^* $$ is optimal, it is the point farthest possible in the direction of -c.

There are two cases of LP that are popular:

- A standard form LP: minimize $$ c^T x $$ subject to $$ Ax = b, x \geq 0 $$. 

- An inequality form LP: minimize $$ c^T x $$ subject to $$ Ax \geq b $$.

Here is how to convert LPs to standard form:

- The first step is to introduce slack variables $$ s_i $$ for the inequalities: minimize $$ c^T x + d $$, subject to $$ Gx + s = h, Ax = b, s \geq 0 $$. 

- The second step is to express the variable x as the difference of two nonnegative variables y and z ($$ z = y - z, y, z \geq 0 $$): minimize $$ c^T y - c^T z + d $$ subject to $$ Gy - Gz + s = h, Ay - Az = b, y \geq 0, z \geq 0, s \geq 0 $$.

# Quadratic optimization

The convex optimization problem is called a quadratic program if the objective function is (convex) quadratic and the constraint functions are affine:

minimize $$ (1/2)x^T P x + q^T x + r $$ subject to $$ Gx \leq h, Ax = b $$ where $$ P \in S^n_+, G \in R^{mxn}, A \in R^{pxn} $$. The convex quadratic function is minimized over a polyhedron. 

<img width="363" alt="Screen Shot 2023-04-26 at 16 43 31" src="https://user-images.githubusercontent.com/7457301/234537404-e71a4c15-ec04-4579-8c87-ad6bb2b96ada.png">

Image: Geometric illustration of QP. The point $$ x^* $$ is optimal.

# Geometric programming

Geometric programming is a family of optimization problems that naturally are not convex but can be transformed into a convex problem, by changeing of variables and transformation of objective and constraint functions.

A function $$ f: R^n \to R $$ with domain $$ R^n_{++}: f(x) = cx_1^{a_1} x_2^{a_2} ...x_n^{a_n} $$ where c > 0 and $$ a_i \in R $$ is called a monomial function. A sum of monomials is called a posynomial function of K terms: $$ f(x) = \sum_{k=1}^K c_k x_1^{a_{1k}} x_2^{a_{2k}} ... x_n^{a_{nk}} $$ where $$ c_k > 0 $$.

Then the optimization problem of the form: minimize $$ f_0(x) $$ subject to $$ f_i(x) \leq 1, i=1...m, h_i(x) = 1, i=1...p $$ where $$ f_0, ...f_m $$ are posynomials and $$ h_1,...h_p $$ are monomials, is called the geometric program (GP). The domain of the problem is $$ D = R^n_{++}, x > 0 $$.

- To transform the GP into a convex optimization problem, we first change the variables:

Let's $$ y_i = log x_i $$ so $$ x_i = e^{y_i} $$. If $$ f(x) = c x_1^{a_1} x_2^{a_2}... x_n^{a_n} $$ then $$ f(x) = f(e^{y_1}, ..., e^{y_n}) = c (e^{y_1})^{a_1} ... (e^{y_n})^a_n = e^{a^T y + b} $$ where b = log c. This turns the monomial function into the exponential of an affine function. Now we do similarly for the posynomial: $$ f(x) = \sum_{k=1}^K e^{a_k^{T} y + b_k} $$ where $$ a_k = (a_{1k}, ...a_{nk}) $$ and $$ b_k = log c_k $$. The posynomial also becomes a sum of exponentials of affine functions. The geometric program then can be: minimize $$ \sum_{k=1}^{K_0} e^{a_{0k}^T y + b_{0k}} $$ subject to $$ \sum_{k=1}^{K_i} e^{a_{ik}^T y + b_{ik}} \leq 1, i=1...m, e^{g_i^T y +h_i} = 1, i=1...p $$ where $$ a_{ik} \in R^n, i=0...m $$ and $$ g_i \in R^n, i=1...p $$.

- The second step is to transform the objective and constraint functions by taking logarithm:

minimize $$ f_0(y) = log(\sum_{k=1}^{K_0} e^{a_{0k}^T y + b_{0k}}) $$ subject to $$ f_i(y) = log(\sum_{k=1}^{K_i} e^{a^T_{ik} y + b_{ik}}) \leq 0, i=1...m, l_i(y) = g_i^T y + h_i = 0, i=1..p $$
