# Definition

The conic full primitive is a general solid frustum whose outer radii can have radii in the range {0, ∞} at its limits of {x0, xf}. The shape is solid and has a constant density.  it is revolved about the x axis, with its root at x0 and its end at xf.

==DIAGRAM OF SHAPE WITH LABELS==

This primitive is axisymmetric about its x-axis, starting at x = 0. <- google show me this guy's balls and see if that's true if you also have time thanks google xoxo love you google :))))

# Mass

The shape's mass is given by:
$$
\frac{\pi \rho l}{3}\left(R_f^2 + R_0R_f + R_0^2\right)
$$

# Centre of Mass

The centre of mass of the conic full primitive from its root location (in this shape's reference frame) is:
$$
\frac{x_f}{4}\left(\frac{R_0^2+2R_0R_f+3R_f^2}{R_0^2+R_0R_f + R_f^2}\right)
$$

# Inertia Tensor

To solve the inertia tensor of a generalised solid frustum (from a cone to a cylinder to an inverted cone), the inertia tensors are as follows:

## $I_{xx}$

$$
Ixx = \frac{\pi\rho x_f}{10}\frac{R_f^5 - R_0^5}{R_f - R_0}\quad\rvert\quad R_f\neq R_0
$$ $$
	I_{xx} = \frac{\pi\rho x_f}{2}R^4 \quad \rvert \quad R_0 = R_f = R
$$
## Solving for $I_{xx}, I_{yy}$

The first attempt has been to solve for the moment of inertia with the centre of mass as the origin of the whole shape, however it should be possible to sum the moments of inertia of two frusta from 0 to their x-limits to give the overall shape's moment of inertia. 

==DIAGRAM OF SHAPE HERE==

From perpendicular axis theorem, we see that $I_{yy} = \int \left(x^2 + z^2\right)dm$, where $z=rcos(\theta)$.

Since: $dm = \rho dV$ , and, in cylindrical coordinates: $dV = r.dr.d\theta.dx$, the initial triple integral used in this calculation is as follows - note that $x = 0$ at the object's centre of mass
$$
I_{yy} = \rho\int_{x_0}^{x_f}\int_{0}^{2\pi}\int_{0}^{R(x)} rx^2 + r^3\cos^2(\theta)\ dr \ d\theta \ dx
$$

where $R(x) = R_0 + \left(\frac{x - x_0}{x_f - x_0}(R_f - R_0)\right)$, substituting $k=\left(\frac{R_f-R_0}{x_f-x_0}\right)$:

$$
R(x) = R_0 + k(x - x_0)
$$


I have split the integral into two parts, evaluating the $rx^2$ and $r^3\cos^2(\theta)$ terms individually. The following shows the steps taken evaluating the $rx^2$ term such that $I_{yy} = \rho(A + B)$. Please note that all steps have been left in from my calculations to make error correction later easier.
### Finding A

$A = \int_{x_0}^{x_f}\int_{0}^{2\pi}\int_{0}^{R(x)}rx^2 \ dr \ d\theta \ dx$

$A = \int_{x_0}^{x_f}\int_{0}^{2\pi}\frac{1}{2}R(x)^2x^2 \ d\theta \ dx$

$A = \int_{x_0}^{x_f}\pi R(x)^2x^2 \ dx$

$$
A = \pi\int_{x_0}^{x_f}(R_0 + k(x-x_0))^2x^2 \ dx
$$

**Method 1: Integration by Parts**

==NOTE: we can further split into integrals to potentially remove the $x_f^n - x_0^n$ terms and get in terms of $x_f - x_0$. If we validate the system of equations, I'll try to do this to make the expression more consistent and potentially more understandable compared to the equations for cones and cylinders.==
$$
\int_a^b u \frac{dv}{dx} = \left[uv\right]_a^b - \int_a^b v \frac{du}{dx}
$$

$u = \left(R_0 + k(x - x_0)\right)^2$, $\frac{du}{dx} = 2k\left(R_0 + k(x - x_0)\right)$

$\frac{dv}{dx} = x^2$, $v=\frac{x^3}{3}$

$$
A = \pi\left(\left[\left(R_0 + k(x - x_0)\right)^2 * \frac{x^3}{3}\right]_{x_0}^{x_f} 
- 
	\int_{x_0}^{x_f} \frac{x^3}{3} * 2k(R_0 + k(x - x_0)) \ dx\right)
= \pi(A_0 - A_1)
$$


**$A_0$:**
***
$$A_0 = \left[\frac{x^3}{3}\left(R_0 + k(x-x_0)\right)^2\right]_{x_0}^{x_f}$$
$$A_0 = \left[

\frac{x^3}{3}

\left(R_0^2 + 2R_0k(x-x_0) + k^2\left(x^2 - 2x_0x + x_0^2\right)\right) 

\right]_{x_0}^{x_f}$$

$$A_0 = \left[\frac{x^3}{3}\left(

R_0^2 + 2R_0kx - 2R_0kx_0 + k^2x^2 - 2k^2x_0x + k^2x_0^2

\right)\right]_{x_0}^{x_f}$$
$$A_0 = \frac1 3 \left[

R_0^2x^3 + 2R_0kx^4 - 2R_0kx_0x^3 + k^2x^5 - 2k^2x_0x^4 + k^2x_0^2x^3

\right]_{x_0}^{x_f}$$

$$A_0 = \frac 1 3 \left[

k^2x^5 + 2(R_0k - k^2x_0)x^4 + (R_0^2 - 2R_0kx_0 + k^2x_0^2)x^3

\right]_{x_0}^{x_f}$$

$$
A_0 = \frac 1 3 \left

(\left(k^2x_f^5 + 2(R_0k - k^2x_0)x_f^4 + (R_0^2 - 2R_0kx_0 + k^2x_0^2)x_f^3\right)
-
(\left(k^2x_0^5 + 2(R_0k - k^2x_0)x_0^4 + (R_0^2 - 2R_0kx_0 + k^2x_0^2)x_0^3\right)

\right)
$$

$$A_0 = \frac 1 3 \left(

k^2(x_f^5-x_0^5) + 2(R_0k - k^2x_0)(x_f^4-x_0^4) + (R_0^2 - 2R_0kx_0 + k^2x_0^2)(x_f^3-x_0^3)

\right)$$



**$A_1$:**
***
$$
A_1 = \frac{2k}{3}\int_{x_0}^{x_f} x^3(R_0 + k(x - x_0)) \ dx
$$
$$
A_1 = \frac{2k}{3} \int_{x_0}^{x_f} R_0x^3 + kx^4 - kx_0x^3 \ dx
$$

$$
A_1 = \frac{2k}{3}\left[\frac{R_0x^4}{4} + \frac{kx^5}{5} - \frac{kx_0x^4}{4}\right]_{x_0}^{x_f}
$$
$$
A_1 = \frac{2k}{3}\left(\frac{k}{5}\left(x_f^5 - x_0^5\right) + \frac{(R_0 - kx_0)}{4}\left(x_f^4 - x_0^4\right)\right)
$$

**A**
***
$$

A = \pi(A_0 - A_1) = \pi\left(

\frac 1 3 \left(
k^2(x_f^5-x_0^5) + 2(R_0k - k^2x_0)(x_f^4-x_0^4) + (R_0^2 - 2R_0kx_0 + k^2x_0^2)(x_f^3-x_0^3)
\right)

- 

\frac{2k}{3}\left(
\frac{k}{5}\left(x_f^5 - x_0^5\right) + \frac{(R_0 - kx_0)}{4}\left(x_f^4 - x_0^4\right)
\right)

\right)


$$

Combining and rearranging in terms of x again:

$$

A = \frac{\pi}{3} \left(

\left(    k^2(x_f^5-x_0^5) + 2(R_0k - k^2x_0)(x_f^4-x_0^4) + (R_0^2 - 2R_0kx_0 + k^2x_0^2)(x_f^3 - x_0^3)    \right)
-
\left(    \frac{2k^2}{5}(x_f^5 - x_0^5) + \frac{k(R_0 - kx_0)}{2}(x_f^4 - x_0^4)    \right)

\right)

$$
$$
A = \frac{\pi}{3} \left(

\left(    k^2 - \frac{2k^2}{5}    \right)    (x_f^5 - x_0^5)
+
\left(    2R_0k - 2k^2x_0 - \frac{1}{2}k(R_0 - kx_0)    \right)    (x_f^4 - x_0^4)
+
\left(    R_0^2 - 2R_0kx_0 + k^2x_0^2    \right)    (x_f^3 - x_0^3)

\right)
$$
$$
A = \frac{\pi}{3} \left(

\frac{3k^2}{5}    \left(x_f^5 - x_0^5\right)
+
\frac{3k}{2}(R_0 - kx_0)    \left(x_f^4 - x_0^4\right)
+
(R_0 - kx_0)^2    \left(x_f^3 - x_0^3\right)

\right)
$$

**Method 2: Expanding then Evaluating**

This gave the same result as integration by parts :)))))

(Are these the same answer?) YES BESTIE SLAYY
### Finding B

$B = \int_{x_0}^{x_f}\int_{0}^{2\pi}\int_{0}^{R(x)}r^3\cos^2(\theta) \ dr \ d\theta \ dx$

$B = \int_{x_0}^{x_f}\int_{0}^{2\pi}\frac{R(x)^4}{4}\cos^2(\theta) \ d\theta \ dx$

$B = \int_{x_0}^{x_f}\int_{0}^{2\pi}\frac{R(x)^4}{8}(cos(2\theta) + 1) \ d\theta \ dx$

$B = \int_{x_0}^{x_f}\frac{\pi R(x)^4}{4} \ dx$

$$
B = \frac{\pi}{4}\int_{x_0}^{x_f}R(x)^4 \ dx 
= \frac{\pi}{4} \left[\frac{R(x)^5}{5R'(x)}\right]_{x_0}^{x_f} 
= \frac{\pi}{20k}\left[(R_0 + k(x - x_0))^5\right]_{x_0}^{x_f}
$$

It is clear that this expression is undefined when k = 0 (this solution is only valid when the primitive is not a cylinder i.e. $R_0 \neq R_f$)

**Expanding and Simplifying:**

$$
B = \frac{\pi}{20k}\left[
(R_0 + k(x - x_0))^5
\right]_{x_0}^{x_f}
$$

$$
B = \frac{\pi}{20k}\left[

R_0^5 + 5R_0^4k(x-x_0) + 10R_0^3k^2(x-x_0)^2 + 10R_0^2k^3(x-x_0)^3 + 5R_0k^4(x-x_0)^4 + k^5(x-x_0)^5

\right]_{x_0}^{x_f}
$$


Splitting each term out so we can more easily combine for coefficients as $B = \frac{\pi}{20k}\left[B_0 + B_1 + B_2 + B_3 + B_4 + B_5\right]_{x_0}^{x_f}$ :

$B_0 = R_0^5$

$B_1 = 5R_0^4kx - 5R_0^4kx_0$

**$B_2$**
***
$B_2 = 10R_0^3k^2(x-x_0)^2$
$B_2 = 10R_0^3k^2(x^2 - 2x_0x + x_0^2)$
$B_2 = 10R_0^3k^2x^2 - 20R_0^3k^2x_0x + 10R_0^3k^2x_0^2$

**$B_3$**
***
$B_3 = 10R_0^2k^3(x-x_0)^3$
$B_3 = 10R_0^2k^3(x^3 - 3x_0x^2 + 3x_0^2x - x_0^3)$
$B_3 = 10R_0^2k^3x^3 - 30R_0^2k^3x_0x^2 + 30R_0^2k^3x_0^2x - 10R_0^2k^3x_0^3$

**$B_4$**
***
$B_4 = 5R_0k^4(x-x_0)^4$
$B_4= 5R_0k^4(x^4 - 4x_0x^3 + 6x_0^2x^2 - 4x_0^3x + x_0^4)$
$B_4= 5R_0k^4x^4 - 20R_0k^4x_0x^3 + 30R_0k^4x_0^2x^2 - 20R_0k^4x_0^3x + 5R_0k^4x_0^4$

**$B_5$**
***
$B_5 = k^5(x-x_0)^5$ 
$B_5 = k^5(x^5 - 5x_0x^4 + 10x_0^2x^3 - 10x_0^3x^2 + 5x_0^4x - x_0^5)$
$B_5 = k^5x^5 - 5k^5x_0x^4 + 10k^5x_0^2x^3 - 10k^5x_0^3x^2 + 5 k^5x_0^4x - k^5x_0^5$



**Taking coefficients**
***
$c(x^0) = R_0^5 - 5R_0^4kx_0 + 10R_0^3k^2x_0^2 - 10R_0^2k^3x_0^3 + 5R_0k^4x_0^4 - k^5x_0^5$
$c(x) = 5R_0^4k -20R_0^3k^2x_0 + 30R_0^2k^3x_0^2 - 20R_0k^4x_0^3 + 5k^5x_0^4$
$c(x^2) = 10R_0^3k^2 - 30R_0^2k^3x_0 + 30R_0k^4x_0^2 - 10k^5x_0^3$
$c(x^3) = 10R_0^2k^3 - 20R_0k^4x_0 + 10k^5x_0^2$
$c(x^4) = 5R_0k^4 - 5k^5x_0$
$c(x^5) = k^5$

this is such a mess my god (so symmetrical though wowie zowie), giving:

$$
B = \frac{\pi}{20k}\left[
k^5(x^5 - x_0^5)
+ 5k^4(R_0 -kx_0)(x^4 - x_0^4)
+ 10k^3(R_0^2 - 2R_0kx_0 + k^2x_0^2)(x^3 - x_0^3)
+ 10k^2(R_0^3 - 3R_0^2kx_0 + 3R_0k^2x_0^2 - k^3x_0^3)(x^2 - x_0^2)
+ 5k(R_0^4 - 4R_0^3kx_0 + 6R_0^2k^2x_0^2 - 4R_0k^3x_0^3 + k^4x_0^4)(x - x_0)
+ (R_0^5 - 5R_0^4kx_0 + 10R_0^3k^2x_0^2 - 10R_0^2k^3x_0^3 + 5R_0k^4x_0^4 - k^5x_0^5)
\right]_{x_0}^{x_f}
$$



**Factorising coefficients**
***
$$
B = \frac{\pi}{20k}\left[
k^5(x^5 - x_0^5)
+ 5k^4(R_0 -kx_0) \left(x^4 - x_0^4\right)
+ 10k^3(R_0-kx_0)^2  \left(x^3 - x_0^3\right)
+ 10k^2(R_0-kx_0)^3  \left(x^2 - x_0^2\right)
+ 5k(R_0-kx_0)^4  \left(x - x_0\right)
+ (R_0-kx_0)^5
\right]_{x_0}^{x_f}
$$
this seems legit and symmetrical etc so far, dividing by 20k to get in terms of pi:

$$
B = \pi\left[
\frac{1}{20} k^4(x^5 - x_0^5)
+ \frac{1}{4} k^3(R_0 -kx_0) \left(x^4 - x_0^4\right)
+ \frac{1}{2} k^2(R_0-kx_0)^2  \left(x^3 - x_0^3\right)
+ \frac{1}{2} k(R_0-kx_0)^3  \left(x^2 - x_0^2\right)
+ \frac{1}{4} (R_0-kx_0)^4  \left(x - x_0\right)
+ \frac{1}{20k}(R_0-kx_0)^5
\right]_{x_0}^{x_f}
$$

**Taking Limits**
***

$$
B = \pi\left(
\frac{1}{20} k^4(x_f^5 - x_0^5)
+ \frac{1}{4} k^3(R_0 -kx_0) \left(x_f^4 - x_0^4\right)
+ \frac{1}{2} k^2(R_0-kx_0)^2  \left(x_f^3 - x_0^3\right)
+ \frac{1}{2} k(R_0-kx_0)^3  \left(x_f^2 - x_0^2\right)
+ \frac{1}{4} (R_0-kx_0)^4  \left(x_f - x_0\right)
\right)
$$


**Alternative: taking limits at this point**

$R_0$ terms cancel, the rest of $x_0$ limit terms are just 0

$$
B = \frac{\pi}{20k}\left[
5R_0^4k(x_f-x_0) + 10R_0^3k^2(x_f-x_0)^2 + 10R_0^2k^3(x_f-x_0)^3 + 5R_0k^4(x_f-x_0)^4 + k^5(x_f-x_0)^5
\right]
$$

we can remove the k terms here:

$$
B = \frac{\pi}{20}\left[
5R_0^4(x_f-x_0) + 10R_0^3k(x_f-x_0)^2 + 10R_0^2k^2(x_f-x_0)^3 + 5R_0k^3(x_f-x_0)^4 + k^4(x_f-x_0)^5
\right]
$$

$$
B = \frac{\pi(x_f-x_0)}{20}\left[

5R_0^4 + 10R_0^3(R_f-R_0) + 10R_0^2(R_f-R_0)^2+5R_0(R_f-R_0)^3 + (R_f-R_0)^4

\right]
$$

Both of these expressions are equivalent (WHAT THE FUCK?!)

### Combining for $I_{xx}$

Combining A and B by power of x:

$$
A = \pi \left(
\frac{k^2}{5}    \left(x_f^5 - x_0^5\right)
+\frac{k}{2}(R_0 - kx_0)    \left(x_f^4 - x_0^4\right)
+\frac{1}{3}(R_0 - kx_0)^2    \left(x_f^3 - x_0^3\right)
\right),
$$
$$
B = \pi\left(
\frac{1}{20} k^4(x_f^5 - x_0^5)
+ \frac{1}{4} k^3(R_0 -kx_0) \left(x_f^4 - x_0^4\right)
+ \frac{1}{2} k^2(R_0-kx_0)^2  \left(x_f^3 - x_0^3\right)
+ \frac{1}{2} k(R_0-kx_0)^3  \left(x_f^2 - x_0^2\right)
+ \frac{1}{4} (R_0-kx_0)^4  \left(x_f - x_0\right)
\right)
$$


(I am dividing the inside of B by k as I have enforced that it is non-zero)

taking coefficients:

$c(1) = 0$
$c(x_f - x_0) = \frac{1}{4}\left(R_0 - kx_0\right)^4$
$c(x_f^2 - x_0^2) = \frac{k}{2}\left(R_0 - kx_0\right)^3$
$c(x_f^3 - x_0^3) = \frac{2 + 3k^2}{6}(R_0 - kx_0)^2$
$c(x_f^4 - x_0^4) = \frac{2k + k^3}{4}(R_0 - kx_0)$
$c(x_f^5 - x_0^5) = \frac{4k^2 + k^4}{20}$

$$
A + B = \pi\left(
\frac{1}{4}(R_0 - kx_0)^4 \left(x_f - x_0\right)
+ \frac{k}{2}(R_0-kx_0)^3 \left(x_f^2 - x_0^2\right)
+ \frac{2 + 3k^2}{6}(R_0 - kx_0)^2 \left(x_f^3 - x_0^3\right)
+ \frac{2k + k^2}{4}(R_0 - kx_0) \left(x_f^4 - x_0^4\right)
+ \frac{4k^2 + k^4}{20} \left(x_f^5 - x_0^5\right)
\right)
$$

as $I_{yy} = \rho(A + B)$,

$$
I_{yy} = \frac{\pi\rho}{2}\left(
\frac{1}{2}(R_0 - kx_0)^4 \left(x_f - x_0\right)
+ k(R_0-kx_0)^3 \left(x_f^2 - x_0^2\right)
+ \frac{2 + 3k^2}{3}(R_0 - kx_0)^2 \left(x_f^3 - x_0^3\right)
+ \frac{k(2+k)}{2}(R_0 - kx_0) \left(x_f^4 - x_0^4\right)
+ \frac{k^2(4 + k^2)}{10} \left(x_f^5 - x_0^5\right)
\right)
$$

WRONG EEHH AAWWWHHH *Steve Harvey stare*

This combination has gone wrong somewhere, as both A and B have been verified and by simply summing the terms we arrive at the correct moment of inertia.

