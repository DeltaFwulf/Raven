https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html

# Section II. MODELLING[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Section-II.-MODELING)

# Chapter 4. 3D Rotations[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Chapter-4.-3D-Rotations)

Although the previous chapter discussed how three-dimensional rotations in SO(3) can be represented as 3x3 matrices, this is not usually the most convenient representation. There are a number of alternative rotation representations in frequent use in robotics, aviation, CAD, computer vision, and computer graphics. Although SO(3) is a 3-dimensional space, it is fundamentally distinct from Cartesian space in a _topological_ sense, meaning that elements of the space of rotation do not "connect" in the same way that normal points in space do. In this chapter we will discuss the meaning of rotation matrices in more detail, as well as the common representations of Euler angles, angle-axis form and the related rotation vector form, and quaternions.

Each representation, in some sense, equivalent, since each may be mapped to a rotation transform; however, certain representations are more convenient for certain tasks, like inversion, composition, interpolation, and sampling. All representations have some weaknesses as well, and there is no "ideal" rotation representation. We will also discuss how to represent continuous changes of rotation, and to calculate derivatives.

## 1  Topology primer: rotations in 2D[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Topology-primer:-rotations-in-2D)

Not only is it relatively difficult to visualise 3D rotations compared to translations, rotations behave in a fundamentally different fashion than translations. To get a better understanding of rotations, we will first introduce a little bit of _topology_, a branch of mathematics that studies how spaces are connected. Topological connectivity properties are preserved under arbitrary invertible transformations, and hence they are fundamental characteristics of the spaces themselves, and not simply their coordinate representations.

We will start by introducing topological concepts in terms of the space of 2D rotations SO(2). The immediate concern that one may have about 2D rotations is that angles "wrap around", so that the direction at $0^\circ$ is identical to $360^\circ$ or any multiple thereof. This is an important issue in many contexts:

- For a robot to rotate from a heading of $30^\circ$ to $330^\circ$, it is faster to rotate the $60^\circ$ CW rather than the $300^\circ$ CCW.
    
- If a joint has greater than a $360^\circ$ range of motion, then its internal position at a given time can only be measured externally modulo $360^\circ$: to estimate absolute position using external means, the history of all the joint's movements must be remembered.
    
- When retrieving items from a table (of control commands or motions, for example) indexed by an angle, one must be careful to retrieve the closest item in terms of absolute angular deviation modulo $360^\circ$ rather than simply taking the absolute difference.
    

Although there is a one-to-one mapping between each element of SO(2) and an angle in the interval $[0,2\pi)$, the "wrap around" issue illustrates that SO(2) is fundamentally not equivalent because its elements do not _behave_ like numbers on the interval when discussing continuity and proximity, as shown in [Fig. 1](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#fig:TopologyRotation2D).

---

![fig:TopologyRotation2D](https://motion.cs.illinois.edu/RoboticSystems/figures/modeling/topology_rotation_2d.svg)

Figure 1. Top: The topology of the space of 2D rotations is homeomorphic to a circle, but not an interval. Although there is a one-to-one mapping between points on the circle and the half-open interval $[0,2\pi)$, the discontinuity at $2\pi$ makes them topologically distinct. Bottom: The shortest path between the two marked points on the circle (solid arrow) wraps around $2\pi$, which is a discontinuous path on the interval. The shortest continuous path on the interval (dashed arrow) results in a longer path on the circle.

---

### 1.1  Basics concepts in topology[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Basics-concepts-in-topology)

The topology of a space describes the connectedness of paths through it. Paths in a space $\mathcal{X}$ are continuous functions $x(s) : [0,1] \rightarrow \mathcal{X}$. Two spaces are topologically equivalent if there exists a bijection between them that preserves path continuity. Such a bijection is called a _homeomorphism_, and topologically equivalent spaces called _homeomorphic_. If there exists no such bijection, they are topologically distinct (not homeomorphic).

Since any mapping between SO(2) and $[0,2\pi)$ breaks the continuity of paths through the 0 angle, they are topologically distinct. It can be shown, however, that SO(2) is homeomorphic to the unit circle $S_1$. The function mapping an angle $\theta$ to the point $(\cos \theta, \sin \theta)$ on the circle (i.e., the direction vector with heading $\theta$) is a bijection. The inverse mapping is an operation known as the _argument_ of the point, and is similar to $\tan^{-1} (y/x)$ but respect the quadrant of each point $(x,y)$ and also handles the case where $x=0$. (In many programming languages this is given as a basic subroutine $atan2(y,x)$.) It can be shown that this bijection preserves the continuity of paths through both SO(2) and $S_1$: continuous paths on SO(2) map to continuous paths on $S_1$ and vice versa, even if they pass through the 0 angle.

As a result, rotational motion is fundamentally similar to motion on a circle. When interpolation is requested between two angles, what is usually wanted is the _shortest_ path between those angles. As a result, rotational interpolation either requires specifying the direction of rotation, or instead requires examining and minimising the distance travelled both CW and CCW.

### 1.2  Geodesic distance and interpolation[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Geodesic-distance-and-interpolation)

The path between two points in a space with minimum distance is known as a _geodesic_. Geodesics are the most direct route between two points on a space, and in Cartesian spaces are simply straight lines. The _geodesic distance_ is a function $d(x_1,x_2)$ giving the length of a geodesic between two points. In more general topological spaces, geodesics are often difficult to calculate, but there are closed form solutions in SO(2) and SO(3).

We define the signed angular distance between two angles $\theta_1$ and $\theta_2$ represented in the range $[0,2\pi)$ as the function:

$$
diff(\theta_2,\theta_1) = \left\lbrace \begin{array}{ll} \theta_2 - \theta_1 & \text{if } -\pi < \theta_2 - \theta_1 \leq \pi \\ \theta_2 - \theta_1 - 2\pi & \text{if } \theta_2 - \theta_1 > \pi \\ \theta_2 - \theta_1 + 2\pi & \text{if } \theta_2 - \theta_1 \leq -\pi \end{array} \right. $$

which results in a value in the range $(-\pi,\pi]$ that produces the angular displacement of $\theta_2$ from $\theta_1$ with minimum absolute value, and whose value is positive for CCW displacement and negative for CW displacement.

The geodesic between two angles is an interpolation among the signed angular distance:

$$\theta(s) = (\theta_1 + s \cdot diff(\theta_2,\theta_1)) \mod (2\pi).$$

The geodesic distance on $SO(2)$ is absolute angular distance, which is simply the absolute value of the signed distance

$$d(\theta_1,\theta_2) = | diff(\theta_2,\theta_1) |.$$

## 2  Representing elements of topological spaces[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Representing-elements-of-topological-spaces)

The most common topological spaces we discuss in robotics include:

- $\mathbb{R}^n$: Cartesian space of $n$ dimensions
    
- $SO(n)$: The special orthogonal group in $n$ dimensions. This is defined as the set of $n \times n$ orthogonal matrices with determinant +1. As we have seen before, SO(2) is the space of 2D rotations, and SO(3) is the space of 3D rotations.
    
- $SE(n)$: The special Euclidean group in $n$ dimensions. This is the set of all rigid transformations in dimension $n$, and is equivalent to $SO(n) \times \mathbb{R}^n$.
    
- $S_n$: the $n$-dimensional sphere. $S_1$ is the circle, and $S_2$ is the standard sphere. It is important to note that this space denotes the _surface_ of the sphere, and $n$ is the _intrinsic dimensionality_ of the surface, even though we typically embed $S_1$ in a 2D plane, and $S_2$ in a 3D space.
    

It is also possible to compose multiple spaces with the Cartesian product, such as $S_1 \times S_1$, which is the topology of the torus. We also use a power notation $S_1 \times S_1 = S_1^2$ to denote repeated applications of the Cartesian product.

Although we usually use a vector of parameters to describe elements of these spaces, this representation (in $\mathbb{R}^n$) does not capture their connectivity perfectly. The vector representation will have deficiencies in having double representations of the same object, infinite representations of the same object (singularities), and in interpolation, which may not produce geodesics.

Let us first consider a simpler example that will illustrate several of the issues we will come across when representing 3D rotations. The latitude and longitude representation of the unit sphere (spherical coordinates), where each point has coordinates

$$\mathbf{x}(\theta,\phi) = \begin{bmatrix} \cos \theta \cos \phi \\ \sin \theta \cos \phi \\ \sin \phi \end{bmatrix}.$$

Here $\theta \in [0,2\pi)$ gives the longitude and $\phi \in [-\pi/2,\pi/2]$ gives the latitude.

Since each point on the sphere can be represented by some $(\theta,\phi)$ pair, this shows that the inherent dimensionality of $S_2$ is no more than 2. However, there are two singularities of this representation, namely, the north and south poles (0,0,1) and (0,0,-1), in which a point on the sphere have an infinite number of 2D representations. $\mathbf{x}(\theta,\pi/2) = (0,0,1)$ for any value of $\theta$, and $\mathbf{x}(\theta,-\pi/2) = (0,0,-1)$ for any value of $\theta$.

The nonlinearity in this mapping also means that as a point approaches and passes either pole, the representation of the point swings wildly. If $\mathbf{x} = (x,y,z)$, we have the inverse relation $\phi = \sin^{-1} z$, and $\theta = atan2(y,x)$ (which is undefined for $x=y=0$). Observe the $\theta$ variable in the following plot as the arc gets closer to passing through the pole (blue and black curves):

In [ ]:

#@title↔

![](https://motion.cs.illinois.edu/RoboticSystems/3DRotations_files/3DRotations_8_0.png)

Another limitation of this representation is that interpolation is not as straightforward as simply blending between two points. Suppose two points $\mathbf{a}$ and $\mathbf{b}$ on the sphere are represented by spherical coordinates $(\theta_a,\phi_a)$ and $(\theta_b,\phi_b)$, respectively. In general, the straight line path from $(\theta_a,\phi_a)$ and $(\theta_b,\phi_b)$ does not map to the shortest line between $\mathbf{a}$ and $\mathbf{b}$ on the sphere. First of all, the connectivity between 0 and $2\pi$ in the $\theta$ dimension (that is, the topology of SO(2) in the domain of $\theta$) needs to be taken into account. But even with that handled properly, these straight line paths are still not the shortest paths on the sphere — in other words, straight line interpolation in the representation space does not produce geodesic paths on the sphere.

You will have observed this phenomenon if you have ever taken an intercontinental flight between two destinations somewhat north of the equator. For example, the flight path from New York to London, when seen on a traditional map, is actually a curve that arcs above the straight line path from New York to London. An even more extreme example is when flying from New York to Beijing: the flight path actually passes close to the North Pole! Airlines are extremely sensitive to fuel usage, so they prefer to travel around the Earth along minimum length paths. The mathematical solution to this problem is to interpolate along _great circles_, which are geodesics on the sphere. Solving for and proving optimality of geodesics is of great interest, and as we shall see below, luckily there are closed-form solutions for geodesics in SO(3) as well.

## 3  3D rotation matrices[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#3D-rotation-matrices)

Now let us return back to the 3D rotation case. As described before, 3D rotations are $3\times 3$ matrices with the following entries:

$$R = \begin{bmatrix} r_{11} & r_{12} & r_{13} \\ r_{21} & r_{22} & r_{23} \\ r_{31} & r_{32} & r_{33} \end{bmatrix}$$

There are 9 parameters in the matrix, but not all possible values of 9 parameters correspond to valid rotation matrices. To qualify as a rotation, the matrix must satisfy the two properties:

1. Orthogonality: $R^T R = I$ (or equivalently, $R R^T = I$).
    
2. Positive orientation: $\det(R) = 1$.
    

The first condition imposes 9 equality constraints, except that 3 of them are redundant. As a result, 6 degrees of freedom are removed from the 9 free parameters, reducing the set of possible rotation matrices to a 3-dimensional set. The second condition only reduces the set of valid parameters by half, because all matrices that satisfy condition 1 must have either determinant +1 or -1 (because the determinant is distributive, $\det(AB) = \det(A)\det(B)$, and invariant to transpose, $\det(A) = \det(A^T)$).

As a result, the space of 3D rotations is itself 3D, and hence _no fewer than 3 continuous parameters are needed to represent all possible rotations_. However, the topology of SO(3) is very different from $\mathbb{R}^3$ in that it is bounded rather than infinite, and wraps around "in all directions", so to speak.

### 3.1  Axis-aligned rotations[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Axis-aligned-rotations)

Rotations about individual axis are the most straightforward to compute, because the behavior of one axis is unchanged and the other two axes undergo a 2D rotation along the orthogonal plane ([Fig. 2](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#fig:AxisAlignedRotations)).

---

![fig:AxisAlignedRotations](https://motion.cs.illinois.edu/RoboticSystems/figures/modeling/axis_aligned_rotations.svg)

Figure 2. Axis-aligned rotations.

---

First, the matrix for rotation about the $Z$ axis contains a 2D rotation matrix in its upper corner:

$$R_Z(\theta) = \begin{bmatrix} \cos \theta & -\sin \theta & 0 \\ \sin\theta & \cos \theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

This can be interpreted by imagining the $Z$ axis pointing out of the page, and the $X$ and $Y$ axes marking the axes of a standard graph on the page. The rotation about $\theta$ is a CCW rotation in the plane of the page. Notice that the $Z$ coordinate of any point is preserved by this operation, a property maintained by the third row $(0,0,1)$, nor does it affect the $X$ and $Y$ coordinates, a property maintained by the first two 0 entries of the third column. In the $(X,Y)$ plane, the upper $2\times 2$ matrix is identical to a 2D rotation matrix.

The rotation about the $X$ axis is similar, with a 2D rotation matrix appearing along the $Y$ and $Z$ axes:

$$R_X(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos \theta & -\sin \theta \\ 0 & \sin\theta & \cos \theta \end{bmatrix}.$$

Here the $X$ coordinate is preserved while the $Y$ and $Z$ entries perform a 2D rotation.

Finally, the rotation about the $Y$ axis is similar, but with a sign switch of the $\sin$ terms:

$$R_Y(\theta) = \begin{bmatrix} \cos \theta & 0 & \sin\theta \\ 0 & 1 & 0\\ -\sin \theta & 0 & \cos \theta \end{bmatrix}.$$

The reason why the $-\sin \theta$ term switches to below the diagonal is that if one were to orient a frame so the $Y$ axis points out of the page, and align the $X$ axis to point rightward, the $Z$ axis would point downward instead of upward. Instead, the matrix can be derived by aligning the $Z$ axis to the rightward direction and the $X$ axis to the upward direction, so that the $-\sin \theta$ term arrives in the $Z,X$ spot. A mnemonic to help remember the sign switch on rotations about $Y$ is that the order of the two coordinate directions defining the orthogonal plane is derived from a _cyclic_ ordering of the axes: $Y$ follows $X$, $Z$ follows $Y$, and $X$ follows $Z$. So, the plane orthogonal to $X$ is $(Y,Z)$, the plane orthogonal to $Y$ is $(Z,X)$, and the plane orthogonal to $Z$ is $(X,Y)$.

### 3.2  Interpretation[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Interpretation)

Suppose the rotation is such that the pre-rotation axes $X$, $Y$, and $Z$, are rotated to post-rotation axes $X^\prime$, $Y^\prime$, and $Z^\prime$, respectively. The first column represents the coordinates of $X^\prime$ axis _relative to the original frame_ after rotation

$$\mathbf{x}^\prime = \begin{bmatrix} r_{11} \\ r_{21} \\ r_{31} \end{bmatrix}.$$

Similarly, the second column gives $\mathbf{y}^\prime = (r_{12},r_{22},r_{32})$ and the third column gives $\mathbf{z}^\prime = (r_{13},r_{23},r_{33})$.

Moreover, since $R^{-1} = R^T$, the coordinates of the old frame's axes _relative to the new frame_ are the individual rows of $R$.

Due to the cosine rule, $r_{11}$ is the cosine of the angle between the old and new $X$ axis (because it is the dot product between $\mathbf{x}^\prime$ and $\mathbf{x} = (1,0,0)$. Similarly, the other diagonal elements give the cosine of the angle between the other axes.

### 3.3  Operations[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Operations)

**Applying rotation to a point.** As we have seen before, applying rotation to a point (i.e., deriving the coordinates of a rotated point in the original coordinate frame) is simply matrix-vector multiplication.

**Composition.** Composition between two rotation matrices $R_1$ and $R_2$ is simple a matrix-matrix multiplication $R_1 R_2$. Note that this matrix is the result of applying $R_2$ first and then $R_1$, and is not the same as the converse.

**Inversion.** As we have seen above, the inverse rotation matrix is simply the matrix transpose.

### 3.4  Discussion[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Discussion)

Properly composing rotations can be confusing due to the convention chosen about whether rotation axes are considered fixed to the world reference frame, or rotating along with the local reference frame. In this class we will use the former convention, which is known as _extrinsic_ rotation. Suppose that $R_1$ is a rotation about axis $A$ and $R_2$ is a rotation about axis $B$. When composed to $R_1 R_2$, this means that a point $P$ given in local coordinates $\mathbf{p}$ will first get rotated about axis $B$ to obtain a point $P^\prime$ and then about axis $A$ _where $A$ is interpreted as being fixed in the un-rotated reference frame_ to obtain $P^{\prime\prime}$. The result $R_1 R_2 \mathbf{p}$ gives the coordinates of $P^{\prime\prime}$ relative to the un-rotated reference frame, as shown in [Fig. 3](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#fig:RotationComposition).

---

![fig:RotationComposition](https://motion.cs.illinois.edu/RoboticSystems/figures/modeling/rotation3d_composition.svg)

Figure 3. The composition $R_1\cdot R_2$ of two rotation matrices corresponds to first performing $R_2$, then performing $R_1$. This means that the coordinate frame represented by $R_2$ is rotated about the axis of $R_1$ _in the original frame_.

---

The confusion often lies when axes of rotation are considered attached to frames already being rotated (_intrinsic_ rotations), which happens when trying to solve problems like the following: "Let $P$ be attached to a rigid body $B$ and have coordinates $(1,2,3)$ relative to $B$'s coordinate frame. Find the world coordinates of the point after rotating $B$ about its $Z$ axis by $90^\circ$, and then by another $90^\circ$ about $B$'s local $X$ axis, and finally translating its origin by the offset $\mathbf{t}(10,0,5)$."

Naïvely performing each transformation in sequence produces the wrong answer:

1. The rotation $R_Z(90^\circ)$ maps $(1,2,3)$ to $(-2,1,3)$.
    
2. The rotation $R_X(90^\circ)$ maps $(-2,1,3)$ to $(-2,-3,1)$.
    
3. The translation maps $(-2,-3,1)$ to $(8,-3,6)$.
    

This answer is incorrect, and is also the answer that would have been obtained through composing the matrices:

$$R_X(90^\circ) R_Z(90^\circ) \mathbf{p} + \mathbf{t}$$

Instead, the key problem is that after the first rotation, the local $X$ axis has rotated along with the body, and is no longer equivalent to the $X$ axis in world coordinates! Instead, it is aligned with the world's $Y$ axis. The correct sequence of operations is:

1. The rotation $R_Z(90^\circ)$ maps $(1,2,3)$ to $(-2,1,3)$.
    
2. The rotation $R_Z(90^\circ)$ also maps the local $X$ axis to the world $Y$ axis.
    
3. The rotation $R_Y(90^\circ)$ maps $(-2,1,3)$ to $(3,1,2)$.
    
4. The translation maps $(3,1,2)$ to $(13,1,7)$.
    
 aligned. The second rotation about ￼￼ is performed with respect to the world ￼￼ axis, which is also equivalent to ￼￼'s original ￼￼ axis. In general, intrinsic rotations are composed in reverse order to extrinsic rotations.

It is also confusing that the correct result is also obtained by switching the order of the rotations!

$$R_Z(90^\circ) R_X(90^\circ) \mathbf{p} + \mathbf{t}.$$

The reason why this switched order works is that in the first rotation about $X$, the local and world $X$ axis are aligned. The second rotation about $Z$ is performed with respect to the world $Z$ axis, which is also equivalent to $B$'s original $Z$ axis. In general, intrinsic rotations are composed in reverse order to extrinsic rotations.

## 4  Euler angles[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Euler-angles)

Euler angles are a three-parameter representation of rotations $(\phi,\theta,\psi)$, and are derived from the definitions of $R_X$, $R_Y$, and $R_Z$ above. They are one of the oldest rotation representations, are easy to interpret, and are also frequently used in aeronautics and robotics. The basic idea is to select three different axes and represent the rotation as a composite of three axis-aligned rotations. The order in which axes are chosen is a matter of convention.

### 4.1  Conventions[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Conventions)

For example, the _roll-pitch-yaw_ convention often used in aerospace assumes that a vehicle's roll angle is about its $X$ axis, pitch is about its $Y$ axis, and yaw is about its $Z$ axis ([Fig. 4](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#fig:RollPitchYaw)), with the composite rotation given by:

$$R_{rpy}(\phi,\theta,\psi) = R_Z(\phi) R_Y(\theta) R_X(\psi)$$

Notice that the order of rotation axes is $Z$, $Y$, $X$, and this is also known as $ZYX$ convention. (Note that in order of application, this applies roll (X) first, then pitch (Y), then yaw (Z).)

---

![fig:RollPitchYaw](https://motion.cs.illinois.edu/RoboticSystems/figures/modeling/roll_pitch_yaw.svg)

Figure 4. Roll-pitch-yaw convention consists of a roll about the vehicle's forward direction, a pitch about its leftward direction, and a yaw around its upward direction.

---

There are a multitude of other conventions possible, each of the form

$$R_{ABC}(\phi,\theta,\psi) = R_A(\phi) R_B(\theta) R_C(\psi)$$

where $A$, $B$, and $C$ are one of $X$, $Y$, or $Z$. To be a valid convention, the span of possible results from the convention must span the range of possible rotation matrices, and this means that no two subsequent axes may be the same, e.g., $XXY$ is not permissible, since two combined rotations about one axis are equivalent to a single rotation about that axis. However, $A$ and $C$ may indeed be the same, for example, in $ZYZ$ convention:

$$R_{ZYZ}(\phi,\theta,\psi) = R_Z(\phi) R_Y(\theta) R_Z(\psi)$$

Here the intervening $Y$ rotation modifies the axis by which one of the $R_Z$ terms rotates, and can in fact span all rotation matrices.

### 4.2  Conversion between matrices[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Conversion-between-matrices)

Conversion from Euler angles to rotation matrices is a straightforward computation of $\eqref{eq:EulerAngles}$. The converse is more challenging and requires calculating the inverse of the forward conversion using some trigonometry.

First, for the given convention we would begin by equating the matrix terms to the sine and cosine terms of the computed rotation matrix, e.g., for roll-pitch-yaw convention:

$$R_{rpy} = \begin{bmatrix} r_{11} & r_{12} & r_{13} \\ r_{21} & r_{22} & r_{23} \\ 
r_{31} & r_{32} & r_{33} \end{bmatrix} = \begin{bmatrix} c_1 c_2 & c_1 s_2 s_3 - s_1 c_3 & s_1 s_3 + c_1 s_2 c_3 \\
s_1 c_2 & c_1 c_3 + s_1 s_2 s_3 & s_1 s_2 c_3 - c_1 s_3 \\
- s_2 & c_2 s_3 & c_2 c_3 \end{bmatrix}
$$

where we use the shorthand $c_1 = \cos \phi$, $s_1 = \sin \phi$, $c_2 = \cos \theta$, $s_2 = \sin \theta$, $c_3 = \cos \psi$, and $s_3 = \sin \psi$.

The simplest term is in the lower left corner, so we find one of the two solutions to $r_{31} = -\sin \theta$, namely $\theta = - \sin^{-1} r_{31}$ or $\theta = - \sin^{-1} r_{31} + \pi$. We also have two possible solutions $c_2 = \pm \sqrt{1 - s_2^2}$. If $c_2$ is nonzero, then we can divide $r_{11}$ and r$_{21}$ by $c_2$ to obtain $s_1$ and $c_1$, respectively, by which we can obtain $\phi$ via the argument of $(s_1, c_1)$. Similarly, we may derive $\psi$ by dividing $r_{32}$ and $r_{33}$ by $c_2$. Finally one of the two possible solutions to $\theta$ can be obtained by verifying the solution to the upper right entries of the matrix.

The other case to consider is when $c_2$ is zero, which indicates that the pitch is $\pm \pi / 2$. If this is the case, then only $r_{31}$ and the $2 \times 2$ entries in the upper right are nonzero. This corresponds to a _singular_ case in which infinite solutions for $\phi$ and $\psi$ exist. All of these solutions have $\phi + \psi$ equal to the argument of $(-r_{12} / s_2 ,r_{22} / s_2)$.

### 4.3  Singularities, aka gimbal lock[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Singularities,-aka-gimbal-lock)

The minimal range of Euler angles $(\phi,\theta,\psi)$ to cover the span of rotations is the set $[0,2\pi) \times [-\pi/2,\pi/2] \times [0,2\pi)$. However, this set is not topologically equivalent to SO(3). There are certain cases in which a single rotation has an infinite number of solutions. For example, in $ABA$ convention, any pure rotation about the $A$ axis can be represented by Euler angles with $\theta=0$ but infinitely many values of $\phi$ and $\psi$ with constant sum. In roll-pitch-yaw convention, pitches of $\pm \pi/2$ align the roll and yaw axes, and hence when a vehicle is pointed directly upward there are an infinite number of solutions for $\phi$ and $\psi$.

Cases like this are known as _singular_. By analogy with the gimbal mechanism which is a physical device with three rotating axes, _gimbal lock_. Gimbals are devices often used in gyroscopes to measure 3D orientation by means of a rapidly spinning mass which maintains its absolute orientation as its cradle rotates. Gimbal lock manifests itself when two axes of the mechanism become aligned, at which point the gimbal readings become useless because most rotations of the cradle fail to de-align the axes properly.

In calculations, singularities cause problems for conversions, calculating derivatives, and interpolation.

### 4.4  Inversion[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Inversion)

The inverse of an Euler angle $(\phi,\theta,\psi)$ with convention $ABC$ is another set of Euler angles $(-\psi,-\theta,-\phi)$ with convention $CBA$. This is quite convenient when an $ABA$ convention is used, because the reverse convention is the same as the forward. However, when $C \neq A$, finding the inverse Euler angles in the same convention is much less convenient, because it requires conversion to matrix form and then back to Euler angle form.

## 5  Axis-angle and rotation vector representations[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Axis-angle-and-rotation-vector-representations)

Another popular rotation representation is axis-angle form, which represents a rotation matrix as a rotation of a given angle about a given unit axis ([Fig. 5](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#fig:AxisAngleRotation)). Perhaps it is a surprising fact that every rotation can be represented in this form!

---

![fig:AxisAngleRotation](https://motion.cs.illinois.edu/RoboticSystems/figures/modeling/axis_angle_rotation.svg)

Figure 5. Any rotation matrix (bold) can be represented by a rotation of some angle about some axis.

---

Axis-angle representation uses a pair $(\mathbf{a},\theta)$ where $\mathbf{a}$ is a unit 3D vector and $\theta$ is an angle in the range $[0,\pi]$, for a total of 4 parameters. However, the constraint $\| \mathbf{a} \| = 1$ removes a degree of freedom, again spanning the 3-D space of rotations.

A related representation, which we call the _rotation vector_ form improves upon some of the deficiencies of axis-angle representation by simply encoding the axis as the direction and the angle as the length of a 3D vector $\mathbf{m} = \theta \mathbf{a}$. They are easy to convert between, with $\theta = \| \mathbf{m} \|$ and $\mathbf{a} = \mathbf{m} / \| \mathbf{m} \|$ if $\| \mathbf{m} \| \neq 0$, or some arbitrary vector like $(1,0,0)$ otherwise. The range of valid parameters is a closed ball of radius $\pi$. It should be noted that the rotation vector representation has been referred to many names in the literature, including Rodrigues parameters, exponential coordinates, or angular velocity. (It will become extremely useful when we discuss the derivative of rotation matrices in the context of the angular velocity of rigid bodies.)

### 5.1  Converting to rotation matrix[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Converting-to-rotation-matrix)

To convert from axis-angle form to rotation matrices, we use _Rodrigues' formula_. The derivation of Rodrigues' formula starts by decomposing a rotated point into its coordinate about the axis $\mathbf{a}$ and its coordinates about an orthogonal plane. The planar coordinates are then rotated by a 2D rotation of angle $\theta$.

Any point $\mathbf{p}$ can be decomposed into a sum of a component $\mathbf{p}_\parallel$ parallel to $\mathbf{a}$ and a component $\mathbf{p}_\perp$ orthogonal to $\mathbf{a}$ such that $\mathbf{p} = \mathbf{p}_\parallel + \mathbf{p}_\perp$.

$$\mathbf{p}_\parallel = (\mathbf{a} \cdot \mathbf{p}) \mathbf{a}$$$$\mathbf{p}_\perp = \mathbf{p} - (\mathbf{a} \cdot \mathbf{p}) \mathbf{a}$$

The rotation leaves $\mathbf{p}_\parallel$ unchanged, and rotates the $\mathbf{p}_\perp$ component about the axis to a new vector $\mathbf{p}_\perp^\prime$, with a changed direction but an unchanged magnitude.

To derive $\mathbf{p}_\perp^\prime$ we establish a basis for the orthogonal plane to $\mathbf{a}$. One axis on this plane, call it $\mathbf{u}$, is in the direction of the cross product $\mathbf{a} \times \mathbf{p}$, which is nonzero as long as $\mathbf{p}_\perp \neq 0$. A second orthogonal axis, call it $\mathbf{v}$, is in the direction of a second cross product $\mathbf{a} \times (\mathbf{a} \times \mathbf{p})$. This is in fact an expression of the negative direction of $\mathbf{p}_\perp$: $\mathbf{p}_\perp = -\mathbf{a} \times (\mathbf{a} \times \mathbf{p})$.

As a result, the coordinates of $\mathbf{p}_\perp$ given these (non-unit) axes are $(0,-1)$, so that post rotation, $\mathbf{p}_\perp^\prime$ will have coordinates $(\sin \theta, - \cos \theta)$. Projecting these coordiantes back into the original space, we have

$$\mathbf{p}_\perp^\prime = \sin \theta (\mathbf{a} \times \mathbf{p}) + \cos \theta \mathbf{p}_\perp$$

Reconstructing the rotated point $\mathbf{p}^\prime$, we obtain:

$$\mathbf{p}^\prime = (\mathbf{a} \cdot \mathbf{p}) \mathbf{a} + \sin \theta (\mathbf{a} \times \mathbf{p}) + \cos \theta (\mathbf{p} - (\mathbf{a} \cdot \mathbf{p}) \mathbf{a}))$$

which is often rearranged to group terms:

$$\mathbf{p}^\prime = \cos \theta \mathbf{p} + \sin \theta (\mathbf{a} \times \mathbf{p}) + (1-\cos \theta) (\mathbf{a} \cdot \mathbf{p}) \mathbf{a}.$$

Although this formula gives an expression for the transform, we can also rearrange terms to express it as a matrix. We will use the cross-product matrix operator notation

$$\hat{\mathbf{v}} = \begin{bmatrix} 0 & -v_z & v_y \\ v_z & 0 & -v_x \\ -v_y & v_x & 0 \end{bmatrix},$$

so called because

$\hat{\mathbf{v}} \mathbf{w} = \mathbf{v} \times \mathbf{w}$.

Hence we can express the term $\mathbf{a} \times \mathbf{p} = \hat{\mathbf{a}} \mathbf{p}$ as a matrix multiplication. Also, we can express the term $\mathbf{p} - (\mathbf{a} \cdot \mathbf{p}) \mathbf{a} = - \mathbf{a} \times (\mathbf{a} \times \mathbf{p}) = -\hat{\mathbf{a}}^2 \mathbf{p}$,

$$\begin{split} \mathbf{p}^\prime &= \cos \theta \mathbf{p} + \sin \theta (\mathbf{a} \times \mathbf{p}) + (1-\cos \theta) (\mathbf{a} \cdot \mathbf{p}) \mathbf{a} - \mathbf{p}) + (1-\cos \theta) \mathbf{p} \\ &=\mathbf{p} + \sin \theta \hat{\mathbf{a}} \mathbf{p} + (1-\cos\theta) \hat{\mathbf{a}}^2 \mathbf{p} \\ &= (I + \sin \theta \hat{\mathbf{a}} + (1-\cos\theta) \hat{\mathbf{a}}^2 ) \mathbf{p} \end{split}$$

The parenthesised term is the rotation matrix corresponding to the axis-angle representation:

$$R_{aa}(\mathbf{a},\theta) = I + \sin \theta \hat{\mathbf{a}} + (1-\cos\theta) \hat{\mathbf{a}}^2.$$

For rotation vector representation we simply convert to axis-angle form and use the formula above.

$$R_m(\mathbf{m}) = R_{aa}(\mathbf{m} / \| \mathbf{m} \|, \| \mathbf{m} \|)$$

### 5.2  Converting from a rotation matrix[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Converting-from-a-rotation-matrix)

To convert back from a rotation matrix $R$ to axis-angle representation, we can calculate the entries of $R_{aa}(\mathbf{a},\theta)$ and invert the equation to match the entries of $R$:

$$R = \begin{bmatrix} r_{11} & r_{12} & r_{13} \\ r_{21} & r_{22} & r_{23} \\ r_{31} & r_{32} & r_{33} \end{bmatrix} = \begin{bmatrix} \cos\theta I + \sin \theta \hat{\mathbf{a}} + (1-\cos\theta ) \mathbf{a} \mathbf{a}^T \end{bmatrix}$$

where we have used the fact that

$$\hat{\mathbf{a}}^2 = \begin{bmatrix} 0 & -a_z & a_y \\ a_z & 0 & -a_x \\ -a_y & a_x 0 \end{bmatrix}^2 = \begin{bmatrix} -a_z^2 - a_y^2 & a_x a_y & a_x a_z \\ a_x a_y & -a_x^2 - v_z^2 & a_y a_z \\ a_x a_z & a_y a_z & -a_x^2 - a_y^2 \end{bmatrix} = \mathbf{a} \mathbf{a}^T - I.$$

Taking the trace of $R$, we get

$$tr(R) = 3 \cos \theta + (1-\cos \theta) tr(\mathbf{a} \mathbf{a}^T) = 3 \cos \theta + (1 - \cos\theta) (\mathbf{a} \cdot \mathbf{a}) = 1 + 2 \cos \theta.$$

Hence, we have the relationship:

\begin{equation} \theta = \cos^{-1} \left(\frac{tr(R)-1}{2}\right) \label{eq:RotationAbsoluteAngle} \end{equation}

which gives the _absolute angle_ of a rotation matrix. This is also significant because it is the minimal angle needed to rotate from the identity matrix to $R$, and hence is a useful pseudo-norm for 3D rotations.

Next, we can determine $\mathbf{a}$ using the off-diagonal components of the matrix. We see that the difference $r_{12} - r_{21} = -a_z \sin \theta + (1-\cos \theta) a_x a_y - (a_z \sin \theta + (1-\cos \theta) a_x a_y) = -2 a_z \sin \theta$, canceling out the contribution of the matrix $\mathbf{a} \mathbf{a}$ since it is symmetric. Hence if $\sin \theta \neq 0$, we can determine

$$\begin{split} a_x &= (r_{32}-r_{23}) / 2 \sin \theta \\ a_y &= (r_{13}-r_{31}) / 2 \sin \theta \\ a_z &= (r_{21}-r_{12}) / 2 \sin \theta. \end{split}$$

In the case where $\sin \theta = 0$, we could either have the case where $\theta = 0$, in which case $R$ is an identity matrix and any axis will do, or $\theta = \pi$, and more work is needed to determine the true axis of rotation. In this case, $\cos \theta = -1$, so $R = -I + 2 \mathbf{a} \mathbf{a}^T$. We can then determine, up to sign, $a_x = \pm \sqrt{(r_{11} + 1)/2}$, $a_y = \pm \sqrt{(r_{22} + 1)/2}$, and $a_z = \pm \sqrt{(r_{33} + 1)/2}$, with the signs of each element (up to a sign of the entire vector) determined by the off-diagonal components.

### 5.3  Singularities[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Singularities)

There are two singularities for axis-angle representation. The first is at $\theta = 0$, where every axis represents the identity rotation. The rotation vector representation does not have this singularity.

The second is at $\theta = \pi$, where each axis and its negation represent the same rotation. This is a potential problem in extrapolation or interpolation because further rotations flip the sign of the axis. Both axis-angle and rotation vector representations share the same singularity, and in the latter case, this means that opposed points on the surface of the ball $\{ \mathbf{m}\in \mathbb{R}^3 \, |\, \|\mathbf{m}\| \leq \pi \}$ represent identical rotations.

Observe that the set of rotation vector representations that maps one-to-one to SO(3) is a sort of "half-open, half-closed" ball in $\mathbb{R}^3$. Like the rotation angle representation of a 2D rotation, this set is not topologically equivalent to SO(3) due to the wrapping-around effect.

### 5.4  Inversion[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Inversion)

One of the advantages of axis-angle representation is that inversion is very simple. The inverse of a rotation $(\mathbf{a},\theta)$ is simply $(-\mathbf{a},\theta)$, or, if negative rotations are allowed, $(\mathbf{a},-\theta)$. Equivalently, the inverse of a rotation vector representation is simply its negative $R_m(\mathbf{m})^{-1} = R_m(-\mathbf{m})$.

### 5.5  Geodesic scaling[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Geodesic-scaling)

One of the key advantages of axis-angle representation is that it conveniently represents the geodesic curve from the identity rotation to any given rotation. If $R_1$ is a given rotation matrix with axis-angle form $(\mathbf{a},\theta)$, then the geodesic interpolation from $I$ to $R_1$ is given by the curve

$$R(s) = R_{aa}(\mathbf{a},s\theta)$$

where at $s=0$ the result is the identity, and at $s=1$ the result is $R_1$.

### 5.6  Geodesic interpolation[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Geodesic-interpolation)

We apply the geodesic scaling operation to produce geodesic interpolation between two rotations. Given an initial rotation $R_0$ and a final rotation $R_1$, we may perform the interpolation

$$R(s) = R_0 R_{aa}(\mathbf{a},s\theta)$$

where $(\mathbf{a},\theta)$ is the axis angle representation of the relative rotation $R_0^T R_1$. (Verify that $R(0) = R_0$ and $R(1) = R_1$.)

Because of its common utility, it may be useful to consider representing this procedure as a SO(3) geodesic subroutine $SO3interp(R_0,R_1,s)$.

## 6  Quaternions[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Quaternions)

Quaternions are also another convenient rotation representation based on an extension of complex numbers. Unit complex numbers $c = \cos \theta + i \sin \theta$ with $|c| = \sqrt{a^2 + b^2} = 1$ can represent 2D rotations because complex multiplication performs rotation: $$c (p + i q) = (p \cos \theta - q \sin \theta) + i (p \sin \theta + q \cos \theta)$$ where the real and imaginary components are respectively the $x$ and $y$ component of a point $(p,q)$ rotated by $\theta$. A long search in the 19th century to find analogues of complex numbers that could perform 3D rotations yielded the quaternion representation.

Quaternions $(q_0,q_1,q_2,q_3)$ are composed of 1 real coordinate $q_0$ and three imaginary coordinates $q_1$, $q_2$, and $q_3$ that represent a complex number

$$q = q_0 + i q_1 + j q_2 + k q_3$$

where $i$, $j$, and $k$ are imaginary numbers defined such that

- $i^2 = j^2 = k^2 = -1$
    
- $i = jk = -kj$
    
- $j = ki = -ik$
    
- $k = ij = -ji$.
    

Note that these multiplication rules are non-symmetric. The product of two quaternions is a non-symmetric operation that is defined by applying standard distributive laws and the above rules: \begin{equation} $$
\begin{aligned}q p &= (q_0 + i q_1 + j q_2 + k q_3) (p_0 + i p_1 + j p_2 + k p_3)\\
&= q_0 (p_0 + i p_1 + j p_2 + k p_3) + i q_1 (p_0 + i p_1 + j p_2 + k p_3) + j q_2 (p_0 + i p_1 + j p_2 + k p_3)+ k q_3 (p_0 + i p_1 + j p_2 + k p_3) \\
&= q_0 p_0 + i q_0 p_1 + j q_0 p_2 + k q_0 p_3 + i q_1 p_0 + i^2 q_1 p_1 + i j q_1 p_2 + i k q_1 p_3 + j q_2 p_0 + j i q_2 p_1 + j^2 q_2 p_2 + j k q_2 p_3 + k q_3 p_0 + k i q_3 p_1 + k j q_3 p_2 + k^2 q_3 p_3 \\
&= (q_0 p_0 - q_1 p_1 - q_2 p_2 - q_3 p_3) + i (q_0 p_1 + q_1 p_0 + q_2 p_3 - q_3 p_2) + j (q_0 p_2 + q_2 p_0 - q_1 p_3 + q_3 p_1) + k (q_0 p_3 + q_3 p_0 + q_1 p_2 - q_2 p_1) 
\end{aligned}
$$Norms are defined as usual so that $|q| = \sqrt{q_0^2 + q_1^2 + q_2^2 + q_3^2} = 1$ defines a unit quaternion. Unit quaternions have the special property that their inverse is simply the negation of their imaginary components: $q^{-1} = q_0 - i q_1 - j q_2 - k q_3$$.

Given a 3D point $\mathbf{p} = (x,y,z)$ and its representation as a quaternion $p = (0,x,y,z)$, there is an operation called the _conjugation_ of $p$ by a unit quaternion $q$ so that the result gives a rotated point $p^\prime = (0,x^\prime,y^\prime,z^\prime)$:

$$p^\prime = q p q^{-1}.$$

More specifically, the quaternion is related to axis-angle representation $(\mathbf{a},\theta)$ using the conversion $(q_0,q_1,q_2,q_3) = ( \cos (\theta / 2), \sin (\theta / 2) a_x, \sin (\theta / 2) a_y, \sin (\theta / 2) a_z)$. Also, the rotation matrix defined by a quaternion is given by:

$$R_q(q) = \left[\begin{array}{ccc} 2 (q_0^2 + q_1^2 ) - 1 & 2 (q_1 q_2 - q_0 q_3) & 2 (q_1 q_3 + q_0 q_2) \\ 2(q_1 q_2 + q_0 q_3 ) & 2 (q_0^2 + q_2^2 ) - 1 & 2 (q_2 q_3 - q_0 q_1) \\ 2(q_1 q_2 - q_0 q_3 ) & 2 (q_2 q_3 + q_0 q_1) & 2 (q_0^2 + q_3^2 ) - 1 \end{array}\right].$$

### 6.1  Double cover of 4 sphere[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Double-cover-of-4-sphere)

Quaternions are similar to points on a 4-D unit sphere, but they have the property that a quaternion and its negative produce equivalent rotations. This can be easily seen from the conjugation equation:

$$p^\prime = q p q^{-1} = (-q) p (-q)^{-1}.$$

As a result, the space of rotations could be thought of as a 4-D sphere except warped so that points on the opposite sides of the sphere are mapped to the same rotation. This representation makes certain operations, like interpolation, somewhat more difficult to perform than on a sphere.

### 6.2  Composition[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Composition)

Quaternions have slight advantages over rotation matrices when performing rotation composition because quaternion products perform composition. This operation requires 16 multiplications and 12 additions / subtractions in ($\ref{eq:QuaternionMultiplication}$) as compared to 27 multiplications and 18 additions for standard matrix multiplication.

### 6.3  Geodesic interpolation[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Geodesic-interpolation)

An operation called spherical linear interpolation, or "slerp," can be used to perform geodesic interpolation of rotations using quaternions. More generally it can be used to interpolate across great circles on spheres of arbitrary dimension.

The process is as follows. Given two quaternions $q$ and $p$, the operation first needs to figure out which sign of $p$ to assign in order to choose the closer way around the 4-D sphere. Specifically, if the dot product of $q$ and $p$, $q \cdot p$, is negative, then the sign of each element of $p$ is flipped. (This step is specific to quaternion rotation interpolation, and skipped for general slerp on spheres.)

Next, the angle $\theta$ between the two quaternions is computed according to $\theta = \cos^{-1} (q \cdot p)$. Finally, the interpolated quaternion is given by

$$r(s) = \frac{\sin ((1-s) \theta)}{\sin \theta} q + \frac{\sin (s \theta)}{\sin \theta} p.$$

It can be shown that this is a geodesic and interpolates across the great circle at constant velocity.

## 7  Summary[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Summary)

There is no ideal rotation representation for all purposes, and in some sense, all are equivalent because each representation has an equivalent rotation matrix representation. However, a choice must indeed be made for calculations and coordinate conventions. To assist in this choice, the table below is a "cheat sheet" that summarizes all of the different rotation representations' advantages and disadvantages.

---

Rotation cheat sheet

| **Representation** | **Parameters**                                                                                                                                |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Matrix             | $3\times 3$ matrix $R$ with 9 parameters, with 6 d.o.f. removed via orthogonality constraints.                                                |
| Euler angles       | 3 parameters $(\phi,\theta,\psi)$, in range $[0,2\pi) \times [-\pi/2,\pi/2] \times [0,2\pi)$.                                                 |
| Axis-angle         | 3 + 1 parameters $(\mathbf{a},\theta)$, in range $S_2 \times [0,\pi)$ with 1 d.o.f. removed via unit vector constraint $\|V\mathbf{a}\| = 1$. |
| Rot. vector        | 3 parameters $\mathbf{m}$, in range $\| \mathbf{m} \| \leq \pi$.                                                                              |
| Quaternion         | 4 parameters $(q_0,q_1,q_2,q_3)$, with 1 d.o.f. removed via unit quaternion constraint.                                                       |
|                    | **Singularities and multiple representations**                                                                                                |
| Matrix             | None.                                                                                                                                         |
| Euler angles       | Singularities at $\theta = 0$ or $\theta = \pm \pi/2$, depending on convention.                                                               |
| Axis-angle         | Singularities at $\theta = 0$ (all axes are equivalent) and $\theta = \pi$ (an axis and its negation are equivalent).                         |
| Rot. vector        | Double representation at $\| \mathbf{m} \| = \pi$.                                                                                            |
| Quaternion         | Double representation $R_q(q) = R_q(-q)$ everywhere.                                                                                          |
|                    | **Inversion**                                                                                                                                 |
| Matrix             | $R^T$                                                                                                                                         |
| Euler angles       | $(-\psi,-\theta,-\phi)$ if using an $ABA$ convention. Otherwise, no straightforward formula.                                                  |
| Axis-angle         | $(-\mathbf{a},\theta)$                                                                                                                        |
| Rot. vector        | $-\mathbf{m}$                                                                                                                                 |
| Quaternion         | $(-q_0,q_1,q_2,q_3)$                                                                                                                          |
|                    | **Composition**                                                                                                                               |
| Matrix             | matrix product $R_1 R_2$.                                                                                                                     |
| Euler angles       | no straightforward formula.                                                                                                                   |
| Axis-angle         | no straightforward formula.                                                                                                                   |
| Rot. vector        | no straightforward formula.                                                                                                                   |
| Quaternion         | quaternion multiplication.                                                                                                                    |

---

Interpolation for all representations can be performed by converting to matrix form and performing geodesic interpolation using the axis-angle method. Quaternions have a special interpolation method using spherical linear interpolation (with care taken to pick closest dual representation on 4-sphere).

### 7.1  Key takeaways[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Key-takeaways)

- 2D and 3D rotations are important for modelling rigid body movement.
    
- Rotation has fundamentally different topology from Cartesian space. Interpolation and distance is not the same as treating rotation representations as points in Cartesian space.
    
- The four major representations of 3D rotations are rotation matrix, Euler angle (e.g., roll-pitch-yaw), axis-angle (which is very similar to the rotation vector representation), and quaternion.
    
- All representations are somewhat equivalent in that they can be converted to a rotation matrix and back again. But each has some strengths and weaknesses.
    
- Robotics and 3D visualisation software packages should contain functions for converting between many of these representations.
    

### 7.2  Glossary[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Glossary)

- [Topology](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#topology): a mathematical structure of a set defining the way that points in the set are connected.
- [Euler angles](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Euler-angles): A 3 parameter representation of 3D rotations indicating a composition of 3 axis-aligned rotations.
- [Roll-pitch-yaw representation](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#roll-pitch-yaw): An Euler angle convention that represents subsequent rotations along the X (roll), Y (pitch), and Z (yaw) axes.
- [Axis-angle representation](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#axis-angle): A 4 parameter representation of 3D rotations that includes an axis and an angle of rotation about that axis.
- [Rotation vector / Rodriguez vector / expontential map representation](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#rotation-vector): A 3-parameter representation closely related to axis-angle representation that scales the axis by the angle.
- [Quaternions](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Quaternions): A 4 parameter representation of 3D rotations that is related to complex numbers. Unit quaternions represent 3D rotations.
- [Geodesic distance](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Geodesic-interpolation): The minimal angle needed to pass from one rotation to another. Given rotation matrices $R_A$ and $R_B$, this is given by the angle of the axis-angle representation of $R_A^T R_B$.
- [Geodesic interpolation](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Geodesic-interpolation): A curve that traces the minimal-length path between two rotation matrices.

## 8  Exercises[](https://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Exercises)

1. Derive a method to compute the "average" of two angles in SO(2). Hint: consider the interpolation formula. For what pairs of angles is the notion of average ill-defined?
    
2. Is SO(3) equivalent to SO(2) $\times$ SO(2) $\times$ SO(2)? Why or why not?
    
3. Explain why the 3D rotation about the Y axis has the signs of its off-diagonal $\sin \theta$ terms in the opposite orientation as the X and Z rotations.
    
4. Give an example of an Euler angle representation for which direct interpolation produces a path of rotations that is very unlike a geodesic in SO(3).
    
5. Symbolically, derive the function that maps a ZYZ Euler angle representation to a $3\times 3$ rotation matrix. Now, compute its inverse (that is, a procedure for mapping a rotation matrix to a ZYZ Euler angle representation). Where are the singularities of this representation?
    
6. Derive a method to compute the average of two 3D rotation matrices. (Use the hint given in problem 1.)
    
7. For the cross-product matrix operator $\hat{\mathbf{v}}$ given by $\eqref{eq:CrossProductMatrix}$, show that $R \hat{\mathbf{v}} R^T = \hat{(R\mathbf{v})}$ (here the hat-operator is applied to the entire vector $R\mathbf{v}$.
    
8. Consider a distance function $d(R_A,R_B)$ between two matrices, defined as the absolute angle of $R_A^T R_B$ (as given by ($\ref{eq:RotationAbsoluteAngle}$)). Prove that it satisfies three conditions needed for being a metric: $d(R,R)=0$ iff $R=R$, $d(R_A,R_B)\geq 0$, $d(R_A,R_B)=d(R_B,R_A)$. Do you think it satisfies the triangle inequality $d(R_A,R_B) \leq d(R_A,R_C) + d(R_C,R_B)$ for all $R_C$? Why or why not? (Hint: consider what it means to be a geodesic.)