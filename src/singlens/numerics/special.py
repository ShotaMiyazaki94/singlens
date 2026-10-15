"""JAX special functions adapted from microJAX; see NOTICE for provenance.

Modified in singlens for scalar/array inputs and finite derivatives at zero.
The Bessel rational approximations derive from JAX PR #17038 and Cephes.
"""

import jax.numpy as jnp
from jax import custom_jvp, jit

__all__ = ["j0", "j1", "j2"]


# I imported the code from https://github.com/google/jax/pull/17038/files
# polynomial coefficients for J1

RP1 = [
    -8.99971225705559398224e8,
    4.52228297998194034323e11,
    -7.27494245221818276015e13,
    3.68295732863852883286e15,
]
RQ1 = [
    1.0,
    6.20836478118054335476e2,
    2.56987256757748830383e5,
    8.35146791431949253037e7,
    2.21511595479792499675e10,
    4.74914122079991414898e12,
    7.84369607876235854894e14,
    8.95222336184627338078e16,
    5.32278620332680085395e18,
]

PP1 = [
    7.62125616208173112003e-4,
    7.31397056940917570436e-2,
    1.12719608129684925192e0,
    5.11207951146807644818e0,
    8.42404590141772420927e0,
    5.21451598682361504063e0,
    1.00000000000000000254e0,
]
PQ1 = [
    5.71323128072548699714e-4,
    6.88455908754495404082e-2,
    1.10514232634061696926e0,
    5.07386386128601488557e0,
    8.39985554327604159757e0,
    5.20982848682361821619e0,
    9.99999999999999997461e-1,
]

QP1 = [
    5.10862594750176621635e-2,
    4.98213872951233449420e0,
    7.58238284132545283818e1,
    3.66779609360150777800e2,
    7.10856304998926107277e2,
    5.97489612400613639965e2,
    2.11688757100572135698e2,
    2.52070205858023719784e1,
]
QQ1 = [
    1.0,
    7.42373277035675149943e1,
    1.05644886038262816351e3,
    4.98641058337653607651e3,
    9.56231892404756170795e3,
    7.99704160447350683650e3,
    2.82619278517639096600e3,
    3.36093607810698293419e2,
]

Z1 = 1.46819706421238932572e1
Z2 = 4.92184563216946036703e1
PIO4 = 0.78539816339744830962  # pi/4
THPIO4 = 2.35619449019234492885  # 3*pi/4
SQ2OPI = 0.79788456080286535588  # sqrt(2/pi)

# polynomial coefficients for J0

PP0 = [
    7.96936729297347051624e-4,
    8.28352392107440799803e-2,
    1.23953371646414299388e0,
    5.44725003058768775090e0,
    8.74716500199817011941e0,
    5.30324038235394892183e0,
    9.99999999999999997821e-1,
]
PQ0 = [
    9.24408810558863637013e-4,
    8.56288474354474431428e-2,
    1.25352743901058953537e0,
    5.47097740330417105182e0,
    8.76190883237069594232e0,
    5.30605288235394617618e0,
    1.00000000000000000218e0,
]

QP0 = [
    -1.13663838898469149931e-2,
    -1.28252718670509318512e0,
    -1.95539544257735972385e1,
    -9.32060152123768231369e1,
    -1.77681167980488050595e2,
    -1.47077505154951170175e2,
    -5.14105326766599330220e1,
    -6.05014350600728481186e0,
]
QQ0 = [
    1.0,
    6.43178256118178023184e1,
    8.56430025976980587198e2,
    3.88240183605401609683e3,
    7.24046774195652478189e3,
    5.93072701187316984827e3,
    2.06209331660327847417e3,
    2.42005740240291393179e2,
]

DR10 = 5.78318596294678452118e0
DR20 = 3.04712623436620863991e1

RP0 = [
    -4.79443220978201773821e9,
    1.95617491946556577543e12,
    -2.49248344360967716204e14,
    9.70862251047306323952e15,
]
RQ0 = [
    1.0,
    4.99563147152651017219e2,
    1.73785401676374683123e5,
    4.84409658339962045305e7,
    1.11855537045356834862e10,
    2.11277520115489217587e12,
    3.10518229857422583814e14,
    3.18121955943204943306e16,
    1.71086294081043136091e18,
]


def j0_small(x):
    """
    Implementation of J0 for x < 5
    """
    z = x * x
    p = (z - jnp.array(DR10)) * (z - jnp.array(DR20))
    p = p * jnp.polyval(jnp.array(RP0), z) / jnp.polyval(jnp.array(RQ0), z)
    return jnp.where(x < 1e-5, 1 - z / 4.0, p)


def j0_large(x):
    """
    Implementation of J0 for x >= 5
    """

    w = 5.0 / x
    q = 25.0 / (x * x)
    p = jnp.polyval(jnp.array(PP0), q) / jnp.polyval(jnp.array(PQ0), q)
    q = jnp.polyval(jnp.array(QP0), q) / jnp.polyval(jnp.array(QQ0), q)
    xn = x - PIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    return p * SQ2OPI / jnp.sqrt(x)


@custom_jvp
def j0_(z):
    """
    Bessel function of the first kind of order zero and a real argument
    - using the implementation from CEPHES, translated to Jax, to match scipy to machine precision.
    Reference:
    Cephes Mathematical Library.
    Args:
        z: The sampling point(s) at which the Bessel function of the first kind are
        computed.
    Returns:
        An array of shape `x.shape` containing the values of the Bessel function
    """
    z = jnp.array(z)
    return jnp.where(jnp.abs(z) < 5.0, j0_small(jnp.abs(z)), j0_large(jnp.maximum(jnp.abs(z), 5.0)))


@j0_.defjvp
def j0_jvp(primals, tangents):
    (m,) = primals
    (m_dot,) = tangents
    return j0_(m), -1.0 * m_dot * j1_(m)


def j1_small(x):
    """
    Implementation of J1 for x < 5
    """
    z = x * x
    w = jnp.polyval(jnp.array(RP1), z) / jnp.polyval(jnp.array(RQ1), z)
    w = w * x * (z - jnp.array(Z1)) * (z - jnp.array(Z2))
    return w


def j1_large(x):
    """
    Implementation of J1 for x > 5
    """
    w = 5.0 / x
    z = w * w
    p = jnp.polyval(jnp.array(PP1), z) / jnp.polyval(jnp.array(PQ1), z)
    q = jnp.polyval(jnp.array(QP1), z) / jnp.polyval(jnp.array(QQ1), z)
    xn = x - THPIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    return p * SQ2OPI / jnp.sqrt(x)


@custom_jvp
def j1_(z):
    """
    Bessel function of the first kind of order one and a real argument
    - using the implementation from CEPHES, translated to Jax, to match scipy to machine precision.
    Reference:
    Cephes mathematical library.
    Args:
        x: The sampling point(s) at which the Bessel function of the first kind are
        computed.
    Returns:
        An array of shape `x.shape` containing the values of the Bessel function
    """
    z = jnp.array(z)

    return jnp.sign(z) * jnp.where(
        jnp.abs(z) < 5.0, j1_small(jnp.abs(z)), j1_large(jnp.maximum(jnp.abs(z), 5.0))
    )


@j1_.defjvp
def j1_jvp(primals, tangents):
    (m,) = primals
    (m_dot,) = tangents
    primal_out = j1_(m)
    safe = jnp.where(m == 0, 1.0, m)
    tangent_out = m_dot * jnp.where(m == 0, 0.5, j0_(m) - j1_(m) / safe)
    return primal_out, tangent_out


j0 = jit(j0_)
j1 = jit(j1_)


def j2_(x):
    x = jnp.asarray(x)
    safe = jnp.where(jnp.abs(x) < 0.1, 1.0, x)
    series = x * x / 8 - x**4 / 96 + x**6 / 3072 - x**8 / 184320
    return jnp.where(jnp.abs(x) < 0.1, series, 2 * j1_(safe) / safe - j0_(safe))


j2 = jit(j2_)
