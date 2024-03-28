from typing import Callable, Any
from jaxtyping import Array, Float, PyTree, jaxtyped
from typeguard import typechecked as typechecker

# Nd arrays
FloatArrayN1 = Float[Array, "N 1"]
FloatArrayN2 = Float[Array, "N 2"]
FloatArrayN3 = Float[Array, "N 3"]
FloatArrayNd = Float[Array, "N d"]
PointType = Float[Array, "d"]