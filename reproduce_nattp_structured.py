import jax
import jax.numpy as jnp
from immrax.taylor.taylor_polynomial import taylor_polynomial_identity, TaylorPolynomial
from immrax.taylor.nattp import nattp
from immrax.inclusion import interval


def reproduction():
    # Define a function taking two arguments
    def f(x, y):
        return x * y + x

    # Create a structured domain (tuple to match f(x, y))
    int_x = interval(0.0, 1.0)
    int_y = interval(2.0, 3.0)
    domain = (int_x, int_y)

    # Create an identity TP for this structured domain
    tp = taylor_polynomial_identity(domain, order=2)

    # Use nattp with structured_center=True
    # The function f expects (x, y), but we pass the single structured TP
    # nattp should unpack it based on the domain structure

    # We need to wrap f to match the unpacked signature if we were calling it normally?
    # No, nattp traces f using the flattened args.
    # When we call the wrapped function, we pass the single TP.
    # But wait, nattp traces f(*args).
    # If structured_center is True, nattp expects the Input to be a single TP.
    # But internally it traces f with UNPACKED args.
    # So f should expect the unpacked args.
    # The domain {'x': ..., 'y': ...} flattens to [x, y] in alphabetic order usually or standard pytree order.
    # Let's verify standard pytree behavior.

    flattened, treedef = jax.tree_util.tree_flatten(domain)
    # flattened has 2 elements.
    # f needs to accept 2 arguments.

    # Let's define f to take 2 args
    def f_flat(x, y):
        return x * y

    f_tp = nattp(f_flat, structured_center=True)

    # Call with the single structured TP
    res = f_tp(tp)

    print("Result type:", type(res))
    print("Result shape:", res.shape)
    print("Result coeffs shape:", res.coeffs.shape)

    # Check evaluation
    pt_x = 0.5
    pt_y = 2.5
    val_true = f_flat(pt_x, pt_y)

    # Evaluate TP at the point (need to form flat point matching domain)
    # The TP expects a flat point for the whole domain
    pt_flat = jnp.array([pt_x, pt_y])
    val_tp = res.evaluate(pt_flat)

    print(f"True val: {val_true}")
    print(f"TP val: {val_tp}")

    assert jnp.allclose(val_true, val_tp)
    print("Verification Successful!")


if __name__ == "__main__":
    reproduction()
