
import jax
import jax.numpy as jnp
from jax.experimental import jet

def debug():
    def f(x):
        return x[0]**2
    
    x0 = jnp.array([0.0])
    
    # 1st order
    print("Order 1:")
    primals, series = jet.jet(f, (x0,), ((jnp.array([1.0]),),))
    print("Primals:", primals)
    print("Series:", series)
    
    # 2nd order
    print("\nOrder 2:")
    # series needs 2 terms
    # V1 = 1, V2 = 0
    primals, series = jet.jet(f, (x0,), ((jnp.array([1.0]), jnp.array([0.0])),))
    print("Primals:", primals)
    print("Series:", series) # Expect [0, 1] if Taylor, [0, 2] if Derivs

if __name__ == "__main__":
    debug()
