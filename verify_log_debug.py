
import jax
import jax.numpy as jnp
from jax.experimental import jet
from immrax.generator.sets.taylor_model import TaylorModel
from immrax.inclusion import interval
from immrax.inclusion.tm import nattm, _tm_univariate, _tm_constant

def debug_log():
    print("Debugging Log TM Construction...")
    
    # Setup x = 1.0 + u, u in [-0.1, 0.1] => x in [0.9, 1.1]
    center = 1.0
    radius = 0.1
    order = 2
    d = 1
    
    # TM for x: const=1.0, linear_coeff=0.1 (since u in [-1,1] maps to [-r, r])
    from immrax.generator.sets.taylor_model import _get_canonical_exponents
    exponents = _get_canonical_exponents(d, order)
    coeffs = jnp.zeros((1, exponents.shape[1]))
    coeffs = coeffs.at[:, 0].set(center)
    coeffs = coeffs.at[:, 1].set(radius)
    
    x_tm = TaylorModel(coeffs, exponents, interval(jnp.array([0.]), jnp.array([0.])), 
                       jnp.array([center]), jnp.array([radius]), _static_order=order)
                       
    print(f"Input TM x: coeffs={x_tm.coeffs}")
    
    # 1. Check jet coefficients for log around c=1.0
    c = jnp.array([1.0])
    unit_series = (1.0,) + (0.0,) * (order - 1)
    prim, series = jet.jet(lambda v: jnp.log(v), (c[0],), (unit_series,))
    print(f"Jet output at c=1.0: prim={prim}, series={series}")
    # Expected: log(1)=0, 1/1=1, -1/1^2 = -1 (unscaled?)
    # Jet series out: f'(c), f''(c)/2 ...
    # Series should be [1.0, -0.5] ? 
    
    # 2. Trace _tm_univariate logic manually
    z = x_tm - c
    print(f"z coeffs (x-c): {z.coeffs}")
    
    coeffs_raw = jnp.concatenate([jnp.array([prim]), jnp.array(series)])
    print(f"Poly coeffs raw: {coeffs_raw}")
    
    # P(z) construction
    result = _tm_constant(jnp.array([coeffs_raw[order]]), d, order, jnp.array([center]), jnp.array([radius]))
    print(f"Init result (highest order {order}): {result.coeffs}")
    
    for k in range(order - 1, -1, -1):
        term_k = _tm_constant(jnp.array([coeffs_raw[k]]), d, order, jnp.array([center]), jnp.array([radius]))
        if k == 0:
            result = result.multiply(z, max_order=order) + term_k
        else:
            result = result.multiply(z, max_order=order) + term_k
        print(f"Step k={k}, result coeffs: {result.coeffs}")
            
    # Check evaluation at u=0 (x=1)
    val_at_center = result.evaluate_polynomial(jnp.array([1.0]))
    print(f"Eval at center (should be 0): {val_at_center}")
    
    # Check evaluation at u=1 (x=1.1)
    # log(1.1) approx 0.0953
    val_at_edge = result.evaluate_polynomial(jnp.array([1.1]))
    print(f"Eval at edge (should be ~0.0953): {val_at_edge}")

if __name__ == "__main__":
    debug_log()
