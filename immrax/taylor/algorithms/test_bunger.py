import jax
import jax.numpy as jnp
import immrax as irx

def f(x) :
    return jnp.array([-x[1], x[0]])

ix0 = irx.icentpert(jnp.ones(2), 0.1)
tm0 = irx.taylor_model_identity(ix0, order=3)

f_tm = irx.nattm(f)
print(f_tm(tm0))

integ_tm0 = irx.integrate_variable(tm0, 0)
print(integ_tm0.coeffs, integ_tm0.exponents)

def K (tm) :
    # Picard Operator
    # Integrate tm from t_0 to t

    tm_intt = irx.integrate_variable(tm, )

