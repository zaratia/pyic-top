from numba import njit
import numpy as np

@njit
def MC_params(cel, disp, dx, dt, n_reach):

    c1_mc = np.empty(n_reach)
    c2_mc = np.empty(n_reach)
    c3_mc = np.empty(n_reach)
    k_MC = 0.0
    x_MC = 0.0

    for i in range(n_reach):
        k_MC = dx[i] / cel[i]
        x_MC = 0.5 - disp[i] / ( cel[i]*dx[i] )
        c1_mc[i] = (dt - 2. * k_MC * x_MC) / (2 * k_MC * (1 - x_MC) + dt)
        c2_mc[i] = (dt + 2. * k_MC * x_MC) / (2 * k_MC * (1 - x_MC) + dt )
        c3_mc[i] = (2 * k_MC * (1 - x_MC) - dt) / (2 * k_MC * (1 - x_MC) + dt)
    
    return c1_mc, c2_mc, c3_mc


#@njit
def MC(
    current_reach, Qupstream,
    nstep, Qsubreach_in, Qsubreach_out,
    c1_mc, c2_mc, c3_mc
):
    Qsubreach_in_t1 = Qsubreach_in[:, current_reach].copy()
    Qsubreach_out_t1 = Qsubreach_out[:, current_reach].copy()

    for n in range(nstep):
        if n == 0:
            Qsubreach_out[n, current_reach] = (
                Qupstream * c1_mc
                + Qsubreach_in_t1[n] * c2_mc
                + Qsubreach_out_t1[n] * c3_mc
            )
        else:
            Qsubreach_in[n - 1, current_reach] = Qsubreach_out[n - 1, current_reach]
            Qsubreach_in_t1[n - 1] = Qsubreach_out_t1[n - 1]
            Qsubreach_out[n, current_reach] = (
                Qsubreach_in[n - 1, current_reach] * c1_mc
                + Qsubreach_in_t1[n - 1] * c2_mc
                + Qsubreach_out_t1[n] * c3_mc
            )

    return Qsubreach_in, Qsubreach_out