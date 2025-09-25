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
    nstep, Qsubreach_in, Qsubreach_out,
    c1_mc, c2_mc, c3_mc
):
    Qsubreach_in_t1 = Qsubreach_in.copy()
    Qsubreach_out_t1 = Qsubreach_out.copy()

    for n in range(nstep):
        if n == 0:
            Qsubreach_out[n] = (
                Qupstream * c1_mc
                + Qsubreach_in_t1[n] * c2_mc
                + Qsubreach_out_t1[n] * c3_mc
            )
        else:
            Qsubreach_in[n - 1] = Qsubreach_out[n - 1]
            Qsubreach_in_t1[n - 1] = Qsubreach_out_t1[n - 1]
            Qsubreach_out[n] = (
                Qsubreach_in[n - 1] * c1_mc
                + Qsubreach_in_t1[n - 1] * c2_mc
                + Qsubreach_out_t1[n] * c3_mc
            )

    Q_mc[ :n_step, time_step] = Qsubreach_out[:n_step]