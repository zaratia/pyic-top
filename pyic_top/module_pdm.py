import numpy as np
from numba import njit


@njit
def soil_water_content(cstar, min_c, max_c, current_b):
    Cmed = (current_b * min_c + max_c) / (current_b + 1.0)
    if max_c <= cstar:
        return min_c + (Cmed - min_c)
    else:
        return min_c + (Cmed - min_c) * (
            1.0 - ((max_c - cstar) / (max_c - min_c)) ** (current_b + 1.0)
        )


@njit
def runoff_function(eff_rain, cstar_t1, cstar, min_c, max_c, current_b, dt):
    Cmed = (current_b * min_c + max_c) / (current_b + 1.0)
    if eff_rain >= 0.0:
        runoff = eff_rain * dt - (Cmed - min_c) * (
            ((max_c - cstar_t1) / (max_c - min_c)) ** (current_b + 1.0)
            - ((max_c - cstar) / (max_c - min_c)) ** (current_b + 1.0)
        )
        return max(runoff, 0.0)
    else:
        return 0.0


@njit
def runoff_propagation(dt, t_k1, t_k2):
    dt1_tmp = np.exp(-dt / t_k1)
    dt2_tmp = np.exp(-dt / t_k2)
    dt1 = -(dt1_tmp + dt2_tmp)
    dt2 = dt1_tmp * dt2_tmp

    if t_k1 == t_k2:
        o0 = 1.0 - (1.0 + 1.0 / t_k1) * dt1_tmp
        o1 = (dt1_tmp - 1.0 + 1.0 / t_k1) * dt1_tmp
    else:
        o0 = (t_k1 * (dt1_tmp - 1.0) - t_k2 * dt2_tmp) / (t_k2 - t_k1)
        o1 = (t_k2 * (dt2_tmp - 1.0) * dt1_tmp - t_k1 * (dt1_tmp - 1.0) * dt2_tmp) / (
            t_k2 - t_k1
        )
    return dt1, dt2, o0, o1


@njit
def glacier_runoff_propagation(dt, t_kglac):
    if t_kglac > 0.0:
        dt1_tmp = np.exp(-dt / t_kglac)
        dt2_tmp = np.exp(-dt / t_kglac)
        dt1 = -(dt1_tmp + dt2_tmp)
        dt2 = dt1_tmp * dt2_tmp
        o0 = 1.0 - (1.0 + 1.0 / t_kglac) * dt1_tmp
        o1 = (dt1_tmp - 1.0 + 1.0 / t_kglac) * dt1_tmp
    else:
        dt1 = dt2 = o0 = o1 = 0.0
    return dt1, dt2, o0, o1


@njit
def Csat(S, cmin0, cmax0, b0):
    Smax = (b0 * cmin0 + cmax0) / (b0 + 1.0)
    Cmed = Smax
    if Smax <= S:
        return cmin0 + (cmax0 - cmin0)
    else:
        return cmin0 + (cmax0 - cmin0) * (
            1.0 - ((Smax - S) / (Cmed - cmin0)) ** (1.0 / (b0 + 1.0))
        )


# -------------------------------------------------
# Initialize state variables from previous time step
# -------------------------------------------------
@njit
def init_PDM_state_var_previous_time(
    current_basin,
    tt,
    stg,
    sgw,
    Cstar,
    runoff,
    runoff_glac,
    Qbase,
    Qrunoff_t1,
    Qglac_t1,  # (n_basin)
    Qrunoff_t2,
    Qglac_t2,  # (n_basin)
    Qrunoff,
    Qglac,  # (n_basin, n_hours + 1)
):
    if tt == 1:
        stg_t1 = stg[current_basin]
        sgw_t1 = sgw[current_basin]
        Cstar_t1 = Cstar[current_basin]
        runoff_t1 = runoff[current_basin]
        runoff_glac_t1 = runoff_glac[current_basin]
        Qbase_t1 = Qbase[current_basin, 0]
    else:
        stg_t1 = stg[current_basin]
        sgw_t1 = sgw[current_basin]
        Cstar_t1 = Cstar[current_basin]
        runoff_t1 = runoff[current_basin]
        Qrunoff_t2[current_basin] = Qrunoff_t1[current_basin]
        Qrunoff_t1[current_basin] = Qrunoff[current_basin, tt - 1]
        Qglac_t2[current_basin] = Qglac_t1[current_basin]
        Qglac_t1[current_basin] = Qglac[current_basin, tt - 1]
        runoff_glac_t1 = runoff_glac[current_basin]
        Qbase_t1 = Qbase[current_basin, tt - 1]

    return (
        stg_t1,
        sgw_t1,
        Cstar_t1,
        runoff_t1,
        Qrunoff_t1,
        Qrunoff_t2,
        Qglac_t1,
        Qglac_t2,
        runoff_glac_t1,
        Qbase_t1,
    )


# -------------------------------------------------
# Main PDM routine
# -------------------------------------------------
@njit
def pdm(
    n_hours,
    n_basin,
    month_array,
    hour_array,
    year_array,
    day_array,
    stg,
    sgw,
    Cstar,
    runoff,
    runoff_glac,
    Qbase,
    PET,
    ET,
    baseflow,
    baseflow_glac,
    Qbase_type,
    cmin,
    cmax,
    stmax,
    stmin,
    b,
    be,
    bg,
    q0,
    m,
    kg,
    ks,
    k1,
    k2,
    k_glac,
    Qglac_t1,
    Qglac_t2,
    Qrunoff_t1,
    Qrunoff_t2,
    Qrunoff,
    Qglac,
    Qsubsurf,
    soil_moisture,
    time_step_duration,
):
    # init soil moisture
    soil_moisture[:, 0] = stg / stmax

    for time_step in range(1, n_hours + 1, 1):
        # print time info
        if hour_array[time_step - 1] == 0:
            print(
                "PDM",
                year_array[time_step - 1],
                month_array[time_step - 1],
                day_array[time_step - 1],
            )
        # loop over basins
        for current_basin in range(n_basin):
            (
                stg_t1,
                sgw_t1,
                Cstar_t1,
                runoff_t1,
                Qrunoff_t1,
                Qrunoff_t2,
                Qglac_t1,
                Qglac_t2,
                runoff_glac_t1,
                Qbase_t1,
            ) = init_PDM_state_var_previous_time(
                current_basin,
                time_step,
                stg,
                sgw,
                Cstar,
                runoff,
                runoff_glac,
                Qbase,
                Qrunoff_t1,
                Qglac_t1,
                Qrunoff_t2,
                Qglac_t2,
                Qrunoff,
                Qglac,
            )
            # calc ET
            ET[current_basin, time_step] = (
                PET[current_basin, time_step]
                * (
                    1.0
                    - ((stmax[current_basin] - stg_t1) / stmax[current_basin])
                    ** be[current_basin]
                )
                if stg_t1 > cmin[current_basin]
                else 0.0
            )
            # grounwater recharge
            Q_to_gw = (
                (
                    (1.0 / kg[current_basin])
                    * (stg_t1 - stmin[current_basin]) ** bg[current_basin]
                )
                if stg_t1 > stmin[current_basin]
                else 0.0
            )
            # topmodel like subsurface flow
            Qsubsurf[current_basin, time_step] = (
                q0[current_basin]
                * np.exp(-(stmax[current_basin] - stg_t1) / m[current_basin])
                if q0[current_basin] > 0.001 or m[current_basin] > 1.0
                else 0.0
            )
            # get the balance. the baseflow start from timestep 1, 0 is IC
            losses = Qsubsurf[current_basin, time_step] + Q_to_gw + ET[current_basin, time_step]
            net_inflow = baseflow[current_basin, time_step - 1] - losses

            runoff_glac[current_basin] = baseflow_glac[current_basin, time_step - 1]
            dt1_glac, dt2_glac, o0_glac, o1_glac = glacier_runoff_propagation(
                time_step_duration, k_glac[current_basin]
            )
            Qglac[current_basin, time_step] = (
                -dt1_glac * Qglac_t1[current_basin]
                - dt2_glac * Qglac_t2[current_basin]
                + o0_glac * runoff_glac[current_basin]
                + o1_glac * runoff_glac_t1
            )

            if net_inflow > 0.0:
                # update cstar
                Cstar[current_basin] = min(
                    Cstar_t1 + net_inflow, cmax[current_basin]
                    )
                # update storage
                stg[current_basin] = min(
                    soil_water_content(
                        Cstar[current_basin],
                        cmin[current_basin],
                        cmax[current_basin],
                        b[current_basin],
                    ),
                    stmax[current_basin],
                )
                # compute runoff
                runoff[current_basin] = runoff_function(
                    net_inflow,
                    Cstar_t1,
                    Cstar[current_basin],
                    cmin[current_basin],
                    cmax[current_basin],
                    b[current_basin],
                    time_step_duration,
                )

                # propagate runoff
                dt1, dt2, o0, o1 = runoff_propagation(
                    time_step_duration, k1[current_basin], k2[current_basin]
                )
                Qrunoff[current_basin, time_step] = (
                    -dt1 * Qrunoff_t1[current_basin]
                    - dt2 * Qrunoff_t2[current_basin]
                    + o0 * runoff[current_basin]
                    + o1 * runoff_t1
                )

            else:
                runoff[current_basin] = 0.0
                # propagate runoff
                dt1, dt2, o0, o1 = runoff_propagation(
                    time_step_duration, k1[current_basin], k2[current_basin]
                )
                Qrunoff[current_basin, time_step] = (
                    -dt1 * Qrunoff_t1[current_basin]
                    - dt2 * Qrunoff_t2[current_basin]
                    + o0 * runoff[current_basin]
                    + o1 * runoff_t1
                )
                # update storage
                stg[current_basin] = stg[current_basin] + net_inflow
                if stg[current_basin] < cmin[current_basin]:
                    Cstar[current_basin] = cmin[current_basin]
                    stg[current_basin] = cmin[current_basin]
                    # update precipitation and losses
                    net_inflow = stg[current_basin] - stg_t1
                    losses = baseflow[current_basin, time_step] - net_inflow
                    if Q_to_gw >= losses:
                        Q_to_gw = losses
                        Qsubsurf[current_basin, time_step] = 0.0
                        ET[current_basin, time_step] = 0.0
                    else:
                        losses = losses - Q_to_gw
                        if Qsubsurf[current_basin, time_step] >= losses:
                            Qsubsurf[current_basin, time_step] = losses
                            ET[current_basin, time_step] = 0.0
                        else:
                            losses = losses - Qsubsurf[current_basin, time_step]
                            ET[current_basin, time_step] = losses


                if stg[current_basin] >= stmax[current_basin]:
                    stg[current_basin] = stmax[current_basin]

                Cstar[current_basin] = Csat(
                    stg[current_basin],
                    cmin[current_basin],
                    cmax[current_basin],
                    b[current_basin],
                )
                if Cstar[current_basin] > cmax[current_basin]:
                    Cstar[current_basin] = cmax[current_basin]
                    stg[current_basin] = soil_water_content(
                        Cstar[current_basin],
                        cmin[current_basin],
                        cmax[current_basin],
                        b[current_basin],
                    )

            # update soil moisture
            soil_moisture[current_basin, time_step] = (
                stg[current_basin] / stmax[current_basin]
            )
            
            # if Qrunoff[current_basin,time_step] < 0.00001:
            #     Qrunoff[current_basin,time_step] = 0.00001

            # Baseflow
            if Qbase_type == 0:
                Qbase[current_basin, time_step] = (
                    Qbase_t1 / (1.0 + Qbase_t1 * time_step_duration / ks[current_basin])
                    if Q_to_gw == 0.0
                    else Qbase_t1
                    / (
                        np.exp(-Q_to_gw * time_step_duration / ks[current_basin])
                        + (Qbase_t1 / Q_to_gw)
                        * (
                            1.0
                            - np.exp(-Q_to_gw * time_step_duration / ks[current_basin])
                        )
                    )
                )
                sgw[current_basin] = (
                    np.log(Qbase[current_basin, time_step]) / ks[current_basin]
                )
            elif Qbase_type == 1:
                if Q_to_gw == 0.0:
                    Qbase[current_basin, time_step] = (
                        Qbase_t1 ** (-2.0 / 3.0)
                        + 2.0 * time_step_duration * ks[current_basin] ** (1.0 / 3.0)
                    ) ** (-3.0 / 2.0)
                else:
                    tmp = sgw_t1 - (1.0 / (3.0 * ks[current_basin] * sgw_t1**2.0)) * (
                        np.exp(
                            -3.0 * ks[current_basin] * time_step_duration * sgw_t1**2
                        )
                        - 1.0
                    ) * (Q_to_gw - ks[current_basin] * sgw_t1**3.0)
                    Qbase[current_basin, time_step] = ks[current_basin] * tmp**3.0
                sgw[current_basin] = (
                    Qbase[current_basin, time_step] / ks[current_basin]
                ) ** (1.0 / 3.0)
            elif Qbase_type == 2:
                Qbase[current_basin, time_step] = (
                    (Qbase_t1 * np.exp(-time_step_duration * ks[current_basin]))
                    if Q_to_gw == 0.0
                    else (
                        Qbase_t1 * np.exp(-time_step_duration * ks[current_basin])
                        + Q_to_gw
                        * (1.0 - np.exp(-time_step_duration * ks[current_basin]))
                    )
                )
                sgw[current_basin] = Qbase[current_basin, time_step] / ks[current_basin]

    return (
        stg,
        sgw,
        Cstar,
        ET,
        Qsubsurf,
        runoff_glac,
        Qglac,
        runoff,
        Qrunoff,
        soil_moisture,
        Qbase,
    )
