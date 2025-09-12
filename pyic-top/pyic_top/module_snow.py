import numpy as np
from numba import njit


# ---------------------------------------------
# MODULE: gequivalent to fortran NINT
# ---------------------------------------------
@njit
def nint(x):
    return np.floor(x + 0.5) if x >= 0 else np.ceil(x - 0.5)


# ---------------------------------------------
# MODULE: get_EI
# ---------------------------------------------
@njit
def get_EI(
    elev_bnd, energy_bnd, current_basin, month, curr_hour, sunrise, sunset, EI, EImin
):
    isday_check = False
    if sunrise[month] <= curr_hour <= sunset[month]:
        isday_check = True
        day_duration = 24.0 / (sunset[month] - sunrise[month])
        EI_current = EI[current_basin, month, elev_bnd, energy_bnd] * day_duration
    else:
        EI_current = EImin
    if EI_current < 0.0:
        EI_current = 0.0
    return EI_current, isday_check


# ---------------------------------------------
# MODULE: get_SCA
# ---------------------------------------------
@njit
def get_SCA(
    current_time, current_WE, WE_threshold, bande_area, SCA, basin, elev_bnd, energy_bnd
):
    if current_WE >= WE_threshold:
        SCA[current_time] += bande_area[basin, elev_bnd, energy_bnd]


# ---------------------------------------------
# MODULE: Redistribute_WE
# ---------------------------------------------
@njit
def Redistribute_WE(
    n_basins,
    n_fasce,
    n_bande,
    fasce_area,
    fasce_elev_avg,
    weth,
    lowerh,
    upperh,
    bande_area,
    basin_area,
    WE,
):
    for bb in range(n_basins):
        we_to_red = 0.0
        area_from_red = 0.0
        area_to_red = 0.0
        for i in range(n_fasce):
            if fasce_area[bb, i] > 0 and fasce_elev_avg[bb, i] >= upperh[bb]:
                for j in range(n_bande):
                    if WE[bb, i, j] >= weth[bb] and bande_area[bb, i, j] > 0.0:
                        we_to_red += (
                            (WE[bb, i, j] - weth[bb])
                            * bande_area[bb, i, j]
                            / basin_area[bb]
                        )
                        WE[bb, i, j] = weth[bb]
                        area_from_red += bande_area[bb, i, j]
        for i in range(n_fasce):
            if (
                fasce_area[bb, i] > 0
                and lowerh[bb] <= fasce_elev_avg[bb, i] < upperh[bb]
            ):
                for j in range(n_bande):
                    if bande_area[bb, i, j] > 0:
                        area_to_red += bande_area[bb, i, j]
        if area_to_red > 0:
            amount_per_area = we_to_red * basin_area[bb] / area_to_red
            for i in range(n_fasce):
                if (
                    fasce_area[bb, i] > 0
                    and lowerh[bb] <= fasce_elev_avg[bb, i] < upperh[bb]
                ):
                    for j in range(n_bande):
                        if bande_area[bb, i, j] > 0.0:
                            WE[bb, i, j] += amount_per_area


# ---------------------------------------------
# MODULE: snow
# ---------------------------------------------
@njit
def snow(
    n_fasce,
    n_bande,
    basin_id,
    day_array,
    year_array,
    month_array,
    hour_array,
    n_basin,
    n_hours,
    fasce_area,
    bande_area,
    basin_area,
    fasce_elev_avg,
    basin_elev,
    t_intercept,
    t_slope,
    id_lapse_rate,
    n_lapse,
    basin_lapse,
    t_idlapse,
    prec,
    PCF,
    precgrad,
    tsnow,
    tmelt,
    snowfall,
    rainfall,
    sumT,
    WE,
    snowfall_fasce,
    snowfall_basin,
    rainfall_fasce,
    rainfall_basin,
    baseflow,
    baseflow_fasce,
    baseflow_glac,
    baseflow_glac_fasce,
    V_snow_melt,
    V_snow_melt_basin,
    V_snow_melt_fasce,
    rhosnow,
    V_glac_melt,
    snow_melt,
    lqw,
    snow_freeze,
    snow_freeze_basin,
    snow_freeze_fasce,
    WE_basin,
    lqw_basin,
    WE_fasce,
    lqw_fasce,
    bande_area_glac,
    albsnow,
    albglac,
    beta2,
    cmf,
    nmf,
    rmf,
    c5,
    c6,
    eta0,
    rhomin,
    EI,
    EImin,
    sunrise,
    sunset,
    SCA,
    WE_threshold,
    delaytime,
    liquidwater,
    refreezing,
):
    for time_step in range(1, n_hours + 1, 1):
        glac_melt = np.full((n_basin, n_fasce, n_bande), 0.0)

        if hour_array[time_step - 1] == 0:
            print(
                "snow",
                year_array[time_step - 1],
                month_array[time_step - 1],
                day_array[time_step - 1],
            )

        for current_basin in range(n_basin):
            current_month = month_array[time_step - 1]
            current_hour = hour_array[time_step - 1]

            rain_on_snow_area = 0.0
            excess_liq_water_basin = 0.0
            check_precip = False

            t_elev_bnd = np.full(n_fasce, -999.0)
            prec_elev_bnd = np.full(n_fasce, 0.0)
            energy_band_area = np.full(n_fasce, 0.0)
            p_vol_band = np.full(n_fasce, 0.0)
            p_vol_area = np.full(n_fasce, 0.0)

            excess_liq_water = np.full((n_fasce, n_bande), 0.0)
            Vstorage = np.full((n_fasce, n_bande), 0.0)

            rain_on_snow_band = np.full((n_fasce, n_bande), 0.0)

            for i in range(n_fasce):
                if fasce_area[current_basin, i] <= 0.0:
                    continue

                for ilapse in range(n_lapse):
                    if (basin_lapse[current_basin]) == (id_lapse_rate[ilapse]) and (
                        t_idlapse[time_step] == id_lapse_rate[ilapse]
                    ):
                        t_elev_bnd[i] = (
                            t_intercept[time_step - 1]
                            + t_slope[time_step - 1] * fasce_elev_avg[current_basin, i]
                        )

                dz = fasce_elev_avg[current_basin, i] - basin_elev[current_basin]
                prec_elev_bnd[i] = (
                    prec[current_basin, time_step - 1]
                    * PCF[current_basin]
                    * (1.0 + dz * precgrad[current_basin] / 1000.0)
                )
                prec_corr = prec[current_basin, time_step]

                energy_band_area[i] = np.sum(bande_area[current_basin, i, :])
                p_vol_band[i] = max(
                    0.0, prec_elev_bnd[i] * (1 / 1000.0) * energy_band_area[i]
                )
                p_vol_area[i] = prec_corr * (1 / 1000.0) * energy_band_area[i]

                # check if redistribution of precipitation leads non-physical precipitation volume in lower elevation bands
                if (
                    (1.0 + dz * precgrad[current_basin] < 0.0)
                    and (energy_band_area[i] > 0.0)
                    and (prec_corr > 0.0)
                ):
                    check_precip = True

                if check_precip:
                    prec[current_basin, time_step] = (
                        prec[current_basin, time_step]
                        * np.sum(p_vol_band[:])
                        / np.sum(p_vol_area[:])
                    )

            for i in range(n_fasce):
                if fasce_area[current_basin, i] <= 0.0:
                    continue

                if 0 < t_elev_bnd[i] < 30000:
                    sumT[current_basin, i] += t_elev_bnd[i]

                for j in range(n_bande):
                    if bande_area[current_basin, i, j] <= 0.0:
                        continue

                    Aband_Abasin_ratio = (
                        bande_area[current_basin, i, j] / basin_area[current_basin]
                    )
                    Aband_Aelev_ratio = (
                        bande_area[current_basin, i, j] / fasce_area[current_basin, i]
                    )

                    EI_current, isday = get_EI(
                        i,
                        j,
                        current_basin,
                        current_month,
                        current_hour,
                        sunrise,
                        sunset,
                        EI,
                        EImin,
                    )

                    if prec[current_basin, time_step] > 0.0:
                        # get the elevation difference from average basin elevation and elevation band
                        dz = (
                            fasce_elev_avg[current_basin, i] - basin_elev[current_basin]
                        )
                        # get precipitation for the current elevation band
                        prec_elev_bnd[i] = (
                            prec[current_basin, time_step]
                            * PCF[current_basin]
                            * (1.0 + dz * precgrad[current_basin] / 1000.0)
                        )

                        if t_elev_bnd[i] < tsnow:
                            snowfall[current_basin, i, j] = prec_elev_bnd[i]
                            snowfall_fasce[current_basin, i, time_step] += (
                                prec_elev_bnd[i] * Aband_Aelev_ratio
                            )
                            snowfall_basin[current_basin, time_step] += (
                                prec_elev_bnd[i] * Aband_Abasin_ratio
                            )

                            # for snow density computation: https://doi.org/10.1016/j.jhydrol.2016.03.061
                            rhofreshsnow = (
                                rhomin + (max(0.0, 1.8 * t_elev_bnd[i] + 32) / 100) ** 2
                            )  # fresh snow density
                            # new snow density is the weighted average between snowpack and fresh snow (simplified approach by MZ)
                            rhosnow[current_basin, i, j] = (
                                snowfall_fasce[current_basin, i, time_step]
                                * rhofreshsnow
                                + WE[current_basin, i, j] * rhosnow[current_basin, i, j]
                            ) / (
                                snowfall_fasce[current_basin, i, time_step]
                                + WE[current_basin, i, j]
                            )

                            WE[current_basin, i, j] += snowfall[current_basin, i, j]
                        else:
                            rainfall[current_basin, i, j] = prec_elev_bnd[i]
                            rainfall_fasce[current_basin, i, time_step] += (
                                prec_elev_bnd[i] * Aband_Aelev_ratio
                            )
                            rainfall_basin[current_basin, time_step] += (
                                prec_elev_bnd[i] * Aband_Abasin_ratio
                            )
                            if WE[current_basin, i, j] > 0.0:
                                rain_on_snow_band[i, j] = prec_elev_bnd[i]
                                rain_on_snow_area = (
                                    rain_on_snow_area
                                    + prec_elev_bnd[i] * Aband_Abasin_ratio
                                )
                            else:
                                baseflow[current_basin, time_step] += (
                                    prec_elev_bnd[i] * Aband_Abasin_ratio
                                )
                                baseflow_fasce[current_basin, i, time_step] += (
                                    prec_elev_bnd[i] * Aband_Aelev_ratio
                                )

                    # Albedo estimate (simplified)
                    if snowfall[current_basin, i, j] > 0.2:
                        albedo = albsnow[current_basin]
                        sumT[current_basin, i] = 0.0
                    elif sumT[current_basin, i] <= 0.1:
                        albedo = albsnow[current_basin]
                    else:
                        albedo = (
                            albsnow[current_basin] - beta2[current_basin]
                        ) - beta2[current_basin] * np.log10(sumT[current_basin, i])
                    if albedo < 0.3:
                        albedo = 0.3

                    # Snow/glacier melt logic
                    if WE[current_basin, i, j] <= 0.0 and t_elev_bnd[i] > tmelt:
                        if prec_elev_bnd[i] < 0.2:
                            if isday:
                                glac_melt[current_basin, i, j] = (
                                    cmf[current_basin]
                                    * EI_current
                                    * (1 - albglac[current_basin])
                                    * (t_elev_bnd[i] - tmelt)
                                )
                            else:
                                glac_melt[current_basin, i, j] = nmf[current_basin] * (
                                    t_elev_bnd[i] - tmelt
                                )
                        else:
                            glac_melt[current_basin, i, j] = (
                                rmf[current_basin] + prec_elev_bnd[i] / 80.0
                            ) * (t_elev_bnd[i] - tmelt)

                        # compute ratio between glacier area and subbasin/elevation band area
                        Aglac_Abasin_ratio = (
                            bande_area_glac[current_basin, i, j]
                            / basin_area[current_basin]
                        )
                        Aglac_Afascia_ratio = (
                            bande_area_glac[current_basin, i, j]
                            / fasce_area[current_basin, i]
                        )
                        Aglac_Abanda_ratio = (
                            bande_area_glac[current_basin, i, j]
                            / bande_area[current_basin, i, j]
                        )

                        # get the discharge from glacier melt, averaged over subb and elev band area
                        baseflow_glac[current_basin, time_step] = (
                            baseflow_glac[current_basin, time_step]
                            + glac_melt[current_basin, i, j] * Aglac_Abasin_ratio
                        )
                        baseflow_glac_fasce[current_basin, i, time_step] = (
                            baseflow_glac_fasce[current_basin, i, time_step]
                            + glac_melt[current_basin, i, j] * Aglac_Afascia_ratio
                        )
                        # get the total glacier melt over time
                        V_glac_melt[current_basin, i, j] = (
                            V_glac_melt[current_basin, i, j]
                            + glac_melt[current_basin, i, j] * Aglac_Abanda_ratio
                        )

                    if WE[current_basin, i, j] > 0.0:
                        # compactation first, according to Saloranta 2016
                        snowdepth = (
                            WE[current_basin, i, j] / rhosnow[current_basin, i, j]
                        )
                        eta = (
                            1
                            / (1 + 60 * lqw[current_basin, i, j] / snowdepth)
                            * rhosnow[current_basin, i, j]
                            / 0.25
                            * eta0
                            * np.exp(
                                -c5 * min(0.5 * t_elev_bnd[i], 0.0)
                                + c6 * rhosnow[current_basin, i, j]
                            )
                        )
                        ms = (
                            0.5 * 9.806 * WE[current_basin, i, j]
                        )  # average weight of the snowpack
                        deltasnowdepth = -ms / eta * snowdepth * 3600  # negative
                        snowdepth = snowdepth + deltasnowdepth
                        # uptade the snow density
                        rhosnow[current_basin, i, j] = (
                            WE[current_basin, i, j] / snowdepth
                        )

                        if t_elev_bnd[i] > tmelt:
                            # melting depends on: is it day or night? - snowfall/rainfall is occurring?
                            if prec_elev_bnd[i] < 0.2:  # no precipitation
                                if isday:  # it is daytime
                                    snow_melt[current_basin, i, j] = (
                                        cmf[current_basin]
                                        * EI_current
                                        * (1 - albedo)
                                        * (t_elev_bnd[i] - tmelt)
                                    )
                                else:
                                    snow_melt[current_basin, i, j] = nmf[
                                        current_basin
                                    ] * (t_elev_bnd[i] - tmelt)
                            else:
                                snow_melt[current_basin, i, j] = (
                                    rmf[current_basin] + prec_elev_bnd[i] / 80.0
                                ) * (t_elev_bnd[i] - tmelt)

                            # update WE
                            # check if WE is greater than melting, if not the whole snowpack is melted
                            if (
                                WE[current_basin, i, j]
                                >= snow_melt[current_basin, i, j]
                            ):
                                WE[current_basin, i, j] = (
                                    WE[current_basin, i, j]
                                    - snow_melt[current_basin, i, j]
                                )
                            else:
                                glac_melt[current_basin, i, j] = (
                                    snow_melt[current_basin, i, j]
                                    - WE[current_basin, i, j]
                                )
                                # compute ratio between glacier area and subbasin/elevation band area
                                Aglac_Abasin_ratio = (
                                    bande_area_glac[current_basin, i, j]
                                    / basin_area[current_basin]
                                )
                                Aglac_Afascia_ratio = (
                                    bande_area_glac[current_basin, i, j]
                                    / fasce_area[current_basin, i]
                                )
                                Aglac_Abanda_ratio = (
                                    bande_area_glac[current_basin, i, j]
                                    / bande_area[current_basin, i, j]
                                )
                                # get the discharge from glacier melt, averaged over subb and elev band area
                                baseflow_glac[current_basin, time_step] = (
                                    baseflow_glac[current_basin, time_step]
                                    + glac_melt[current_basin, i, j]
                                    * Aglac_Abasin_ratio
                                )
                                baseflow_glac_fasce[current_basin, i, time_step] = (
                                    baseflow_glac_fasce[current_basin, i, time_step]
                                    + glac_melt[current_basin, i, j]
                                    * Aglac_Afascia_ratio
                                )
                                # get the total glacier melt over time
                                V_glac_melt[current_basin, i, j] = (
                                    V_glac_melt[current_basin, i, j]
                                    + glac_melt[current_basin, i, j]
                                    * Aglac_Abanda_ratio
                                )

                                # snowpack is completely melted
                                snow_melt[current_basin, i, j] = WE[current_basin, i, j]
                                WE[current_basin, i, j] = 0.0

                            # get the liquid water storage into the snowpack
                            max_liq_water = WE[current_basin, i, j] * liquidwater
                            lqw[current_basin, i, j] = (
                                lqw[current_basin, i, j]
                                + snow_melt[current_basin, i, j]
                                + rain_on_snow_band[i, j]
                            )

                            # check if the liquid water content exceeded the maximum water content allowed
                            if lqw[current_basin, i, j] > max_liq_water:
                                excess_liq_water[i, j] = (
                                    lqw[current_basin, i, j] - max_liq_water
                                )
                                lqw[current_basin, i, j] = max_liq_water
                                storage_time = nint(
                                    delaytime / 1000.0 * WE[current_basin, i, j]
                                )

                                # if retention time into snowpack is > dt --> some melted water is stored, otherwise all the melted water goes in "baseflow"
                                if storage_time > 1:
                                    Vstorage[i, j] = excess_liq_water[i, j] * np.exp(
                                        -1.0 / storage_time
                                    )
                                else:
                                    Vstorage[i, j] = 0.0

                                baseflow[current_basin, time_step] = (
                                    baseflow[current_basin, time_step]
                                    + (excess_liq_water[i, j] - Vstorage[i, j])
                                    * Aband_Abasin_ratio
                                )
                                baseflow_fasce[current_basin, i, time_step] = (
                                    baseflow_fasce[current_basin, i, time_step]
                                    + (excess_liq_water[i, j] - Vstorage[i, j])
                                    * Aband_Aelev_ratio
                                )
                                lqw[current_basin, i, j] = (
                                    lqw[current_basin, i, j] + Vstorage[i, j]
                                )
                                excess_liq_water_basin = (
                                    excess_liq_water_basin
                                    + (excess_liq_water[i, j] - Vstorage[i, j])
                                    * Aband_Abasin_ratio
                                )

                                if storage_time < 1:
                                    excess_liq_water[i, j] = 0.0

                            # cumulative snow melting over energy band, elevation band and subbasin
                            V_snow_melt[current_basin, i, j] = (
                                V_snow_melt[current_basin, i, j]
                                + snow_melt[current_basin, i, j]
                            )
                            V_snow_melt_basin[current_basin, time_step] = (
                                V_snow_melt_basin[current_basin, time_step]
                                + snow_melt[current_basin, i, j] * Aband_Abasin_ratio
                            )
                            V_snow_melt_fasce[current_basin, i, time_step] = (
                                V_snow_melt_fasce[current_basin, i, time_step]
                                + snow_melt[current_basin, i, j] * Aband_Aelev_ratio
                            )

                        else:  # (t_elev_bnd[i] < tmelt):
                            # if T elevation band < Tmelting --> REFREEZING
                            snow_melt[current_basin, i, j] = 0.0
                            glac_melt[current_basin, i, j] = 0.0
                            snow_freeze[current_basin, i, j] = refreezing * (
                                tmelt - t_elev_bnd[i]
                            )
                            # check if refrezeed volume is > liquid water content
                            if (
                                snow_freeze[current_basin, i, j]
                                < lqw[current_basin, i, j]
                            ):
                                lqw[current_basin, i, j] = (
                                    lqw[current_basin, i, j]
                                    - snow_freeze[current_basin, i, j]
                                )
                            else:
                                snow_freeze[current_basin, i, j] = lqw[
                                    current_basin, i, j
                                ]
                                lqw[current_basin, i, j] = 0.0

                            snow_freeze_basin[current_basin, time_step] = (
                                snow_freeze_basin[current_basin, time_step]
                                + snow_freeze[current_basin, i, j] * Aband_Abasin_ratio
                            )
                            snow_freeze_fasce[current_basin, i, time_step] = (
                                snow_freeze_fasce[current_basin, i, time_step]
                                + snow_freeze[current_basin, i, j] * Aband_Aelev_ratio
                            )

                            # update snow water equivalent with possible refreezed water volume
                            WE[current_basin, i, j] = (
                                WE[current_basin, i, j]
                                + snow_freeze[current_basin, i, j]
                            )

                    WE_basin[current_basin, time_step] += (
                        WE[current_basin, i, j] * Aband_Abasin_ratio
                    )
                    lqw_basin[current_basin, time_step] += (
                        lqw[current_basin, i, j] * Aband_Abasin_ratio
                    )
                    WE_fasce[current_basin, i, time_step] += (
                        WE[current_basin, i, j] * Aband_Aelev_ratio
                    )
                    lqw_fasce[current_basin, i, time_step] += (
                        lqw[current_basin, i, j] * Aband_Aelev_ratio
                    )

                    get_SCA(
                        time_step,
                        WE[current_basin, i, j],
                        WE_threshold,
                        bande_area,
                        SCA,
                        current_basin,
                        i,
                        j,
                    )

    return WE, rhosnow, WE_basin, snowfall_basin, rainfall_basin, baseflow
