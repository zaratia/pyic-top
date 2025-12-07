import numpy as np
from numba import njit


# ---------------------------------------------
# MODULE: calc radiation
# ---------------------------------------------
@njit
def get_Ra(yy, mm, dd, hh, avg_lat):
    n_day_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # check if it is a leap year
    num_days = 365
    if (yy % 4 == 0) and ((yy % 100 != 0) or (yy % 400 == 0)):
        num_days = 366

    # Get the julian day (number of day during the year: from 1 to 365 or 366)
    julian_day = 0
    for i in range(12):
        if i + 1 < mm:
            julian_day = julian_day + n_day_month[i + 1]

    if (num_days == 366) and (mm > 2):
        julian_day = julian_day + 1

    if hh == 24:
        julian_day = julian_day + dd + 1
    else:
        julian_day = julian_day + dd

    # get the latitude in radiants
    phi = (avg_lat / 360.0) * 2.0 * np.pi

    # compute the distance between Earth - Sun
    dr = 1 + 0.033 * np.cos(2 * np.pi / num_days * julian_day)

    # compute solar declination [rad]
    delta = 0.4093 * np.sin(2 * np.pi / num_days * julian_day - 1.405)

    # compute sunset angle [rad]
    omega_s = np.acos(-np.tan(phi) * np.tan(delta))

    # compute the solar radiation (out-of-Earth)
    rad = (
        15.392
        * dr
        * (
            omega_s * np.sin(phi) * np.sin(delta)
            + np.cos(phi) * np.cos(delta) * np.sin(omega_s)
        )
    )

    return rad


# ---------------------------------------------
# MODULE: calc Hargreaves PET
# ---------------------------------------------
@njit
def PET_Hargreaves(
    avg_lat,
    n_hours,
    hour_array,
    year_array,
    month_array,
    day_array,
    n_basin,
    PET,
    n_lapse,
    basin_elev,
    basin_lapse,
    t_intercept,
    t_slope,
    ecf,
    dT_month,
    id_lapse_rate,
    t_idlapse,
):
    for time_step in range(1, n_hours + 1, 1):
        if hour_array[time_step - 1] == 0:
            print(
                "PET_Hargreaves",
                year_array[time_step - 1],
                month_array[time_step - 1],
                day_array[time_step - 1],
            )

        rad = get_Ra(
            yy=year_array[time_step - 1],
            mm=month_array[time_step - 1],
            dd=day_array[time_step - 1],
            hh=hour_array[time_step - 1],
            avg_lat=avg_lat,
        )

        for current_basin in range(n_basin):
            current_month = month_array[time_step - 1]

            for ilapse in range(n_lapse):
                if (basin_lapse[current_basin] == id_lapse_rate[ilapse]) and (
                    t_idlapse[time_step - 1] == id_lapse_rate[ilapse]
                ):
                    t_basin = (
                        t_intercept[time_step - 1]
                        + basin_elev[current_basin] * t_slope[time_step - 1]
                    )
                    PET[current_basin, time_step] = (
                        ecf
                        * (t_basin + 17.8)
                        * 0.0023
                        * rad
                        * ((np.abs(dT_month[current_month - 1, current_basin])) ** 0.5)
                        / 24.0
                    )

    return PET
