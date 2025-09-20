import os

import numpy as np
import pandas as pd

from pyic_top.ictop_utils import init_basin_vars, init_temper
from pyic_top.module_pet import PET_Hargreaves

# global variables
INPUT_FOLDER = os.path.join(
    "D:/25_ARFFS/59_progetti/02_arffs/06_ARFFS_TEXT_gfortran_python/"
    "topmelt_ichymod_ics/PY-TOP_V1.0",
    "INPUT",
)
OUTPUT_FOLDER = os.path.join(
    "D:/25_ARFFS/59_progetti/02_arffs/06_ARFFS_TEXT_gfortran_python/"
    "topmelt_ichymod_ics/PY-TOP_V1.0",
    "OUTPUT",
)
INITCOND_FOLDER = os.path.join("initcond")
TOPOLOGICAL_ELEMENT_FOLDER = "topological_elements"
PARAMETER_FOLDER = "parameters"
EEB_FOLDER = "elev_energy_bands"
TOPOLOGY_FOLDER = "topology"
METEO_FOLDER = "meteo"
TO_PDM_FOLDER = "to_pdm"
START_TIME = "2018-10-01 00:00"
END_TIME = "2019-10-01 00:00"
AVG_LAT = 46.7
WE_THRESHOLD = 20.0
QBASE_TYPE = 1
FLOAT_FORMAT_SM = "%.4f"
# first hour is initial condition

if __name__ == "__main__":
    start_time = pd.to_datetime(START_TIME, format="%Y-%m-%d %H:%M") + pd.Timedelta(
        hours=1
    )
    end_time = pd.to_datetime(END_TIME, format="%Y-%m-%d %H:%M")

    # count numer of simulated hours (first is IC)
    n_hours = int((end_time - start_time).total_seconds() / 3600 + 1)
    # build time array
    time_array = pd.date_range(start=start_time, end=end_time, freq="h")

    # read basin list, basin_id must be the model sequence
    df_basins, basin_id, n_basin, basin_elev, basin_area, basin_lapse = init_basin_vars(
        os.path.join(INPUT_FOLDER, TOPOLOGICAL_ELEMENT_FOLDER, "basins.txt")
    )
    # read general params
    df_general_params = pd.read_csv(
        os.path.join(INPUT_FOLDER, PARAMETER_FOLDER, "parameters_general.csv"),
        skipinitialspace=True,
    )
    # sunrise, sunset params
    df_sun_params = pd.read_csv(
        os.path.join(INPUT_FOLDER, PARAMETER_FOLDER, "sunrise_sunset.txt"),
        skipinitialspace=True,
    )
    # ET params
    df_dt_month = pd.read_csv(
        os.path.join(INPUT_FOLDER, PARAMETER_FOLDER, "evapparams.txt"),
        skipinitialspace=True,
    )
    dt_month = (
        np.asarray(df_dt_month["deltat"].values).reshape(n_basin, 12).transpose(1, 0)
    )

    # read temperature. must be ordered by time ad idlapse
    id_lapse_rate, t_idlapse, t_slope, t_intercept = init_temper(
        os.path.join(INPUT_FOLDER, METEO_FOLDER, "temperature.txt"),
        START_TIME,
        END_TIME,
    )

    # All dataframe must respect the same sequence order of the basins!!!
    # init (n_basin, n_hours) real vars
    PET = np.full((n_basin, n_hours + 1), -999.0)

    # build np arrays for PET from python non-numpy vars
    month_array = time_array.month.values
    hour_array = time_array.hour.values
    year_array = time_array.year.values
    day_array = time_array.day.values
    ecf = df_general_params["ECF"].values[0]

    # call PET
    PET = PET_Hargreaves(
        avg_lat=AVG_LAT,
        n_hours=n_hours,
        hour_array=hour_array,
        year_array=year_array,
        month_array=month_array,
        day_array=day_array,
        n_basin=n_basin,
        PET=PET,
        n_lapse=len(id_lapse_rate),
        basin_elev=basin_elev,
        basin_lapse=basin_lapse,
        t_intercept=t_intercept,
        t_slope=t_slope,
        ecf=ecf,
        dT_month=dt_month,
        id_lapse_rate=id_lapse_rate,
        t_idlapse=t_idlapse,
    )

    # # write final state vars
    # n_bn_array = np.arange(1, n_fasce + 1, 1)
    # n_cl_array = np.arange(1, n_bande + 1, 1)
    # # baseflow glac
    # pd.DataFrame({
    #     'time': time_array[-1].strftime('%Y-%m-%d %H'),
    #     'idba': basin_id,
    #     'value': baseflow_glac[:,-1]
    # }).to_csv(
    #     os.path.join(OUTPUT_FOLDER, 'state_var_baseflow_glac.txt'),
    #     index=False
    #     )

    # write PET
    pd.DataFrame(
        {
            "time": np.tile(time_array.strftime("%Y-%m-%d %H"), n_basin),
            "id_ba": np.repeat(basin_id, n_hours),
            "value": PET[:, 1:].reshape(n_basin * n_hours),
        }
    ).to_csv(
        os.path.join(OUTPUT_FOLDER, "pet.txt"),
        index=False,
        float_format=FLOAT_FORMAT_SM,
    )
