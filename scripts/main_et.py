import json
import os

import numpy as np
import pandas as pd

from pyic_top.ictop_utils import init_basin_vars, init_temper
from pyic_top.module_pet import PET_Hargreaves


def _read_init(cfg_path: str = "config.json") -> dict:
    try:
        with open(cfg_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON file: {e}")
        return {}


config_dir = _read_init("config.json")
init_info = _read_init("init.json")

# global variables
INPUT_FOLDER = os.path.join(
    config_dir["main_dir"],
    config_dir["input_dir"],
)
OUTPUT_FOLDER = os.path.join(
    config_dir["main_dir"],
    config_dir["output_dir"],
)
INITCOND_FOLDER = config_dir["initcond_dir"]
TOPOLOGICAL_ELEMENT_FOLDER = config_dir["topo_ele_dir"]
PARAMETER_FOLDER = config_dir["param_dir"]
EEB_FOLDER = config_dir["eeb_dir"]
TOPOLOGY_FOLDER = config_dir["topology_dir"]
METEO_FOLDER = config_dir["meteo_dir"]
TO_PDM_FOLDER = config_dir["to_pdm_dir"]
START_TIME = init_info["start_time"]
END_TIME = init_info["end_time"]
AVG_LAT = init_info["average_lat"]
AVG_LON = init_info["average_lon"]
WE_THRESHOLD = init_info["sca_we_threshold"]
QBASE_TYPE = init_info["qbase_type"]
FLOAT_FORMAT_SM = "%.4f"

# first hour is initial condition

if __name__ == "__main__":
    start_time = pd.to_datetime(START_TIME, format="%Y-%m-%d %H:%M:%S") + pd.Timedelta(
        hours=1
    )
    end_time = pd.to_datetime(END_TIME, format="%Y-%m-%d %H:%M:%S")

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
        os.path.join(INPUT_FOLDER, PARAMETER_FOLDER, "sunrise_sunset.csv"),
        skipinitialspace=True,
    )
    # ET params
    df_dt_month = pd.read_csv(
        os.path.join(INPUT_FOLDER, PARAMETER_FOLDER, "evapparams.csv"),
        skipinitialspace=True,
    )
    dt_month = np.asarray(df_dt_month["deltat"]).reshape(n_basin, 12).transpose(1, 0)

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
    month_array = np.asarray(time_array.month)
    hour_array = np.asarray(time_array.hour)
    year_array = np.asarray(time_array.year)
    day_array = np.asarray(time_array.day)
    ecf = np.asarray(df_general_params["ECF"])[0]

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
            "idba": np.repeat(basin_id, n_hours),
            "value": PET[:, 1:].reshape(n_basin * n_hours),
        }
    ).to_csv(
        os.path.join(OUTPUT_FOLDER, "pet.txt"),
        index=False,
        float_format=FLOAT_FORMAT_SM,
    )
