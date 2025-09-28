import json
import os

import numpy as np
import pandas as pd

from pyic_top.ictop_utils import init_baseflow, init_basin_vars, init_temper
from pyic_top.module_pdm import pdm
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
    config_dir['main_dir'],
    config_dir['input_dir'],
)
OUTPUT_FOLDER = os.path.join(
    config_dir['main_dir'],
    config_dir['output_dir'],
)
INITCOND_FOLDER = config_dir['initcond_dir']
TOPOLOGICAL_ELEMENT_FOLDER = config_dir['topo_ele_dir']
PARAMETER_FOLDER = config_dir['param_dir']
EEB_FOLDER = config_dir['eeb_dir']
TOPOLOGY_FOLDER = config_dir['topology_dir']
METEO_FOLDER = config_dir['meteo_dir']
TO_PDM_FOLDER = config_dir['to_pdm_dir']
TO_TRNSPRT_FOLDER = config_dir['to_trnsprt_dir']
START_TIME = init_info['start_time']
END_TIME = init_info['end_time']
AVG_LAT = init_info['average_lat']
AVG_LON = init_info['average_lon']
WE_THRESHOLD = init_info['sca_we_threshold']
QBASE_TYPE = init_info['qbase_type']
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
    df_basins, basin_id, n_basin, basin_elev, basin_area, basin_lapse, basin_node = init_basin_vars(
        os.path.join(INPUT_FOLDER, TOPOLOGICAL_ELEMENT_FOLDER, "basins.txt")
    )
    # read snow basin params
    df_pdm_params = pd.read_csv(
        os.path.join(INPUT_FOLDER, PARAMETER_FOLDER, "parameters_PDM.csv"),
        skipinitialspace=True,
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
    dt_month = (
        np.asarray(df_dt_month["deltat"]).reshape(n_basin, 12).transpose(1, 0)
    )

    # read temperature. must be ordered by time ad idlapse
    id_lapse_rate, t_idlapse, t_slope, t_intercept = init_temper(
        os.path.join(INPUT_FOLDER, METEO_FOLDER, "temperature.txt"),
        START_TIME,
        END_TIME,
    )
    # read baseflow. must be ordered by time ad idlapse
    baseflow = init_baseflow(
        os.path.join(INPUT_FOLDER, TO_PDM_FOLDER, "baseflow.txt"),
        START_TIME,
        END_TIME,
    )
    baseflow_glac = init_baseflow(
        os.path.join(INPUT_FOLDER, TO_PDM_FOLDER, "baseflow_glac.txt"),
        START_TIME,
        END_TIME,
    )

    # All dataframe must respect the same sequence order of the basins!!!
    # init (n_basin, n_hours) real vars
    PET = np.full((n_basin, n_hours + 1), -999.0)
    ET = np.full((n_basin, n_hours + 1), -999.0)
    Qrunoff = np.full((n_basin, n_hours + 1), -999.0)
    Qglac = np.full((n_basin, n_hours + 1), -999.0)  # TODO: this variable can be Qglac(n_basin)
    Qbase = np.full((n_basin, n_hours + 1), -999.0)
    Qsubsurf = np.full((n_basin, n_hours + 1), -999.0)
    soil_moisture = np.full((n_basin, n_hours + 1), -999.0)

    # these variables are not stored step by step
    # basin vars
    Cstar = np.asarray(pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_Cstar.txt"),
        skipinitialspace=True,
    )["value"])
    Qbase[:, 0] = np.asarray(pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_Qbase.txt"),
        skipinitialspace=True,
    )["value"])
    Qrunoff_t1 = np.asarray(pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_Qrunoff_t1.txt"),
        skipinitialspace=True,
    )["value"])
    Qglac_t1 = np.asarray(pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_Qrunoff_t1glac.txt"),
        skipinitialspace=True,
    )["value"])
    Qrunoff_t2 = np.asarray(pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_Qrunoff_t2.txt"),
        skipinitialspace=True,
    )["value"])
    Qglac_t2 = np.asarray(pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_Qrunoff_t2glac.txt"),
        skipinitialspace=True,
    )["value"])
    runoff_glac = np.asarray(pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_runoff_glac.txt"),
        skipinitialspace=True,
    )["value"])
    runoff = np.asarray(pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_runoff.txt"),
        skipinitialspace=True,
    )["value"])
    sgw = np.asarray(pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_sgw.txt"),
        skipinitialspace=True,
    )["value"])
    stg = np.asarray(pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_stg.txt"),
        skipinitialspace=True,
    )["value"])

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

    # build np arrays from python non-numpy vars
    # TODO: always use np.asarray when extracting values
    cmin = np.asarray(df_pdm_params["CMIN"])
    cmax = np.asarray(df_pdm_params["CMAX"])
    b = np.asarray(df_pdm_params["B"])
    stmax = (b * cmin + cmax) / (b + 1)
    stmin = np.asarray(df_pdm_params["STMIN"])
    be = np.asarray(df_pdm_params["BE"])
    bg = np.asarray(df_pdm_params["BG"])
    q0 = np.asarray(df_pdm_params["Q0"])
    m = np.asarray(df_pdm_params["M"])
    kg = np.asarray(df_pdm_params["KG"])
    ks = np.asarray(df_pdm_params["KS"])
    k1 = np.asarray(df_pdm_params["K1"])
    k2 = np.asarray(df_pdm_params["K2"])
    k_glac = np.asarray(df_pdm_params["KGLAC"])

    (
        stg,  # PDM storage
        sgw,  # groundwater
        Cstar,  # PDM capacity
        ET,  # evapotranspiration
        Qsubsurf,  # topmodel subsurface flow
        runoff_glac,  # surface glac runoff
        Qglac,  # glac flow at the outlet
        runoff,  # surface ruoff
        Qrunoff,  # runoff at the outlet
        soil_moisture,  # % storage (vs max storage)
        Qbase,  # base flow
    ) = pdm(
        n_hours=n_hours,
        n_basin=n_basin,
        month_array=month_array,
        hour_array=hour_array,
        year_array=year_array,
        day_array=day_array,
        stg=stg,
        sgw=sgw,
        Cstar=Cstar,
        runoff=runoff,
        runoff_glac=runoff_glac,
        Qbase=Qbase,
        PET=PET,
        ET=ET,
        baseflow=baseflow,
        baseflow_glac=baseflow_glac,
        Qbase_type=QBASE_TYPE,
        cmin=cmin,
        cmax=cmax,
        stmax=stmax,
        stmin=stmin,
        b=b,
        be=be,
        bg=bg,
        q0=q0,
        m=m,
        kg=kg,
        ks=ks,
        k1=k1,
        k2=k2,
        k_glac=k_glac,
        Qglac_t1=Qglac_t1,
        Qglac_t2=Qglac_t2,
        Qrunoff_t1=Qrunoff_t1,
        Qrunoff_t2=Qrunoff_t2,
        Qrunoff=Qrunoff,
        Qglac=Qglac,
        Qsubsurf=Qsubsurf,
        soil_moisture=soil_moisture,
        time_step_duration=1,
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

    # soil_moisture
    pd.DataFrame(
        {
            "time": np.tile(time_array.strftime("%Y-%m-%d %H"), n_basin),
            "id_ba": np.repeat(basin_id, n_hours),
            "value": np.asarray(soil_moisture[:, 1:]).reshape(n_basin * n_hours),
        }
    ).to_csv(
        os.path.join(OUTPUT_FOLDER, "soilmoisture.txt"),
        index=False,
        float_format=FLOAT_FORMAT_SM,
    )

    # to transport models
    pd.DataFrame(
        {
            "time": np.tile(time_array.strftime("%Y-%m-%d %H"), n_basin),
            "idba": np.repeat(basin_id, n_hours),
            "nout": np.repeat(basin_node, n_hours),
            "value": (
                np.asarray(Qglac[:, 1:]).reshape(n_basin * n_hours) +
                np.asarray(Qrunoff[:, 1:]).reshape(n_basin * n_hours) +
                np.asarray(Qbase[:, 1:]).reshape(n_basin * n_hours) +
                np.asarray(Qsubsurf[:, 1:]).reshape(n_basin * n_hours)
            ) * np.repeat(basin_area, n_hours) / 1000 / 3600  # m3/s
        }
    ).to_csv(
        os.path.join(INPUT_FOLDER, TO_TRNSPRT_FOLDER, "discharge.txt"),
        index=False,
        float_format=FLOAT_FORMAT_SM,
    )

    # to transport models
    pd.DataFrame(
        {
            "time": np.tile(time_array.strftime("%Y-%m-%d %H"), n_basin),
            "idba": np.repeat(basin_id, n_hours),
            "qglac": np.asarray(Qglac[:, 1:]).reshape(n_basin * n_hours),
            "qroff": np.asarray(Qrunoff[:, 1:]).reshape(n_basin * n_hours),
            "qbase": np.asarray(Qbase[:, 1:]).reshape(n_basin * n_hours),
            "qsubs": np.asarray(Qsubsurf[:, 1:]).reshape(n_basin * n_hours),
        }
    ).to_csv(
        os.path.join(OUTPUT_FOLDER, "basin_runoff.txt"),
        index=False,
        float_format=FLOAT_FORMAT_SM,
    )
