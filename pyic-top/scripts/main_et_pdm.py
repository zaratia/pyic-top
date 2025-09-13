import os

import numpy as np
import pandas as pd

from pyic_top.ictop_utils import init_baseflow, init_basin_vars, init_temper
from pyic_top.module_pdm import pdm
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
INIT_FILE = "TOPMELT_sim_file.txt"
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
        1, "h"
    )
    end_time = pd.to_datetime(END_TIME, format="%Y-%m-%d %H:%M")

    # count numer of simulated hours (first is IC)
    n_hours = end_time - start_time
    n_hours = np.int32(n_hours.total_seconds() / 3600 + 1)
    # build time array
    time_array = pd.date_range(start=start_time, end=end_time, freq="h")

    # read basin list, basin_id must be the model sequence
    df_basins, basin_id, n_basin, basin_elev, basin_area, basin_lapse = init_basin_vars(
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
        os.path.join(INPUT_FOLDER, PARAMETER_FOLDER, "sunrise_sunset.txt"),
        skipinitialspace=True,
    )
    # ET params
    df_dt_month = pd.read_csv(
        os.path.join(INPUT_FOLDER, PARAMETER_FOLDER, "evapparams.txt"),
        skipinitialspace=True,
    )
    dt_month = df_dt_month["deltat"].values.reshape(n_basin, 12).transpose(1, 0)

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
    Qglac = np.full((n_basin, n_hours + 1), -999.0)
    Qbase = np.full((n_basin, n_hours + 1), -999.0)
    soil_moisture = np.full((n_basin, n_hours + 1), -999.0)

    # these variables are not stored step by step
    # basin vars
    Cstar = pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_Cstar.txt"),
        skipinitialspace=True,
    )["value"].values
    Qbase[:, 0] = pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_Qbase.txt"),
        skipinitialspace=True,
    )["value"].values
    Qrunoff_t1 = pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_Qrunoff_t1.txt"),
        skipinitialspace=True,
    )["value"].values
    Qglac_t1 = pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_Qrunoff_t1glac.txt"),
        skipinitialspace=True,
    )["value"].values
    Qrunoff_t2 = pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_Qrunoff_t2.txt"),
        skipinitialspace=True,
    )["value"].values
    Qglac_t2 = pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_Qrunoff_t2glac.txt"),
        skipinitialspace=True,
    )["value"].values
    runoff_glac = pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_runoff_glac.txt"),
        skipinitialspace=True,
    )["value"].values
    runoff = pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_runoff.txt"),
        skipinitialspace=True,
    )["value"].values
    sgw = pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_sgw.txt"),
        skipinitialspace=True,
    )["value"].values
    stg = pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_stg.txt"),
        skipinitialspace=True,
    )["value"].values

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

    # build np arrays from python non-numpy vars
    cmin = df_pdm_params["CMIN"].values
    cmax = df_pdm_params["CMAX"].values
    b = df_pdm_params["B"].values
    stmax = (b * cmin + cmax) / (b + 1)
    stmin = df_pdm_params["STMIN"].values
    be = df_pdm_params["BE"].values
    bg = df_pdm_params["BG"].values
    q0 = df_pdm_params["Q0"].values
    m = df_pdm_params["M"].values
    kg = df_pdm_params["KG"].values
    ks = df_pdm_params["KS"].values
    k1 = df_pdm_params["K1"].values
    k2 = df_pdm_params["K2"].values
    k_glac = df_pdm_params["KGLAC"].values

    (
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
        soil_moisture=soil_moisture,
        time_step_duration=3600,
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
            "value": soil_moisture[:, 1:].reshape(n_basin * n_hours),
        }
    ).to_csv(
        os.path.join(OUTPUT_FOLDER, "soilmoisture.txt"),
        index=False,
        float_format=FLOAT_FORMAT_SM,
    )
