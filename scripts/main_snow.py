import json
import os

import numpy as np
import pandas as pd

from pyic_top.ictop_utils import init_basin_vars, init_elev_ener_vars, init_meteo
from pyic_top.module_snow import snow


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
FLOAT_FORMAT_SD = "%.4f"

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
    time_array_full = pd.date_range(
        start=start_time - pd.Timedelta(hours=1), end=end_time, freq="h"
    )

    # read basin list, basin_id must be the model sequence
    df_basins, basin_id, n_basin, basin_elev, basin_area, basin_lapse, basin_node = (
        init_basin_vars(
            os.path.join(INPUT_FOLDER, TOPOLOGICAL_ELEMENT_FOLDER, "basins.txt")
        )
    )
    # read snow basin params
    df_snow_params = pd.read_csv(
        os.path.join(INPUT_FOLDER, PARAMETER_FOLDER, "parameters_TOPMELT.csv"),
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

    # read elev bands, energy bands, EI, and sort by model sequence basin_id
    df_EI = pd.read_csv(
        os.path.join(INPUT_FOLDER, EEB_FOLDER, "EI.txt"), skipinitialspace=True
    )
    df_elevbnds = pd.read_csv(
        os.path.join(INPUT_FOLDER, EEB_FOLDER, "elevbnds.txt"),
        skipinitialspace=True,
    )
    df_energybnds = pd.read_csv(
        os.path.join(INPUT_FOLDER, EEB_FOLDER, "energybnds.txt"),
        skipinitialspace=True,
    )
    # init elev and energy bands arrays, they are
    # already ordered by model sequence
    (
        n_fasce,
        n_bande,
        fasce_area,
        bande_area,
        fasce_elev_avg,
        bande_area_glac,
        EI,
    ) = init_elev_ener_vars(df_elevbnds, df_energybnds, df_EI, n_basin)

    # read meteo. temperature must be ordered by time ad idlapse
    id_lapse_rate, t_idlapse, t_slope, t_intercept, prec = init_meteo(
        os.path.join(INPUT_FOLDER, METEO_FOLDER, "precipitation.txt"),
        os.path.join(INPUT_FOLDER, METEO_FOLDER, "temperature.txt"),
        START_TIME,
        END_TIME,
    )

    # read baseflow_glac and baseflow state vars
    baseflow_glac_0 = pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_baseflow_glac.txt"),
        skipinitialspace=True,
    )
    baseflow_0 = pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_baseflow.txt"),
        skipinitialspace=True,
    )
    # All dataframe must respect the same sequence order of the basins!!!
    # init (n_basin, n_hours) real vars
    baseflow_glac = np.full((n_basin, n_hours + 1), 0.0)
    baseflow = np.full((n_basin, n_hours + 1), 0.0)
    baseflow_glac[:, 0] = baseflow_glac_0["value"]
    baseflow[:, 0] = baseflow_0["value"]

    # these variables are not stored step by step
    # class vars
    # snow_freeze = np.asarray(pd.read_csv(
    #     os.path.join(INPUT_FOLDER, INITCOND_FOLDER,
    # 'state_var_SNOW_freezed.txt'),
    #     skipinitialspace=True
    # )['value']).reshape(n_basin, n_fasce, n_bande)
    # cumulative glacier melt
    rhosnow = np.asarray(
        pd.read_csv(
            os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_SNOW_rhosnow.txt"),
            skipinitialspace=True,
        )["value"]
    ).reshape(n_basin, n_fasce, n_bande)
    V_glac_melt = np.asarray(
        pd.read_csv(
            os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_SNOW_glacmelt.txt"),
            skipinitialspace=True,
        )["value"]
    ).reshape(n_basin, n_fasce, n_bande)
    lqw = np.asarray(
        pd.read_csv(
            os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_SNOW_liqW.txt"),
            skipinitialspace=True,
        )["value"]
    ).reshape(n_basin, n_fasce, n_bande)
    # snow_melt = np.asarray(pd.read_csv(
    #     os.path.join(INPUT_FOLDER, INITCOND_FOLDER,
    # 'state_var_SNOW_melt.txt'),
    #     skipinitialspace=True
    # )['value']).reshape(n_basin, n_fasce, n_bande)
    WE = np.asarray(
        pd.read_csv(
            os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_SNOW_WE.txt"),
            skipinitialspace=True,
        )["value"]
    ).reshape(n_basin, n_fasce, n_bande)
    # elev band vars
    sumT = np.asarray(
        pd.read_csv(
            os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_SNOW_sumT.txt"),
            skipinitialspace=True,
        )["value"]
    ).reshape(n_basin, n_fasce)

    # build np arrays from python non-numpy vars
    month_array = np.asarray(time_array.month)
    hour_array = np.asarray(time_array.hour)
    year_array = np.asarray(time_array.year)
    day_array = np.asarray(time_array.day)
    PCF = np.asarray(df_snow_params["PCF"])
    precgrad = np.asarray(df_snow_params["PRECGRAD"])
    tsnow = np.asarray(df_general_params["Tsnow"])[0]
    tmelt = np.asarray(df_general_params["Tmelt"])[0]
    albsnow = np.asarray(df_snow_params["ALBSNOW"])
    albglac = np.asarray(df_snow_params["ALBGLAC"])
    beta2 = np.asarray(df_snow_params["BETA2"])
    cmf = np.asarray(df_snow_params["CMF"])
    nmf = np.asarray(df_snow_params["NMF"])
    rmf = np.asarray(df_snow_params["RMF"])
    sunrise = np.asarray(df_sun_params["sunrise"])
    sunset = np.asarray(df_sun_params["sunset"])
    delaytime = np.asarray(df_general_params["DelayTime"])[0]
    liquidwater = np.asarray(df_general_params["LiquidWater"])[0]
    refreezing = np.asarray(df_general_params["Refreezing"])[0]
    c5 = np.asarray(df_general_params["c5"])[0]
    c6 = np.asarray(df_general_params["c6"])[0]
    eta0 = np.asarray(df_general_params["eta0"])[0]
    rhomin = np.asarray(df_general_params["rhomin"])[0]

    WE, rhosnow, WE_basin, snowfall_basin, rainfall_basin, baseflow = snow(
        n_fasce=n_fasce,
        n_bande=n_bande,
        basin_id=basin_id,
        day_array=day_array,
        year_array=year_array,
        month_array=month_array,
        hour_array=hour_array,
        n_basin=n_basin,
        n_hours=n_hours,
        fasce_area=fasce_area,
        bande_area=bande_area,
        basin_area=basin_area,
        fasce_elev_avg=fasce_elev_avg,
        basin_elev=basin_elev,
        t_intercept=t_intercept,
        t_slope=t_slope,
        id_lapse_rate=id_lapse_rate,
        n_lapse=len(id_lapse_rate),
        basin_lapse=basin_lapse,
        t_idlapse=t_idlapse,
        prec=prec,
        PCF=PCF,
        precgrad=precgrad,
        tsnow=tsnow,
        tmelt=tmelt,
        snowfall=np.full((n_basin, n_fasce, n_bande), -999.0),
        rainfall=np.full((n_basin, n_fasce, n_bande), -999.0),
        sumT=sumT,
        WE=WE,
        snowfall_fasce=np.full((n_basin, n_fasce, n_hours + 1), 0.0),
        snowfall_basin=np.full((n_basin, n_hours + 1), 0.0),
        rainfall_fasce=np.full((n_basin, n_fasce, n_hours + 1), 0.0),
        rainfall_basin=np.full((n_basin, n_hours + 1), 0.0),
        baseflow=baseflow,
        baseflow_fasce=np.full((n_basin, n_fasce, n_hours + 1), 0.0),
        baseflow_glac=baseflow_glac,
        baseflow_glac_fasce=np.full((n_basin, n_fasce, n_hours + 1), 0.0),
        V_snow_melt=np.full((n_basin, n_fasce, n_bande), 0.0),
        V_snow_melt_basin=np.full((n_basin, n_hours + 1), 0.0),
        V_snow_melt_fasce=np.full((n_basin, n_fasce, n_hours + 1), 0.0),
        rhosnow=rhosnow,
        V_glac_melt=V_glac_melt,
        snow_melt=np.full((n_basin, n_fasce, n_bande), 0.0),
        lqw=lqw,
        snow_freeze=np.full((n_basin, n_fasce, n_bande), 0.0),
        snow_freeze_basin=np.full((n_basin, n_hours + 1), 0.0),
        snow_freeze_fasce=np.full((n_basin, n_fasce, n_hours + 1), 0.0),
        WE_basin=np.full((n_basin, n_hours + 1), 0.0),
        lqw_basin=np.full((n_basin, n_hours + 1), 0.0),
        WE_fasce=np.full((n_basin, n_fasce, n_hours + 1), 0.0),
        lqw_fasce=np.full((n_basin, n_fasce, n_hours + 1), 0.0),
        bande_area_glac=bande_area_glac,
        albsnow=albsnow,
        albglac=albglac,
        beta2=beta2,
        cmf=cmf,
        nmf=nmf,
        rmf=rmf,
        c5=c5,
        c6=c6,
        eta0=eta0,
        rhomin=rhomin,
        EI=EI,
        EImin=1.0,
        sunrise=sunrise,
        sunset=sunset,
        SCA=np.full(n_hours + 1, 0.0),
        WE_threshold=WE_THRESHOLD,
        delaytime=delaytime,
        liquidwater=liquidwater,
        refreezing=refreezing,
    )

    # write final state vars
    n_bn_array = np.arange(1, n_fasce + 1, 1)
    n_cl_array = np.arange(1, n_bande + 1, 1)
    # baseflow glac
    pd.DataFrame(
        {
            "time": time_array[-1].strftime("%Y-%m-%d %H"),
            "idba": basin_id,
            "value": baseflow_glac[:, -1],
        }
    ).to_csv(os.path.join(OUTPUT_FOLDER, "state_var_baseflow_glac.txt"), index=False)
    # snow WE
    pd.DataFrame(
        {
            "time": time_array[-1].strftime("%Y-%m-%d %H"),
            "idba": np.repeat(basin_id, n_fasce * n_bande),
            "n_bn": np.tile(np.repeat(n_bn_array, n_bande), n_basin),
            "n_cl": np.tile(n_cl_array, n_fasce * n_basin),
            "value": WE.reshape(n_basin * n_fasce * n_bande),
        }
    ).to_csv(os.path.join(OUTPUT_FOLDER, "state_var_SNOW_WE.txt"), index=False)
    # snow density
    pd.DataFrame(
        {
            "time": time_array[-1].strftime("%Y-%m-%d %H"),
            "idba": np.repeat(basin_id, n_fasce * n_bande),
            "n_bn": np.tile(np.repeat(n_bn_array, n_bande), n_basin),
            "n_cl": np.tile(n_cl_array, n_fasce * n_basin),
            "value": rhosnow.reshape(n_basin * n_fasce * n_bande),
        }
    ).to_csv(
        os.path.join(OUTPUT_FOLDER, "state_var_SNOW_rhosnow.txt"),
        index=False,
        float_format=FLOAT_FORMAT_SD,
    )

    # to_pdm
    # liquid inflow to basin
    pd.DataFrame(
        {
            "time": np.tile(time_array.strftime("%Y-%m-%d %H"), n_basin),
            "idba": np.repeat(basin_id, n_hours),
            "value": baseflow[:, 1:].reshape(n_hours * n_basin),
        }
    ).to_csv(os.path.join(INPUT_FOLDER, TO_PDM_FOLDER, "baseflow.txt"), index=False)

    # rain + snow
    pd.DataFrame(
        {
            "time": np.tile(time_array.strftime("%Y-%m-%d %H"), n_basin),
            "idba": np.repeat(basin_id, n_hours),
            "rain": rainfall_basin[:, 1:].reshape(n_hours * n_basin),
            "snow": snowfall_basin[:, 1:].reshape(n_hours * n_basin),
        }
    ).to_csv(
        os.path.join(OUTPUT_FOLDER, "rain_snow.txt"),
        index=False,
    )

    # glacier melt to basin
    pd.DataFrame(
        {
            "time": np.tile(time_array.strftime("%Y-%m-%d %H"), n_basin),
            "idba": np.repeat(basin_id, n_hours),
            "value": baseflow_glac[:, 1:].reshape(n_hours * n_basin),
        }
    ).to_csv(
        os.path.join(INPUT_FOLDER, TO_PDM_FOLDER, "baseflow_glac.txt"),
        index=False,
    )
