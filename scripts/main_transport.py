import json
import os

import numpy as np
import pandas as pd

from pyic_top.ictop_utils import init_reach_vars
from pyic_top.module_mc import MC, MC_params


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

    # read reaches list, reach id must be the model sequence
    df_reaches, reaches_id, n_reach, reach_in, reach_out, n_sub_reaches = init_reach_vars(
        os.path.join(INPUT_FOLDER, TOPOLOGICAL_ELEMENT_FOLDER, "reaches.txt")
    )
    # read nodelinks list
    df_nodelinks = pd.read_csv(
        os.path.join(INPUT_FOLDER, TOPOLOGY_FOLDER, "nodelinks.txt"),
        skipinitialspace=True,
    )
    # read transport sequence
    df_trans_seq = pd.read_csv(
        os.path.join(INPUT_FOLDER, TOPOLOGY_FOLDER, "transport_seq.txt"),
        skipinitialspace=True,
    )
    # read element list
    df_elements = pd.read_csv(
        os.path.join(INPUT_FOLDER, TOPOLOGY_FOLDER, "elements.txt"),
        skipinitialspace=True,
    ).set_index("idel")

    # read reaches params
    df_reach_params = pd.read_csv(
        os.path.join(INPUT_FOLDER, PARAMETER_FOLDER, "parameters_MC.csv"),
        skipinitialspace=True,
    ).set_index("idre")
    # calc MC params
    df_reach_params["c1_mc"], df_reach_params["c2_mc"], df_reach_params["c3_mc"] = MC_params(
        cel=df_reach_params["CEL"].to_numpy(),
        disp=df_reach_params["DISP"].to_numpy(),
        dx=df_reach_params["DX"].to_numpy(),
        dt=1.0 * 3600.0,  # time step in seconds
        n_reach=n_reach,
    )

    # All dataframe must respect the same sequence order of the reaches!!!
    # init (n_reach, n_hours) real vars
    Qnodelink = np.full((len(df_nodelinks), n_hours + 1), -999.0)

    # init flows at nodelinks
    df_Qnodelink0 = pd.read_csv(
        os.path.join(INPUT_FOLDER, TO_TRNSPRT_FOLDER, "discharge.txt"),
        skipinitialspace=True,
    )["value"].to_numpy()  # first nodelink is basin outlet

    # init (n_reach) real vars
    df_Qsubreach_in = pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_MC_Qin.txt"),
        skipinitialspace=True,
    ).set_index("idre")
    df_Qsubreach_out = pd.read_csv(
        os.path.join(INPUT_FOLDER, INITCOND_FOLDER, "state_var_MC_Qout.txt"),
        skipinitialspace=True,
    ).set_index("idre")

    # lopp over time steps exceluded init time
    for t in range(1, len(time_array) + 1, 1):
        if time_array.hour[t - 1] == 0:
            print(
                "Transport",
                time_array.year[t - 1],
                time_array.month[t - 1],
                time_array.day[t - 1]
            )
        # transport loop over transport sequence
        for i, row in df_trans_seq.iterrows():
            if row["idmo"] == 1:  # basin
                continue
            elif row["idmo"] == 2:  # MC
                # find upstream nodelink
                reach_id = df_elements.loc[row["idel"]]["idxx"]
                # Muskingum-Cunge
                MC(
                    nstep=df_reaches.loc[reach_id]["nreaches"],
                    Qsubreach_in=df_Qsubreach_in.loc[reach_id]['value'].to_numpy(),
                    Qsubreach_out=df_Qsubreach_out.loc[reach_id]['value'].to_numpy(),
                    c1_mc=df_reach_params.loc[reach_id]["c1_mc"],
                    c2_mc=df_reach_params.loc[reach_id]["c2_mc"],
                    c3_mc=df_reach_params.loc[reach_id]["c3_mc"],
                )
                

                MC(
                    reach_id=reach_id,
                    time_step=time_step,
                    nstep=n_sub_reaches,
                    Qsubreach_in=Qsubreach_in,
                    Qsubreach_out=Qsubreach_out,
                    n_nodelinks=len(df_nodelinks),
                    nodelink_node_d=df_nodelinks["node_d"].to_numpy(),
                    reach_node_u=df_reaches["node_u"].to_numpy(),
                    nodelink_id=df_nodelinks["idno"].to_numpy(),
                    Qnodelink=MC_Qout.reshape(-1, 1),  # (nodelink, time)
                    c1_mc=df_reach_params["C1_MC"].to_numpy(),
                    c2_mc=df_reach_params["C2_MC"].to_numpy(),
                    c3_mc=df_reach_params["C3_MC"].to_numpy(),
                    Q_mc=Qrunoff,
                )

                # write output step by step
                pd.DataFrame(
                    {
                        "time": time_array.strftime("%Y-%m-%d %H"),
                        "idre": np.repeat(reaches_id[current_reach], n_hours),
                        "value": Qrunoff[current_reach, 1 : n_hours + 1],
                    }
                ).to_csv(
                    os.path.join(
                        OUTPUT_FOLDER,
                        TO_PDM_FOLDER,
                        f"discharge_reach_{reaches_id[current_reach]}.txt",
                    ),
                    index=False,
                    float_format="%.4f",
                )


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
            "idno": np.repeat(basin_node, n_hours),
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
