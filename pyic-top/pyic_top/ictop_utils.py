import json
import os

import numpy as np
import pandas as pd


def init_basin_vars(filename):
    """Initialize basin parameters.

    Args:
        filename (_type_): _description_

    Returns
    -------
        _type_: _description_
    """
    basins = pd.read_csv(filename)
    basin_id = np.array(basins["idba"])
    basin_elev = np.array(basins["elev"])
    basin_area = np.array(basins["area"])
    basin_lapse = np.array(basins["idlapse"])
    n_basin = len(basin_id)

    # check unique id
    n2 = len(np.unique(basin_id))
    if n2 < n_basin:
        raise ValueError("Non unique basin id")
        return

    return basins, basin_id, n_basin, basin_elev, basin_area, basin_lapse


def sort_df_by_idba_seq(df0, basin_id):
    """Resort dataframe to given basin array sequence.

    Args:
        filename (_type_): _description_

    Returns
    -------
        _type_: _description_
    """
    df0["idba"] = pd.Categorical(df0["idba"], categories=basin_id, ordered=True)
    df0 = df0.sort_values("idba")

    return df0


def sort_df_by_idba_time_seq(df0, basin_id):
    """Resort dataframe to given basin array sequence and time.

    Args:
        filename (_type_): _description_

    Returns
    -------
        _type_: _description_
    """
    df0["idba"] = pd.Categorical(df0["idba"], categories=basin_id, ordered=True)
    df0 = df0.sort_values(["idba", "time"])

    return df0


def write_dict_to_json(dictionary, filename):
    """Write a dictionary to a .json file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)
    return


def read_json_seq(filename):
    """Read a json sequence file."""
    with open(filename) as f:
        seq_of_bas = json.load(f)
    # Converti le chiavi in interi
    seq_of_bas = {int(k): v for k, v in seq_of_bas.items()}
    return seq_of_bas


def reorder_df_to_seq(
    seq_order: str,
    seq_id1: str,
    seq_id2: str,
    df: pd.DataFrame,
    fold_path: str,
    file_path: str = "sequence_of_basins.json",
) -> pd.DataFrame:
    """Reorder a dataframe according to a sequence and a id field.

    Given a json file with the sequence order of field id1, the
    dataframe will be finally sorted by field id2.

    Args:
        fold_path (str): _description_
        file_path (str): _description_
        seq_order (str): _description_
        seq_id1 (str): first id field, the one in sequence order
        seq_id2 (str): second id field, the one to define final sorting
        df (pd.DataFrame): _description_

    Returns
    -------
        pd.DataFrame: _description_
    """
    # load basin sequence
    seq_of_x = read_json_seq(os.path.join(fold_path, file_path))

    # sort by model sequence
    df[seq_order] = df[seq_id1].map(seq_of_x)
    df = df.sort_values(by=[seq_order, seq_id2]).drop(columns=seq_order)

    # # uncomment this to sort by model sequence
    # df_elevbnds["seq_order"] = df_elevbnds["idba"].map(seq_of_bas)
    # df_elevbnds = df_elevbnds.sort_values(by=["seq_order", "h1"]).drop(
    #     columns="seq_order"
    # )

    # df_energybnds["seq_order"] = df_energybnds["idba"].map(seq_of_bas)
    # df_energybnds = df_energybnds.sort_values(by=["seq_order", "idbn"]).drop(
    #     columns="seq_order"
    # )

    # df_EI["seq_order"] = df_EI["idba"].map(seq_of_bas)
    # df_EI = df_EI.sort_values(by=["seq_order", "idbn"]).drop(
    #     columns="seq_order"
    # )

    # df_precipitation["seq_order"] = df_precipitation["idba"].map(seq_of_bas)
    # df_precipitation = df_precipitation.sort_values(
    #     by=["seq_order", "time"]
    # ).drop(columns="seq_order")

    return


def init_elev_ener_vars(df_elevbnds, df_energybnds, df_EI, n_basin):
    """Init variables according to fortran dimensions."""
    n_fasce = len(df_elevbnds["n_bn"].unique())
    n_bande = len(df_energybnds["n_cl"].unique())
    df_elevbnds["h_avg"] = (df_elevbnds["h2"].values + df_elevbnds["h1"].values) / 2
    df_elevbnds["idba"] = pd.Categorical(
        df_elevbnds["idba"],
        categories=df_elevbnds["idba"].unique(),
        ordered=True,
    )
    fasce_area = df_elevbnds.pivot(
        index=["idba"], columns=["n_bn"], values="area"
    ).values
    fasce_elev_avg = df_elevbnds.pivot(
        index=["idba"], columns=["n_bn"], values="h_avg"
    ).values

    df_energybnds["idba"] = pd.Categorical(
        df_energybnds["idba"],
        categories=df_energybnds["idba"].unique(),
        ordered=True,
    )
    bande_area = df_energybnds.pivot(
        index=["idba"], columns=["n_bn", "n_cl"], values="area"
    ).values.reshape(n_basin, n_fasce, n_bande)
    bande_area_glac = df_energybnds.pivot(
        index=["idba"], columns=["n_bn", "n_cl"], values="area_glac"
    ).values.reshape(n_basin, n_fasce, n_bande)

    df_EI["idba"] = pd.Categorical(
        df_EI["idba"], categories=df_EI["idba"].unique(), ordered=True
    )
    EI = (
        df_EI.pivot(index=["idba"], columns=["idmo", "n_bn", "n_cl"], values="EI")
        .values.reshape(n_basin, n_fasce, 12, n_bande)
        .transpose(0, 2, 1, 3)
    )

    return (
        n_fasce,
        n_bande,
        fasce_area,
        bande_area,
        fasce_elev_avg,
        bande_area_glac,
        EI,
    )


def init_meteo(fprec, ftemp, START_TIME, END_TIME):
    df_precipitation = pd.read_csv(fprec, skipinitialspace=True)
    df_temperature = pd.read_csv(ftemp, skipinitialspace=True)
    df_precipitation = df_precipitation[
        (df_precipitation["time"] > START_TIME) & (df_precipitation["time"] <= END_TIME)
    ]
    df_temperature = df_temperature[
        (df_temperature["time"] > START_TIME) & (df_temperature["time"] <= END_TIME)
    ]

    t_slope = np.array(df_temperature["slo"])
    t_intercept = np.array(df_temperature["int"])
    t_idlapse = np.array(df_temperature["idlapse"])
    id_lapse_rate = np.unique(t_idlapse)

    df_precipitation["idba"] = pd.Categorical(
        df_precipitation["idba"],
        categories=df_precipitation["idba"].unique(),
        ordered=True,
    )
    prec = df_precipitation.pivot(
        index=["idba"], columns=["time"], values="value"
    ).values

    return id_lapse_rate, t_idlapse, t_slope, t_intercept, prec


def init_temper(ftemp, START_TIME, END_TIME):
    df_temperature = pd.read_csv(ftemp, skipinitialspace=True)

    df_temperature = df_temperature[
        (df_temperature["time"] > START_TIME[0:13])
        & (df_temperature["time"] <= END_TIME[0:13])
    ]

    if len(df_temperature) == 0:
        raise ValueError("Data time range is outside start and end time limits.")

    t_slope = np.array(df_temperature["slo"])
    t_intercept = np.array(df_temperature["int"])
    t_idlapse = np.array(df_temperature["idlapse"])
    id_lapse_rate = np.unique(t_idlapse)

    return id_lapse_rate, t_idlapse, t_slope, t_intercept


def init_baseflow(fbflow, START_TIME, END_TIME):
    df_baseflow = pd.read_csv(fbflow, skipinitialspace=True)

    df_baseflow = df_baseflow[
        (df_baseflow["time"] > START_TIME) & (df_baseflow["time"] <= END_TIME)
    ]

    df_baseflow["idba"] = pd.Categorical(
        df_baseflow["idba"],
        categories=df_baseflow["idba"].unique(),
        ordered=True,
    )
    baseflow = df_baseflow.pivot(
        index=["idba"], columns=["time"], values="value"
    ).values

    return baseflow
