import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np


def find_stationary_timepoints(df: pd.DataFrame) -> tuple[list[int], list[float]]:
    """

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing columns 'cycle', 'frame', 'x', 'dtot', and 'time_since_nc11'.

    Returns
    -------
    tuple[list[int]], list[float]]
        (min_mvmt_frames, times)

    """
    min_mvmt_frames = []
    times = []
    cycles = [10, 11, 12, 13, 14]
    df = df.copy()

    for cycle in cycles:
        cdf = df[df["cycle"] == cycle].copy()
        fgb = cdf.groupby("frame")

        max_pts = fgb["x"].count().max()

        in_cycle = fgb["x"].count() > (max_pts * 0.8)
        cycle_frames = in_cycle.index[in_cycle]

        min_mvmt_frame = df[df["frame"].isin(cycle_frames)].groupby("frame")["dtot"].mean().rolling(5).mean().idxmin()
        times.append(
            df[df["frame"].isin(cycle_frames)].groupby("time_since_nc11")["dtot"].mean().rolling(5).mean().idxmin())
        min_mvmt_frames.append(min_mvmt_frame)

    return min_mvmt_frames, times


def generate_timepoint_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates an abridged dataframe from a full spots, dataframe,
    where each row corresponds to a spot at one of the stationary timepoints,
    and the lineage tree is subsetted accordingly.
    Parameters
    ----------
    df
        spots dataframe, generated in load_data.load_spots_data

    Returns
    -------
    subsetted dataframe at stationary timepoints
    """
    min_mvmt_frames, times = find_stationary_timepoints(df)

    parent_ids = df[df["n_children"] == 2].index

    daughter_nuclei_df = df[df["parent_id"].isin(parent_ids)]

    tracklets = daughter_nuclei_df["tracklet_id"].values
    parent_tracklets = daughter_nuclei_df["parent_id"].map(df["tracklet_id"]).values

    tracklet_mapper = dict(zip(tracklets, parent_tracklets))

    timepoint_df = df[df["frame"].isin(min_mvmt_frames)].copy()
    timepoint_df["prev_tracklet_id"] = timepoint_df["tracklet_id"].map(tracklet_mapper)

    tracklets = timepoint_df["tracklet_id"].values
    tracklet_id_mapper = dict(zip(tracklets, timepoint_df.index))
    timepoint_df["prev_id"] = timepoint_df["prev_tracklet_id"].map(tracklet_id_mapper)

    for cycle, time in zip([10, 11, 12, 13, 14], times):
        cycle_df = timepoint_df[timepoint_df["cycle"] == cycle]
        points = cycle_df[["x", "y", "z"]].values
        pairwise_distances = cdist(points, points)
        np.fill_diagonal(pairwise_distances, 500)
        first_neighbor_distances = pairwise_distances.min(axis=1)

        timepoint_df.loc[cycle_df.index, "first_neighbor_distance"] = first_neighbor_distances

    return timepoint_df
