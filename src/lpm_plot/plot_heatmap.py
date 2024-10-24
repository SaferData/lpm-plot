import altair as alt
import polars as pl
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import numpy as np


def plot_heatmap(df: pl.DataFrame, cmap="greens"):
    """
    Generates a clustered heatmap using hierarchical clustering on a Polars DataFrame and visualizes it with Altair.

    Parameters
    ----------
    df : pl.DataFrame
        A Polars DataFrame containing the data to be plotted, with columns "Column 1", "Column 2", and "Score".
        "Column 1" and "Column 2" represent categorical labels along the x- and y-axes, respectively,
        while "Score" provides quantitative values for coloring the heatmap.
    cmap : str, optional
        Color scheme name for the heatmap. Defaults to "greens".

    Returns
    -------
    alt.Chart
        An Altair Chart object representing the clustered heatmap.

    Notes
    -----
    - The function first converts the Polars DataFrame to a pandas DataFrame to support operations required
      for hierarchical clustering and plotting with Altair.
    - A square matrix (pivot table) is created with "Column 1" and "Column 2" as indices, containing "Score"
      as values.
    - Hierarchical clustering is performed on the data to define the optimal ordering of rows and columns
      for visualization, providing clearer patterns in the heatmap.
    - The generated Altair Chart includes tooltips for "Column 1", "Column 2", and "Score" for interactive exploration.
    """
    assert "Column 1" in df.columns
    assert "Column 2" in df.columns
    assert "Score" in df.columns
    # Convert the polars dataframe to pandas for clustering and plotting
    df_pandas = df.to_pandas()

    # Pivot the data to create a square matrix for clustering
    df_pivot = df_pandas.pivot(index="Column 1", columns="Column 2", values="Score")
    # Create a condensed distance matrix from the pivoted matrix
    distance_matrix = 1 - df_pivot
    condensed_matrix = distance_matrix.fillna(0).to_numpy()  # Handle NaNs

    # Perform hierarchical clustering
    linkage_matrix = linkage(
        squareform(condensed_matrix, checks=False), method="average"
    )

    # Get the order of the rows/columns after clustering
    order = [df_pivot.index[i] for i in leaves_list(linkage_matrix)]
    # Create the heatmap using Altair
    base = (
        alt.Chart(df_pandas)
        .mark_rect()
        .encode(
            x=alt.X(
                "Column 1:N",
                title="Column 1",
                sort=order,  # Replace with your desired order
            ),
            y=alt.Y(
                "Column 2:N",
                title="Column 2",
                sort=order,  # Replace with your desired order
            ),
            color=alt.Color(
                "Score:Q",
                scale=alt.Scale(scheme=cmap),
            ),
            tooltip=["Column 1", "Column 2", "Score:Q"],
        )
    )

    return base
