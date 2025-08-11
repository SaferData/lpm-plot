import altair as alt
import numpy as np
import polars as pl
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def plot_heatmap(df: pl.DataFrame, detailed_df: pl.DataFrame = None, cmap="greens"):
    """
    Generates a clustered heatmap using hierarchical clustering on a Polars DataFrame and visualizes it with Altair.

    Parameters
    ----------
    df : pl.DataFrame
        A Polars DataFrame containing the data to be plotted, with columns "Column 1", "Column 2", and "Score".
        "Column 1" and "Column 2" represent categorical labels along the x- and y-axes, respectively,
        while "Score" provides quantitative values for coloring the heatmap.
    detailed_df : pl.DataFrame, optional
        A Polars DataFrame containing more detailed data about the data in df.
    cmaps : str, optional
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

    #
    click = alt.selection_point(fields=["Column 1", "Column 2"], on="click")

    # Create the heatmap using Altair
    base = (
        alt.Chart(df_pandas)
        .mark_rect()
        .add_params(click)
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
    if detailed_df is not None:
        detail_charts = (
            alt.layer(
                # Numerical vs Numerical - Scatter Plot
                alt.Chart(detailed_df)
                .mark_circle(size=60)
                .encode(
                    x=alt.X("x_data:Q"),
                    y=alt.Y("y_data:Q"),
                    tooltip=["x_data:Q", "y_data:Q"],
                )
                .transform_filter(click & (alt.datum.comparison_type == "num-num")),
                # Numerical vs Categorical - Box Plot
                alt.Chart(detailed_df)
                .mark_boxplot(size=60)
                .encode(
                    x=alt.X("x_data:N", scale=alt.Scale(padding=0.5)),
                    y="y_data:Q",
                    tooltip=["x_data:N", "y_data:Q"],
                )
                .transform_filter(click & (alt.datum.comparison_type == "num-cat")),
                # Categorical vs Categorical - Heatmap
                alt.Chart(detailed_df)
                .mark_rect()
                .encode(
                    x=alt.X("x_data:N"),
                    y=alt.Y("y_data:N"),
                    color=alt.Color(
                        "heat_score:Q",
                        scale=alt.Scale(scheme=cmap),
                    ),
                    tooltip=["x_data:N", "y_data:N", "heat_score:Q"],
                )
                .transform_filter(click & (alt.datum.comparison_type == "cat-cat")),
            )
            .resolve_scale(x="independent", y="independent")
            .properties(width=300, height=200)
        )

        return alt.hconcat(base, detail_charts)
    return base
