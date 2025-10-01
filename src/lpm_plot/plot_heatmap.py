import altair as alt
import polars as pl
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def plot_heatmap(
    df: pl.DataFrame,
    detailed_df: pl.DataFrame = None,
    cmap_main: str = "greens",
    cmap_detail: str = "greys",
    detail_color: str = "black",
    interactive: bool = True,
):
    """
    Generates a clustered heatmap using hierarchical clustering on a Polars DataFrame and visualizes it with Altair.

    Parameters
    ----------
    df : pl.DataFrame
        A Polars DataFrame with columns named "Column 1", "Column 2", and "Score."
        "Column 1" and "Column 2" represent categorical labels along the x- and y-axes, respectively,
        while "Score" provides numerical values for coloring the heatmap.
    detailed_df : pl.DataFrame, optional
        A Polars DataFrame containing more detailed information about the additional graph displayed after clicking specific heatmap cells.
        The detailed_df has the following columns:
        - "Column 1": Data class 1
        - "Column 2": Data class 2
        - "comparison_type": Specifies the graph type used to compare the two classes and can take the values "num-num", "num-cat", "cat-num", "cat-cat", and "same-same"
        - "x_data": Data for the x-axis
        - "y_data": Data for the y-axis
    cmap_main : str, optional
        Color scheme name for the mutual information heatmap. Defaults to "greens".
    cmap_detail : str, optional
        Color scheme name for the detail heatmap. Defaults to "greys".
    detail_color : str, optional
        Color for detail scatter and box plots. Defaults to "black".
    interactive : bool, optional
        Whether to make the plot interactive. When True, enables zooming, panning, and a detailed 2D subplot of the selected heatmap cell.
        Defaults to True.

    Returns
    -------
    alt.Chart
        An Altair Chart object representing the clustered heatmap.

    Notes
    -----
    1) Converts the Polars DataFrame to a pandas DataFrame to support operations required for hierarchical clustering and plotting with Altair.
    2) Creates a square matrix (pivot table) with "Column 1" and "Column 2" as indices, containing "Score"
      as values.
    3) Performs hierarchical clustering on the data to define the optimal ordering of rows and columns for visualization, providing clearer patterns in the heatmap.
    4) The generated Altair Chart includes tooltips for "Column 1", "Column 2", and "Score" for interactive exploration.


    - Current limitations: For numerical vs categorical comparisons in the detail graph, the number ticks appear on the opposite sides to avoid conflicting with
      other graph labels. These number ticks show as a single 0 on other graphs which is why they are moved to the opposite
      side to avoid overlaps. This is an unavoidable consequence of making an interactive vega-lite graphs like this.
    - The detail graph has a frequency bar that is unable to be hidden while non-heatmap graphs are displayed due to the limitations of vega-lite
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

    # Define filter fields for selected cells (only if interactive)
    if interactive:
        click = alt.selection_point(
            fields=["Column 1", "Column 2"],
            on="click",
            value=[{"Column 1": None}, {"Column 2": None}],
            clear=False,
        )
    else:
        click = None

    # Create the heatmap using Altair
    base = alt.Chart(df_pandas).mark_rect()

    # Add click parameter only if interactive
    if click is not None:
        base = base.add_params(click)

    base = base.encode(
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
        color=alt.condition(
            alt.datum.Score == 0,
            alt.value("white"),
            alt.Color(
                "Score:Q", scale=alt.Scale(scheme=cmap_main), legend=alt.Legend()
            ),
        ),
        tooltip=["Column 1", "Column 2", "Score:Q"],
    ).properties(
        width=400, height=400
    )  # Default scale

    # Return heatmap if no detailed data is provided
    if detailed_df is None:
        if interactive:
            return base.interactive()
        else:
            return base.properties(title="Mutual Information Heatmap")

    # If not interactive, return just the base heatmap even if detailed_df is provided
    if not interactive:
        return base.properties(title="Mutual Information Heatmap")

    detailed_df = detailed_df.vstack(
        pl.DataFrame(
            {
                "Column 1": None,
                "Column 2": None,
                "comparison_type": "none",
                "x_data": None,
                "y_data": None,
            }
        )
    )

    cat_cat_df = detailed_df.filter(pl.col("comparison_type") == "cat-cat")

    # Get frequency counts for all categorical-categorical data
    counted_df = (
        cat_cat_df.group_by(
            ["Column 1", "Column 2", "comparison_type", "x_data", "y_data"]
        )
        .len()
        .rename({"len": "Frequency"})
    )

    # Get all present unique category combo for each cat-cat comparison
    column_combos = cat_cat_df["Column 1", "Column 2"].unique()
    combo_filtered_dfs: list[pl.DataFrame] = []
    for combo in column_combos.iter_rows():
        combo_filtered_dfs.append(
            cat_cat_df.filter(
                (pl.col("Column 1") == combo[0]) & (pl.col("Column 2") == combo[1])
            ).unique(subset=["x_data", "y_data"])
        )

    # Add missing category combos with a frequency of 0
    for combo_filered_df in combo_filtered_dfs:
        combos = combo_filered_df["x_data", "y_data"].rows()
        x_vals = combo_filered_df["x_data"].unique()
        y_vals = combo_filered_df["y_data"].unique()
        for x in x_vals:
            for y in y_vals:
                if not x == y and (x, y) not in combos:
                    counted_df = counted_df.vstack(
                        pl.DataFrame(
                            [combo_filered_df.row(0)[:3] + (x, y, 0)],
                            schema=counted_df.schema,
                            orient="row",
                        )
                    )

    # Replace detail_df cat-cat comparisons with counted versions
    detailed_df = (
        detailed_df.filter(pl.col("comparison_type") != "cat-cat")
        .with_columns(pl.lit(0, dtype=pl.UInt32).alias("Frequency"))
        .vstack(counted_df)
    )

    # Empty graph that is displayed when data is compared to itself
    empty = alt.Chart(detailed_df).mark_text(
        text="No Data: self comparison", strokeWidth=0.5
    )
    if click is not None:
        empty = empty.transform_filter(
            click & (alt.datum.comparison_type == "same-same")
        )

    # Empty graph that is displayed when nothing is selected at the start
    empty2 = alt.Chart(detailed_df).mark_text(text="Nothing selected", strokeWidth=0.5)
    if click is not None:
        empty2 = empty2.transform_filter(click & (alt.datum.comparison_type == "none"))

    # Scatter plot when the data compared are both numerical
    scatter_plot = (
        alt.Chart(detailed_df)
        .mark_circle(size=60, color=detail_color)
        .encode(
            x=alt.X("x_data:Q", title=None, axis=alt.Axis(orient="bottom")),
            y=alt.Y("y_data:Q", title=None, axis=alt.Axis(orient="left")),
            tooltip=["x_data:Q", "y_data:Q"],
        )
    )
    if click is not None:
        scatter_plot = scatter_plot.transform_filter(
            click & (alt.datum.comparison_type == "num-num")
        )

    # Box plot when the data compared is numerical and categorical
    box_plot_horizontal = (
        alt.Chart(detailed_df)
        .mark_boxplot(size=60, color=detail_color)
        .encode(
            x=alt.X("x_data:Q", title=None, axis=alt.Axis(orient="top")),
            y=alt.Y(
                "y_data:N",
                scale=alt.Scale(padding=0.5),
                title=None,
                axis=alt.Axis(orient="left"),
            ),
            tooltip=["x_data:N", "y_data:Q"],
        )
    )
    if click is not None:
        box_plot_horizontal = box_plot_horizontal.transform_filter(
            click & (alt.datum.comparison_type == "num-cat")
        )

    # Box plot when the data compared is categorical and numerical
    box_plot_vertical = (
        alt.Chart(detailed_df)
        .mark_boxplot(size=60, color=detail_color)
        .encode(
            x=alt.X(
                "x_data:N",
                scale=alt.Scale(padding=0.5),
                title=None,
                axis=alt.Axis(orient="bottom"),
            ),
            y=alt.Y("y_data:Q", title=None, axis=alt.Axis(orient="right")),
            tooltip=["x_data:N", "y_data:Q"],
        )
    )
    if click is not None:
        box_plot_vertical = box_plot_vertical.transform_filter(
            click & (alt.datum.comparison_type == "cat-num")
        )

    # Heat map when the data compared are both categorical
    heatmap = (
        alt.Chart(detailed_df)
        .mark_rect(stroke="white")
        .encode(
            x=alt.X("x_data:N", title=None, axis=alt.Axis(orient="bottom")),
            y=alt.Y("y_data:N", title=None, axis=alt.Axis(orient="left")),
            color=alt.condition(
                alt.datum.Frequency == 0,
                alt.value("white"),
                alt.Color(
                    "Frequency:Q",
                    scale=alt.Scale(scheme=cmap_detail, domainMin=0),
                    legend=alt.Legend(orient="right", offset=10),
                ),
            ),
            tooltip=["x_data:N", "y_data:N", "Frequency:Q"],
        )
    )
    if click is not None:
        heatmap = heatmap.transform_filter(
            click & (alt.datum.comparison_type == "cat-cat")
        )

    # X axis labels - positioned to align with the detail chart
    labels_x = (
        alt.Chart(detailed_df)
        .mark_text(align="center", strokeWidth=0.5, fontSize=12)
        .encode(
            text="Column 1:N",
            x=alt.value(205),  # Center of the 350px wide detail chart (30 + 175)
            y=alt.value(15),  # Below the detail chart
        )
    )
    if click is not None:
        labels_x = labels_x.transform_filter(click)

    # Y axis label - positioned to align with the detail chart
    labels_y = (
        alt.Chart(detailed_df)
        .mark_text(angle=270, strokeWidth=0.5, fontSize=12)
        .encode(
            text="Column 2:N",
            x=alt.value(15),  # Left of the detail chart
            y=alt.value(150),  # Center of the 300px high detail chart
        )
    )
    if click is not None:
        labels_y = labels_y.transform_filter(click)

    # Layer charts together
    detail_charts = alt.hconcat(
        labels_y.properties(width=30),  # Narrow space for Y labels
        alt.layer(
            empty,
            empty2,
            scatter_plot,
            box_plot_horizontal,
            box_plot_vertical,
            heatmap,
        )
        .resolve_scale(x="shared", y="shared", color="independent")
        .resolve_legend(color="independent")
        .properties(width=350, height=300),  # Increased width to accommodate legend
        spacing=5,
    )

    # Create the main layout with proper spacing
    chart = alt.vconcat(
        base.properties(title="Mutual Information Heatmap"),
        detail_charts.properties(title="2D Detailed View (click on heatmap cells)"),
        labels_x,
    ).properties(padding={"left": 40, "right": 40, "top": 20, "bottom": 40}, spacing=20)

    # Apply interactivity if requested
    if interactive:
        return chart.interactive()
    else:
        return chart


def reformat_data(
    heatmap_df: pl.DataFrame, all_data: pl.DataFrame, data_margin: float = 1.0
):
    """
    Reformats the provided polars data frame so they can be used by the plot_heatmap function

    Parameters
    ----------
    heatmap_df : pl.DataFrame
        A Polars DataFrame containing the data to be plotted, with columns "Column 1", "Column 2", and "Score".
        "Column 1" and "Column 2" represent categorical labels along the x- and y-axes, respectively,
        while "Score" provides quantitative values for coloring the heatmap.
    all_data : pl.DataFrame
        A Polars DataFrame containing more detailed data about the data used to make the heatmap. Columns are specific data classes
        and each row should represent all the data about specific sample.
    data_margin : float, optional
        Ratio all_data that should be reformatted from 0-1

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        Returns a tuple containing the reformatted heatmap and all_data data frames

    Notes
    -----
    - For the heatmap dataframe the only change is replace the missing value for score in the case of matching column values
    - all_data is reformatted to have "Column 1", "Column 2", "comparison_type", "x_data", and "y_data". "Column 1" and "Column 2"
      represent which 2 data classes are being compared and "comparison_tpye" determines which graph is used to show
      the comparison and can be "num-num", "num-cat", "cat-num", "cat-cat", and "same-same". "x_data" and "y_data" are the
      data being shown in the comparison referring to "Column 1" and "Column 2" respectively.
    """
    # Create df that will become the return resilt for all_data
    detail_df = pl.DataFrame(
        {
            "Column 1": pl.Series("Column 1", [], dtype=pl.String),
            "Column 2": pl.Series("Column 2", [], dtype=pl.String),
            "comparison_type": pl.Series("comparison_type", [], dtype=pl.String),
            "x_data": pl.Series("x_data", [], dtype=pl.Object),
            "y_data": pl.Series("y_data", [], dtype=pl.Object),
        }
    )

    # Get margin of all_data to be used
    data_slice = int(len(all_data) * data_margin)
    all_data = all_data[:data_slice]

    for row in heatmap_df.iter_rows():
        # Iterate through every column combo and get data pertaining to it
        cat1_data = all_data.get_column(row[0])
        cat2_data = all_data.get_column(row[1])

        # Handle if the combo is a self comparison
        if row[0] == row[1]:
            detail_df = pl.concat(
                [
                    detail_df,
                    pl.DataFrame(
                        {
                            "Column 1": [row[0]],
                            "Column 2": [row[1]],
                            "comparison_type": ["same-same"],
                            "x_data": [None],
                            "y_data": [None],
                        }
                    ),
                ],
                how="vertical_relaxed",
            )
        else:
            for i in range(len(all_data)):
                if cat1_data[i] is not None and cat2_data[i] is not None:
                    # Determine the comparison type
                    comparison_type = ""
                    if cat1_data.dtype == pl.String:
                        comparison_type = "cat-" + comparison_type
                    elif cat1_data.dtype == pl.Int64 or cat1_data.dtype == pl.Float64:
                        comparison_type = "num-" + comparison_type

                    if cat2_data.dtype == pl.String:
                        comparison_type += "cat"
                    elif cat2_data.dtype == pl.Int64 or cat1_data.dtype == pl.Float64:
                        comparison_type += "num"

                    # Add the data to the resulting data frame
                    detail_df = pl.concat(
                        [
                            detail_df,
                            pl.DataFrame(
                                {
                                    "Column 1": [row[0]],
                                    "Column 2": [row[1]],
                                    "comparison_type": [comparison_type],
                                    "x_data": [cat1_data[i]],
                                    "y_data": [cat2_data[i]],
                                }
                            ),
                        ],
                        how="vertical_relaxed",
                    )

    return (
        # Make all self-comparisons have a score of 0 in heatmap_df
        heatmap_df.with_columns(
            pl.when(pl.col("Column 1") == pl.col("Column 2"))
            .then(pl.lit(0.0))
            .otherwise(pl.col("Score"))
            .alias("Score")
        ),
        detail_df,
    )
