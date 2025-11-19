import altair as alt
import polars as pl

OBSERVED_COLOR = "#000000"
SYNTHETIC_COLOR = "#f28e2b"

alt.data_transformers.enable("vegafusion")


def get_max_frequency(column, data):
    "Calculate the maximum frequency value for a given column. This is used to align the axes of the comparison plots."
    result = (
        data.group_by(["data_source", column])
        .agg(pl.count(column).alias("count"))
        .group_by("data_source")
        .agg(pl.max("count").alias("max_count"))
    )
    return result.select(pl.max("max_count")).item()


def plot_marginal_1d(observed_df, synthetic_df, columns):
    "Plot 1D marginal plots for a given list of columns."
    assert len(columns) > 0.0
    for c in columns:
        assert c in observed_df.columns, "column not in observed data"
        assert c in synthetic_df.columns, "column not in synthetic data"

    observed_df = observed_df[columns].with_columns(
        pl.lit("observed").alias("data_source")
    )
    synthetic_df = synthetic_df[columns].with_columns(
        pl.lit("synthetic").alias("data_source")
    )

    data = pl.concat([observed_df, synthetic_df])

    # Issue: Altair doesn't allow me to add a custom legend, using this dummy data workaround.
    dummy_data_for_legend = pl.DataFrame(
        {
            "category": ["Observed", "Synthetic"],
            "dummy": [0, 0],
        }
    )

    # Create an empty plot with a legend and no visible marks
    legend = (
        alt.Chart(dummy_data_for_legend)
        .mark_point(size=0, opacity=0)
        .encode(
            color=alt.Color(
                "category:N",
                scale=alt.Scale(
                    domain=["Observed", "Synthetic"],
                    range=[OBSERVED_COLOR, SYNTHETIC_COLOR],
                ),
                legend=alt.Legend(title="Legend", symbolStrokeWidth=4),
            )
        )
    )

    # Creating the charts for observed and synthetic collections
    def create_comparison(column, data):
        max_count = get_max_frequency(column, data)

        # Pre-aggregate data in Polars to avoid VegaFusion type casting issues
        # Filter and aggregate observed data
        observed_agg = (
            data.filter(pl.col("data_source") == "observed")
            .filter(pl.col(column).is_not_null())
            .group_by(column)
            .agg(pl.len().alias("count"))
            .with_columns(pl.lit("observed").alias("data_source"))
        )

        # Filter and aggregate synthetic data
        synthetic_agg = (
            data.filter(pl.col("data_source") == "synthetic")
            .filter(pl.col(column).is_not_null())
            .group_by(column)
            .agg(pl.len().alias("count"))
            .with_columns(pl.lit("synthetic").alias("data_source"))
        )

        # Create charts with pre-aggregated data
        chart_observed = (
            alt.Chart(observed_agg)
            .mark_bar(color=OBSERVED_COLOR)
            .encode(
                x=alt.X(
                    "count:Q",
                    scale=alt.Scale(domain=[0, max_count]),
                    axis=alt.Axis(orient="top"),
                ),
                y=alt.Y(
                    f"{column}:N",
                    axis=alt.Axis(
                        titleAnchor="start",
                        titleAlign="right",
                        titlePadding=1,
                        titleAngle=0,
                    ),
                ),
                color=alt.value(OBSERVED_COLOR),
            )
            .properties(width=300, height=200)
        )

        chart_synthetic = (
            alt.Chart(synthetic_agg)
            .mark_bar(color=SYNTHETIC_COLOR)
            .encode(
                x=alt.X(
                    "count:Q",
                    scale=alt.Scale(domain=[0, max_count]),
                    axis=alt.Axis(orient="top"),
                ),
                y=alt.Y(
                    f"{column}:N",
                    axis=alt.Axis(
                        titleAnchor="start",
                        titleAlign="right",
                        titlePadding=1,
                        titleAngle=0,
                    ),
                ),
                color=alt.value(SYNTHETIC_COLOR),
            )
            .properties(width=300, height=200)
        )
        return alt.hconcat(chart_observed, chart_synthetic)

    one_d_plots = [create_comparison(column, data) for column in columns]
    combined_chart = (
        alt.vconcat(*one_d_plots, legend)
        .resolve_scale(color="independent")
        .properties(title="1-D Marginals")
    )
    return combined_chart


def prepare_2d_marginal_data(observed_df, synthetic_df, x, y):
    """
    Prepare data for 2D marginal plotting by calculating normalized frequencies.

    Args:
        observed_df (pl.DataFrame): Observed data
        synthetic_df (pl.DataFrame): Synthetic data
        x (str): First categorical column name
        y (str): Second categorical column name

    Returns:
        pl.DataFrame: Combined dataframe with Source and Normalized frequency columns
    """
    # Ensure both dataframes have the same column order
    columns = [x, y]
    observed_subset = observed_df.select(columns)
    synthetic_subset = synthetic_df.select(columns)

    observed_labeled = observed_subset.with_columns(pl.lit("Observed").alias("Source"))
    synthetic_labeled = synthetic_subset.with_columns(
        pl.lit("Synthetic").alias("Source")
    )

    combined = pl.concat([observed_labeled, synthetic_labeled])

    freq_data = combined.group_by(["Source", x, y]).agg(
        pl.count().cast(pl.Int64).alias("count")
    )

    total_counts = freq_data.group_by("Source").agg(
        pl.sum("count").cast(pl.Int64).alias("total_count")
    )

    result = (
        freq_data.join(total_counts, on="Source")
        .with_columns(
            (
                pl.col("count").cast(pl.Float64)
                / pl.col("total_count").cast(pl.Float64)
            ).alias("Normalized frequency")
        )
        .drop("total_count")
    )

    return result


def plot_marginal_2d(combined_df, x, y, hm_order=None, cmap="oranges"):
    """
    Plots 2D marginal heatmaps of normalized frequencies to compare relationships between categorical variables in different dataset sources (e.g. observed and synthetic).

    This function generates a series of 2D heatmaps (one per data source) that visualize the bivariate
    frequencies of two categorical variables (`x` and `y`). The heatmaps are displayed side by side, sharing
    the same color scale. The normalized frequency values are plotted using colored rectangles, where the
    color intensity indicates the frequency.

    Args:
        combined_df (pl.DataFrame): A Polars DataFrame containing the combined data from different sources.
            It must contain the columns specified by `x`, `y`, and a "Source" column, as well as a
            "Normalized frequency" column with the normalized frequencies pre-calculated (can use prepare_2d_marginal_data for this).
        x (str): The name of the first categorical column (horizontal axis of the heatmap).
        y (str): The name of the second categorical column (vertical axis of the heatmap).
        hm_order (list of str, optional): A custom order for the sources for the plot.
        cmap (str, optional): The color map to be used for the heatmap. Defaults to "oranges". Can be
            any valid Altair color scheme (e.g., "blues", "reds").

    Returns:
        alt.Chart: An Altair chart object containing the concatenated heatmaps, one for each source.
    """
    # Check if users wanted to order the datasources.
    if hm_order is None:
        order = sorted(combined_df["Source"].unique().to_list())
    else:
        order = hm_order

    # Pre-filter data in Polars for each source to avoid VegaFusion type casting issues
    heatmaps = []
    for source in order:
        source_df = combined_df.filter(pl.col("Source") == source)
        heatmap = (
            alt.Chart(source_df)
            .mark_rect()
            .encode(
                x=alt.X(f"{x}:N", title=f"{x}"),
                y=alt.Y(f"{y}:N", title=f"{y}"),
                color=alt.Color(
                    "Normalized frequency:Q",
                    scale=alt.Scale(scheme=cmap),
                    title="Normalized Count",
                ),
                tooltip=[x, y, "Normalized frequency:Q"],
            )
            .properties(width=400, height=400, title=source)
        )
        heatmaps.append(heatmap)

    combined_heatmap = alt.hconcat(*heatmaps).resolve_scale(color="shared")
    return combined_heatmap


def plot_marginal_numerical_numerical(
    observed_df: pl.DataFrame,
    synthetic_df: pl.DataFrame,
    x: str,
    y: str,
    x_domain: tuple[float | None, float | None] = [None, None],
    y_domain: tuple[float | None, float | None] = [None, None],
):
    """
    Plots 2D marginal scatter plot comparing numerical observed and synthetic data
    which are displayed black and orange respectively.

    Args:
        observed_df : pl.DataFrame
            A Polars DataFrame containing the observed data and it must contain the columns specified by `x`, `y`
        synthetic_df : pl.DataFrame
            A Polars DataFrame containing the synthetic data and it must contain the columns specified by `x`, `y`
        x : str
            The name of the first numerical column (horizontal axis of the plot).
        y : str
            The name of the second numerical column (vertical axis of the plot).
        x_domain : tuple[float | None, float | None], optional
            The domain of the x-axis and defaults to min and max of data. If None is provided instead of a min
            or a max, then the program will default to using the min or max of the data respectively.
        y_domain : tuple[float | None, float | None], optional
            The domain of the y-axis and defaults to min and max of data. If None is provided instead of a min
            or a max, then the program will default to using the min or max of the data respectively.

    Returns:
        alt.Chart: An Altair chart object containing the scatter plot.
    """
    x_domain = [
        (
            min(observed_df[x].min(), synthetic_df[x].min())
            if x_domain[0] is None
            else x_domain[0]
        ),
        (
            max(observed_df[x].max(), synthetic_df[x].max())
            if x_domain[1] is None
            else x_domain[1]
        ),
    ]
    y_domain = [
        (
            min(observed_df[y].min(), synthetic_df[y].min())
            if y_domain[0] is None
            else y_domain[0]
        ),
        (
            max(observed_df[y].max(), synthetic_df[y].max())
            if y_domain[1] is None
            else y_domain[1]
        ),
    ]

    # Make a combined data frame with a new dataset column specifying if the data is observed or synthetic
    observed_df_labeled = observed_df.with_columns(pl.lit("Observed").alias("dataset"))
    synthetic_df_labeled = synthetic_df.with_columns(
        pl.lit("Synthetic").alias("dataset")
    )
    combined_df = pl.concat([observed_df_labeled, synthetic_df_labeled], how="vertical")

    chart = (
        alt.Chart(combined_df)
        .mark_circle(color=OBSERVED_COLOR)
        .encode(
            x=alt.X(x, scale=alt.Scale(domain=x_domain)),
            y=alt.Y(y, scale=alt.Scale(domain=y_domain)),
            color=alt.Color(
                "dataset:N",
                scale=alt.Scale(
                    domain=["Observed", "Synthetic"],
                    range=[OBSERVED_COLOR, SYNTHETIC_COLOR],
                ),
                legend=alt.Legend(title="Legend", symbolStrokeWidth=4),
            ),
        )
        .properties(width=500, height=500)
    )

    return chart


def plot_marginal_numerical_categorical(
    observed_df: pl.DataFrame,
    synthetic_df: pl.DataFrame,
    x: str,
    y: str,
    size: float = 30.0,
    y_domain: tuple[float, float] = [None, None],
):
    """
    Plots 2D marginal box plot comparing numerical vs categorical observed and synthetic data
    which are displayed black and orange respectively.

    Args:
        observed_df : pl.DataFrame
            A Polars DataFrame containing the observed data. The columns of the dataframe should be the names
            of each category of the categorical data and each column contains the numerical data pertaining to that
            category.
        synthetic_df : pl.DataFrame
            A Polars DataFrame containing the synthetix data. The columns of the dataframe should be the names
            of each category of the categorical data and each column contains the numerical data pertaining to that
            category. Must have the
            same columns as the observed dataframe
        x : str
            The name of the categorical data (horizontal axis of the plot).
        y : str
            The name of the numerical data (vertical axis of the plot).
        size : float, optional
            The width of the box plot boxes.
        y_domain : tuple[float, float], optional
            The domain of the y-axis and defaults to min and max of data. If None is provided instead of a min
            or a max, then the program will default to using the min or max of the data respectively.

    Returns:
        alt.Chart: An Altair chart object containing the box plot.
    """
    # Add a new column to distinguish which data set the data came from
    observed_df_labeled = observed_df.with_columns(pl.lit("Observed").alias("dataset"))
    synthetic_df_labeled = synthetic_df.with_columns(
        pl.lit("Synthetic").alias("dataset")
    )

    combined_df = pl.concat([observed_df_labeled, synthetic_df_labeled])

    y_domain = [
        combined_df[y].min() if y_domain[0] is None else y_domain[0],
        combined_df[y].max() if y_domain[1] is None else y_domain[1],
    ]

    return (
        alt.Chart(combined_df)
        .mark_boxplot(size=size, outliers=True)
        .encode(
            x=alt.X(f"{x}:N", scale=alt.Scale(padding=0.5)),
            y=alt.Y(f"{y}:Q", scale=alt.Scale(domain=y_domain)),
            color=alt.Color(
                "dataset:N",
                scale=alt.Scale(
                    domain=["Observed", "Synthetic"],
                    range=[OBSERVED_COLOR, SYNTHETIC_COLOR],
                ),
            ),
            xOffset=alt.XOffset(
                "dataset:N",
                scale=alt.Scale(
                    domain=["Observed", "Synthetic"],
                    range=[-size, size],
                ),
            ),
        )
        .properties(width=(size * 2 + 50) * combined_df[x].n_unique(), height=400)
    )
