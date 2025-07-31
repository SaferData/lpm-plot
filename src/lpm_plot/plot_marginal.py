import altair as alt
import polars as pl

OBSERVED_COLOR = "#000000"
SYNTHETIC_COLOR = "#f28e2b"

alt.data_transformers.enable("vegafusion")


def get_max_frequency(column, data):
    # Group by `data_source` and the given `column`, then count occurrences
    result = (
        data.group_by(["data_source", column])
        .agg(pl.count(column).alias("count"))  # Count occurrences of each value
        .group_by("data_source")
        .agg(pl.max("count").alias("max_count"))  # Get max count for each data_source
    )
    # Return the max of the max counts
    return result.select(pl.max("max_count")).item()


def plot_marginal_1d(observed_df, synthetic_df, columns):
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

    # XXX: horrible hack; but Altair doesn't allow me to add a custom legend.
    dummy_data_for_legend = pl.DataFrame(
        {
            "category": ["Observed", "Synthetic"],
            "dummy": [0, 0],  # Dummy column to avoid visible marks
        }
    ).to_pandas()

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

    def create_marginal_chart(field, color, max_count):
        return (
            alt.Chart(data.to_pandas())
            .mark_bar(color=color)
            .encode(
                x=alt.X(
                    "count():Q",
                    scale=alt.Scale(domain=[0, max_count]),
                    axis=alt.Axis(orient="top"),
                ),
                y=alt.Y(
                    f"{field}:N",
                    axis=alt.Axis(
                        titleAnchor="start",
                        titleAlign="right",
                        titlePadding=1,
                        titleAngle=0,  # Rotate the y-axis label by 90 degrees (change as needed)
                    ),
                ),
                color=alt.value(color),
            )
            .transform_filter(f"datum.{field} != null")  # Filter out null values
        )

    # Creating the charts for observed and synthetic collections
    def create_comparison(column, data):
        max_count = get_max_frequency(column, data)
        chart_observed = create_marginal_chart(
            column, OBSERVED_COLOR, max_count
        ).transform_filter(alt.datum.data_source == "observed")
        chart_synthetic = create_marginal_chart(
            column, SYNTHETIC_COLOR, max_count
        ).transform_filter(alt.datum.data_source == "synthetic")
        return alt.hconcat(chart_observed, chart_synthetic)

    one_d_plots = [create_comparison(column, data) for column in columns]
    # Layer the charts for observed and synthetic data points
    combined_chart = (
        alt.vconcat(*one_d_plots, legend)
        .resolve_scale(color="independent")
        .properties(title="1-D Marginals")
    )
    return combined_chart


def plot_marginal_2d(combined_df, x, y, hm_order=None, cmap="oranges"):
    """
    Plots 2D marginal heatmaps of normalized frequencies for categorical variables across multiple sources.

    This function generates a series of 2D heatmaps (one per data source) that visualize the bivariate
    frequencies of two categorical variables (`x` and `y`). The heatmaps are displayed side by side, sharing
    the same color scale. The normalized frequency values are plotted using colored rectangles, where the
    color intensity indicates the frequency.

    Args:
        combined_df (pl.DataFrame): A Polars DataFrame containing the combined data from different sources.
            It must contain the columns specified by `x`, `y`, and a "Source" column, as well as a
            "Normalized frequency" column with the frequencies.
        x (str): The name of the first categorical column (horizontal axis of the heatmap).
        y (str): The name of the second categorical column (vertical axis of the heatmap).
        hm_order (list of str, optional): A custom order for the sources for the plot.
        cmap (str, optional): The color map to be used for the heatmap. Defaults to "oranges". Can be
            any valid Altair color scheme (e.g., "blues", "reds").

    Returns:
        alt.Chart: An Altair chart object containing the concatenated heatmaps, one for each source.
    """
    base = (
        alt.Chart(combined_df.to_pandas())
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
    )
    # Check if users wanted to order the datasources.
    if hm_order is None:
        order = sorted(combined_df["Source"].unique())
    else:
        order = hm_order
    heatmaps = [
        base.transform_filter(alt.datum.Source == source).properties(title=source)
        for source in order
    ]
    # Concatenate the heatmaps horizontally and ensure they share the color scale
    combined_heatmap = alt.hconcat(*heatmaps).resolve_scale(
        color="shared"  # Share the color scale between the heatmaps
    )
    return combined_heatmap


def plot_marginal_numerical_numerical(observed_df, synthetic_df, x, y):
    chart1 = alt.Chart(observed_df).mark_circle(color=OBSERVED_COLOR).encode(x=x, y=y)
    chart2 = alt.Chart(synthetic_df).mark_circle(color=SYNTHETIC_COLOR).encode(x=x, y=y)
    return chart1 + chart2

