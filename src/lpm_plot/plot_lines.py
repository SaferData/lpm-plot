import altair as alt
import polars as pl

alt.data_transformers.enable("vegafusion")


def plot_lines(
    data: dict[str, list[float]],
    x_title: str = "Step",
    y_title: str = "Value",
    width: int = 500,
    height: int = 300,
    y_scale: str | None = None,
) -> alt.Chart:
    """Plot multiple lines on a single chart.

    Args:
        data: Dictionary mapping series names to lists of y-values.
            All lists must have the same length. X-axis is the index.
        x_title: Label for x-axis.
        y_title: Label for y-axis.
        width: Chart width in pixels.
        height: Chart height in pixels.
        y_scale: Scale type for y-axis (e.g., "log", "sqrt", "symlog").
            If None, uses linear scale.

    Returns:
        Altair Chart with one line per series.
    """
    if not data:
        raise ValueError("data must not be empty")

    lengths = [len(v) for v in data.values()]
    if len(set(lengths)) != 1:
        raise ValueError("All series must have the same length")

    n_steps = lengths[0]
    series_names = list(data.keys())

    df = pl.DataFrame(
        {
            "x": list(range(n_steps)) * len(series_names),
            "y": [v for series in series_names for v in data[series]],
            "series": [name for name in series_names for _ in range(n_steps)],
        }
    )

    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("x:O", title=x_title, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "y:Q",
                title=y_title,
                scale=alt.Scale(type=y_scale) if y_scale else alt.Undefined,
            ),
            color=alt.Color("series:N", legend=alt.Legend(title="Series")),
        )
        .properties(width=width, height=height)
    )

    return chart
