import altair
import polars as pl

from lpm_plot import plot_marginal_1d, plot_marginal_2d


def test_plot_marginal_1d_smoke():
    observed_df = pl.read_csv("tests/resources/hand-written-observed.csv")
    synthetic_df = pl.read_csv("tests/resources/hand-written-observed.csv")
    columns = ["foo", "bar", "quagga"]
    assert isinstance(
        plot_marginal_1d(observed_df, synthetic_df, columns),
        altair.vegalite.v5.api.VConcatChart,
    )


def test_plot_marginal_2d_smoke():
    x = "foo"
    y = "bar"

    df1 = pl.DataFrame(
        {
            x: ["A", "A", "B", "B", "C", "C"],
            y: ["X", "Y", "X", "Y", "Z", "Z"],
            "Normalized frequency": [
                0.2,
                0.3,
                0.1,
                0.25,
                0.3,
                0.15,
            ],  # Normalized values for Dataset 1
        }
    )
    df2 = pl.DataFrame(
        {
            x: ["A", "A", "B", "B", "C", "C"],
            y: ["X", "Y", "X", "Y", "Z", "Z"],
            "Normalized frequency": [
                0.15,
                0.2,
                0.05,
                0.1,
                0.2,
                0.25,
            ],  # Normalized values for Dataset 2
        }
    )
    # Add a distinguishing column to each DataFrame and combine them
    df1 = df1.with_columns(pl.lit("Dataset 1").alias("Source"))
    df2 = df2.with_columns(pl.lit("Dataset 2").alias("Source"))
    combined_df = pl.concat([df1, df2])
    assert isinstance(
        plot_marginal_2d(combined_df, x, y),
        altair.vegalite.v5.api.HConcatChart,
    )
