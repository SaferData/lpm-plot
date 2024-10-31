import altair
import polars as pl

from lpm_plot import plot_heatmap


def test_plot_heatmap_smoke():
    df = pl.DataFrame(
        {
            "Column 1": ["A", "A", "B", "B"],
            "Column 2": ["A", "B", "B", "A"],
            "Score": [
                None,
                0.5,
                None,
                0.1,
            ],
        }
    )

    assert isinstance(
        plot_heatmap(df),
        altair.vegalite.v5.api.Chart,
    )
