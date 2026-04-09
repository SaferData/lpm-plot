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


def test_cluster_order():
    df = pl.read_csv("tests/resources/heatmap-test-data.csv")
    chart = plot_heatmap(df, interactive=False)
    sort_order = chart.encoding.x.to_dict()["sort"]
    assert sort_order == ["num1", "num2", "cat1", "cat2"]


# %%
if __name__ == "__main__":
    import polars as pl

    from lpm_plot import plot_heatmap

    df = pl.DataFrame(
        {
            "Column 1": [
                "age",
                "age",
                "age",
                "income",
                "income",
                "income",
                "education",
                "education",
                "education",
            ],
            "Column 2": [
                "age",
                "income",
                "education",
                "age",
                "income",
                "education",
                "age",
                "income",
                "education",
            ],
            "Score": [None, 0.8, 0.5, 0.8, None, 0.6, 0.5, 0.6, None],
        }
    )
    chart = plot_heatmap(df, interactive=False)
    chart.show()
