import altair
import polars as pl

from lpm_plot import plot_fidelity


def test_plot_fidelity_smoke_tvd():
    # Make up some fake fidelity data measuring tvd.
    fidelity_data_tvd = pl.DataFrame(
        [
            {
                "column-1": "total_score",
                "column-2": "sports_flg",
                "tvd": 0.001757401,
                "model": "LPM",
                "index": 0,
            },
            {
                "column-1": "terrace_flg",
                "column-2": "darts_flg",
                "tvd": 0.0018333333,
                "model": "LPM",
                "index": 1,
            },
            {
                "column-1": "total_score",
                "column-2": "live_flg",
                "tvd": 0.0020490196,
                "model": "LPM",
                "index": 2,
            },
            {
                "column-1": "total_score",
                "column-2": "closed",
                "tvd": 0.056201461,
                "model": "LPM",
                "index": 35,
            },
        ]
    )
    assert isinstance(
        plot_fidelity(fidelity_data_tvd), altair.vegalite.v5.api.LayerChart
    )
    assert isinstance(
        plot_fidelity(fidelity_data_tvd, metric="tvd"),
        altair.vegalite.v5.api.LayerChart,
    )


def test_plot_fidelity_smoke_kl():
    # Make up some fake fidelity data measuring kl.
    fidelity_data_kl = pl.DataFrame(
        [
            {
                "column-1": "total_score",
                "column-2": "sports_flg",
                "kl": 0.001757401,
                "model": "LPM",
                "index": 0,
            },
            {
                "column-1": "terrace_flg",
                "column-2": "darts_flg",
                "kl": 0.0018333333,
                "model": "LPM",
                "index": 1,
            },
            {
                "column-1": "total_score",
                "column-2": "live_flg",
                "kl": 0.0020490196,
                "model": "LPM",
                "index": 2,
            },
            {
                "column-1": "total_score",
                "column-2": "closed",
                "kl": 0.056201461,
                "model": "LPM",
                "index": 35,
            },
        ]
    )
    assert isinstance(
        plot_fidelity(fidelity_data_kl, metric="kl"), altair.vegalite.v5.api.LayerChart
    )


# %%
if __name__ == "__main__":
    import polars as pl

    from lpm_plot import plot_fidelity

    fidelity_data = pl.DataFrame(
        [
            {
                "column-1": "age",
                "column-2": "income",
                "tvd": 0.02,
                "model": "LPM",
                "index": 0,
            },
            {
                "column-1": "age",
                "column-2": "education",
                "tvd": 0.03,
                "model": "LPM",
                "index": 1,
            },
            {
                "column-1": "income",
                "column-2": "education",
                "tvd": 0.05,
                "model": "LPM",
                "index": 2,
            },
            {
                "column-1": "age",
                "column-2": "occupation",
                "tvd": 0.08,
                "model": "LPM",
                "index": 3,
            },
            {
                "column-1": "income",
                "column-2": "occupation",
                "tvd": 0.12,
                "model": "LPM",
                "index": 4,
            },
            {
                "column-1": "age",
                "column-2": "income",
                "tvd": 0.04,
                "model": "Baseline",
                "index": 0,
            },
            {
                "column-1": "age",
                "column-2": "education",
                "tvd": 0.06,
                "model": "Baseline",
                "index": 1,
            },
            {
                "column-1": "income",
                "column-2": "education",
                "tvd": 0.09,
                "model": "Baseline",
                "index": 2,
            },
            {
                "column-1": "age",
                "column-2": "occupation",
                "tvd": 0.15,
                "model": "Baseline",
                "index": 3,
            },
            {
                "column-1": "income",
                "column-2": "occupation",
                "tvd": 0.20,
                "model": "Baseline",
                "index": 4,
            },
        ]
    )
    chart = plot_fidelity(fidelity_data, metric="tvd")
    chart.show()
