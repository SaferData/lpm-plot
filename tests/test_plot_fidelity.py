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
