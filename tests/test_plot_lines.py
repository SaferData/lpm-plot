import altair
import pytest

from lpm_plot import plot_lines


def test_plot_lines_smoke():
    data = {
        "train": [5.0, 4.0, 3.0, 2.5],
        "test": [5.2, 4.3, 3.5, 3.0],
    }
    chart = plot_lines(data)
    assert isinstance(chart, altair.vegalite.v5.api.Chart)


def test_plot_lines_empty_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        plot_lines({})


def test_plot_lines_unequal_lengths_raises():
    data = {
        "a": [1.0, 2.0, 3.0],
        "b": [1.0, 2.0],
    }
    with pytest.raises(ValueError, match="same length"):
        plot_lines(data)


def test_plot_lines_y_scale_log():
    data = {"series": [1.0, 10.0, 100.0]}
    chart = plot_lines(data, y_scale="log")
    spec = chart.to_dict(format="vega")
    assert spec["scales"][1]["type"] == "log"


def test_plot_lines_y_scale_sqrt():
    data = {"series": [1.0, 4.0, 9.0]}
    chart = plot_lines(data, y_scale="sqrt")
    spec = chart.to_dict(format="vega")
    assert spec["scales"][1]["type"] == "sqrt"


def test_plot_lines_y_scale_default():
    data = {"series": [1.0, 2.0, 3.0]}
    chart = plot_lines(data)
    spec = chart.to_dict(format="vega")
    # Default is linear scale
    assert spec["scales"][1]["type"] == "linear"


# %%

if __name__ == "__main__":
    data = {
        "train_loss": [5.0, 3.5, 2.8, 2.2, 1.9, 1.6, 1.4],
        "val_loss": [5.2, 3.8, 3.2, 2.8, 2.5, 2.3, 2.1],
        "test_loss": [5.5, 4.0, 3.5, 3.0, 2.7, 2.5, 2.3],
    }
    chart = plot_lines(data, x_title="Epoch", y_title="Loss")
    chart.show()
# %%
