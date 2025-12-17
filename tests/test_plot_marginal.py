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


# %%
if __name__ == "__main__":
    import polars as pl

    from lpm_plot import (
        plot_marginal_1d,
        plot_marginal_2d,
        plot_marginal_numerical_categorical,
        plot_marginal_numerical_numerical,
    )

    # --- plot_marginal_1d ---
    observed_df = pl.DataFrame(
        {
            "category": ["A", "A", "B", "B", "B", "C"],
            "status": ["active", "inactive", "active", "active", "inactive", "active"],
        }
    )
    synthetic_df = pl.DataFrame(
        {
            "category": ["A", "B", "B", "C", "C", "C"],
            "status": ["active", "active", "inactive", "inactive", "active", "active"],
        }
    )
    chart_1d = plot_marginal_1d(observed_df, synthetic_df, ["category", "status"])
    chart_1d.show()

# %%
if __name__ == "__main__":
    # --- plot_marginal_2d ---
    x, y = "category", "status"
    df1 = pl.DataFrame(
        {
            x: ["A", "A", "B", "B", "C", "C"],
            y: ["active", "inactive", "active", "inactive", "active", "inactive"],
            "Normalized frequency": [0.25, 0.15, 0.20, 0.10, 0.18, 0.12],
        }
    ).with_columns(pl.lit("Observed").alias("Source"))
    df2 = pl.DataFrame(
        {
            x: ["A", "A", "B", "B", "C", "C"],
            y: ["active", "inactive", "active", "inactive", "active", "inactive"],
            "Normalized frequency": [0.22, 0.18, 0.18, 0.12, 0.16, 0.14],
        }
    ).with_columns(pl.lit("Synthetic").alias("Source"))
    combined_df = pl.concat([df1, df2])
    chart_2d = plot_marginal_2d(combined_df, x, y)
    chart_2d.show()

# %%
if __name__ == "__main__":
    # --- plot_marginal_numerical_numerical ---
    import random

    random.seed(42)
    observed_num = pl.DataFrame(
        {
            "age": [random.gauss(35, 10) for _ in range(100)],
            "income": [random.gauss(50000, 15000) for _ in range(100)],
        }
    )
    synthetic_num = pl.DataFrame(
        {
            "age": [random.gauss(36, 11) for _ in range(100)],
            "income": [random.gauss(52000, 14000) for _ in range(100)],
        }
    )
    chart_num_num = plot_marginal_numerical_numerical(
        observed_num, synthetic_num, "age", "income"
    )
    chart_num_num.show()

# %%
if __name__ == "__main__":
    # --- plot_marginal_numerical_categorical ---
    observed_cat = pl.DataFrame(
        {
            "department": ["Sales", "Sales", "Engineering", "Engineering", "HR", "HR"]
            * 10,
            "salary": [random.gauss(60000, 10000) for _ in range(60)],
        }
    )
    synthetic_cat = pl.DataFrame(
        {
            "department": ["Sales", "Sales", "Engineering", "Engineering", "HR", "HR"]
            * 10,
            "salary": [random.gauss(62000, 11000) for _ in range(60)],
        }
    )
    chart_num_cat = plot_marginal_numerical_categorical(
        observed_cat, synthetic_cat, "department", "salary"
    )
    chart_num_cat.show()
