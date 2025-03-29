import altair as alt
import geopandas as gpd
import json

GEOJSON_PATH = "resources/japan.geojson"  # Replace with your GeoJSON file path
with open(GEOJSON_PATH, "r") as f:
    JAPAN_GEOJSON = json.load(f)


def plot_map(probabilities):
    """
    Generates a choropleth map of Japan using Altair, visualizing probability values by prefecture.

    Parameters:
    -----------
    probabilities : polaras.DataFrame object
        A DataFrame containing at least two columns:
        - "prefecture": The names of Japanese prefectures.
        - "probability": Numeric values representing probabilities associated with each prefecture.

    Returns:
    --------
    altair.Chart
        An Altair chart object representing a choropleth map of Japan with probabilities visualized as color intensity.

    Raises:
    -------
    NotImplementedError
        If the `probabilities` DataFrame does not contain a column named
        "prefecture" - we only support this for Japan right now.

    Example:
    --------
    >>> import polars as pl
    >>> probabilities = pd.DataFrame({
    ...     "prefecture": ["Tokyo", "Osaka", "Kyoto"],
    ...     "probability": [0.8, 0.5, 0.6]
    ... })
    >>> chart = plot_map(probabilities)
    >>> chart.show()  # Render the map
    """
    if "prefecture" in probabilities.columns:
        geo_column = "prefecture"
        # Load GeoJSON data for Japan
        geo_df = gpd.GeoDataFrame.from_features(JAPAN_GEOJSON["features"])
        prefs = [
            name.split(" ")[0].replace("Osaka", "Ōsaka").replace("Hyogo", "Hyōgo")
            for name in geo_df.nam
        ]
        geo_df["prefecture"] = prefs
        geo_df = geo_df.rename(
            columns={"properties.prefecture": "prefecture"}
        )  # Match property name if needed

        # Merge the GeoDataFrame with the data DataFrame
        merged_df = geo_df.merge(probabilities.to_pandas(), on="prefecture", how="left")
        # Convert GeoDataFrame back to GeoJSON format for Altair
        merged_geojson = merged_df.to_json()

        # Create the Altair chart with a projection
        chart = (
            alt.Chart(alt.Data(values=json.loads(merged_geojson)["features"]))
            .mark_geoshape()
            .encode(
                color=alt.Color(
                    "properties.probability:Q",  # Replace "value" with your data column name
                    title="Probability",  # Set your desired title
                ),
                tooltip=[
                    "properties.prefecture:N",
                    "properties.probability:Q",
                ],  # Add tooltip
            )
            .project(
                type="mercator",  # Choose a projection (e.g., "mercator")
                scale=1550,  # Adjust the scale to zoom in or out
                center=[138.0, 38.0],  # Center the map on Japan (longitude, latitude)
            )
            .properties(
                width=800,
                height=600,
            )
        )
        return chart
    else:
        raise NotImplementedError
