"""
This module contains utility functions for the project.
"""


import cartopy.io.shapereader as shpreader
import geopandas as gpd
import pandas as pd


def get_bbox(ISO_A2):
    """
    Get the bounding box of a country based on its ISO 3166-1 alpha-2 code.

    Args:
        ISO_A2 (str): The ISO 3166-1 alpha-2 code of the country.

    """
    shp = shpreader.Reader(
        shpreader.natural_earth(
            resolution="10m", category="cultural", name="admin_0_countries"
        )
    )
    de_record = list(filter(lambda c: c.attributes["ISO_A2"] == ISO_A2, shp.records()))[0]
    de = pd.Series({**de_record.attributes, "geometry": de_record.geometry})
    x_west, y_south, x_east, y_north = de["geometry"].bounds
    return x_west, y_south, x_east, y_north