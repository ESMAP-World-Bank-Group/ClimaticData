"""
This module contains utility functions for the project.
"""
import cartopy.io.shapereader as shpreader
import pandas as pd
import xarray as xr
import zipfile
import os
import matplotlib.pyplot as plt
import numpy as np
import folium
from folium import GeoJson, GeoJsonTooltip
import geopandas as gpd

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


def read_grib_file(grib_path, step_type=None):
    """
    Reads a GRIB file and returns an xarray dataset.
    This function uses the cfgrib engine to read the GRIB file and filter by stepType.
    It merges the datasets for 'avgid' and 'avgas' step types.
    :param grib_path: Path to the GRIB file.
    :return: An xarray dataset containing the data from the GRIB file.
    """
    print(f"Opening GRIB file: {grib_path}")
    try:
        if step_type:
            ds_avgid = xr.open_dataset(grib_path, engine='cfgrib',
                                       backend_kwargs={"filter_by_keys": {"stepType": "avgid"}},
                                       decode_timedelta=True)
            ds_avgas = xr.open_dataset(grib_path, engine="cfgrib",
                                       backend_kwargs={"filter_by_keys": {"stepType": "avgas"}},
                                       decode_timedelta=True)
            dataset = xr.merge([ds_avgid, ds_avgas])
        else:
            dataset = xr.open_dataset(grib_path, engine='cfgrib',
                                       decode_timedelta=True)

        #print(dataset)  # or process the dataset as needed
    except Exception as e:
        print(f"Failed to open {grib_path}: {e}")

    return dataset


def extract_data(zip_path, step_type=None, extract_to='era5_extracted_files'):
    """
    Extracts GRIB files from a ZIP file and reads them into xarray datasets.

    :param zip_path: Path to the ZIP file containing GRIB files.
    :return: A list of xarray datasets created from the GRIB files.
    """
    # extract files from zip file
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]

    # Extract all files
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Loop through extracted files and open GRIB files
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            if file.endswith('.grib') or file.endswith('.grb'):
                grib_path = os.path.join(root, file)

                # Rename GRIB file using ZIP file name
                new_grib_name = f"{zip_name}.grib"
                new_grib_path = os.path.join(root, new_grib_name)

                # Avoid overwriting if already renamed
                if grib_path != new_grib_path:
                    os.rename(grib_path, new_grib_path)
                    grib_path = new_grib_path  # update path to the renamed file

                dataset = read_grib_file(grib_path, step_type=step_type)

    return dataset


def convert_dataset_units(ds):
    """
    Convert ERA5-Land variables in a Dataset to commonly used units.

    Parameters:
        ds (xarray.Dataset): The input dataset with ERA5-Land variables.
        output_md_path (str): File path to save the Markdown table.

    Returns:
        ds_converted (xarray.Dataset): Dataset with converted units.
        output_md_path (str): Path to the Markdown file.
    """
    conversions = {
        "t2m": {"description": "2m temperature", "original_unit": "K", "factor": 1, "offset": -273.15, "new_unit": "degC"},
        "tp": {"description": "Total precipitation", "original_unit": "m/day", "factor": 1000, "offset": 0, "new_unit": "mm/day"},
        "ro": {"description": "Runoff", "original_unit": "m/day", "factor": 1000, "offset": 0, "new_unit": "mm/day"},
        "sro": {"description": "Surface runoff", "original_unit": "m/day", "factor": 1000, "offset": 0, "new_unit": "mm/day"},
        "pev": {"description": "Potential evaporation", "original_unit": "m/day", "factor": 1000, "offset": 0, "new_unit": "mm/day"},
        "e": {"description": "Total evaporation", "original_unit": "m/day", "factor": 1000, "offset": 0, "new_unit": "mm/day"},
        "sd": {"description": "Snow depth water equivalent", "original_unit": "m", "factor": 1000, "offset": 0, "new_unit": "mm"}
    }

    ds_converted = ds.copy()
    for var, conv in conversions.items():
        if var in ds_converted.data_vars:
            ds_converted[var] = ds_converted[var] * conv["factor"] + conv["offset"]
            ds_converted[var].attrs["units"] = conv["new_unit"]
    return ds_converted


def plot_mean_map(ds, var, folder=None):
    """
    Plot the mean of a variable over time.
    :param ds: xarray dataset
    :param var: variable name to plot
    :return: None
    """
    mean_field = ds[var].mean(dim='time')
    plt.figure(figsize=(8, 6))
    mean_field.plot()
    plt.title(f"Mean {var.upper()} Over Time")
    plt.xlabel("lon_name")
    plt.ylabel("lat_name")
    if folder:
        plt.savefig(os.path.join(folder, f"mean_{var}.jpg"), dpi=300)
        plt.close()

    else:
        plt.show()


def plot_monthly_climatology_grid(ds, var, folder=None):
    """
    Plot the monthly climatology of a variable in a grid format.
    :param ds:
    :param var:
    :return:
    """
    # Group by month and average over years
    monthly_clim = ds[var].groupby('time.month').mean(dim='time')

    # Calculate shared color scale
    vmin = monthly_clim.min().item()
    vmax = monthly_clim.max().item()

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
    axes = axes.flatten()

    for i in range(12):
        ax = axes[i]
        im = monthly_clim.sel(month=i+1).plot(
            ax=ax,
            add_colorbar=False,
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(f"Month {i+1}")
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Add a single shared colorbar for all axes
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.9)
    cbar.set_label(ds[var].attrs.get('units', var))

    fig.suptitle(f"Interannual Monthly Mean of {var.upper()}", fontsize=16)
    #plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Leave space for suptitle and colorbar
    if folder:
        plt.savefig(os.path.join(folder, f'monthly_climatology_{var}.jpg'), dpi=300)
        plt.close()
    else:
        plt.show()


def plot_spatial_mean_timeseries_all_vars(ds, lat_name='latitude', lon_name='longitude', folder=None):
    """
    Plot the spatial mean time series for all variables in the dataset.
    :param ds:
    :return:
    """
    plt.figure(figsize=(12, 6))

    for var in ds.data_vars:
        # Calculate spatial average over lat/lon
        spatial_mean = ds[var].mean(dim=[lat_name, lon_name])

        # Get units if available
        units = ds[var].attrs.get('units', '')
        label = f"{var.upper()} ({units})" if units else var.upper()

        # Plot it
        plt.plot(ds.time, spatial_mean, label=label)

    plt.title("Spatial Mean Time Series for All Variables")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if folder:
        plt.savefig(os.path.join(folder, "spatial_mean_timeseries_all_vars.png.jpg"), dpi=300)
        plt.close()
    else:
        plt.show()


def calculate_resolution_netcdf(dataset, lon_name='X', lat_name='Y'):
    """
    Calculate spatial resolution (in degrees and km) from a netCDF-like xarray.Dataset.

    Parameters:
        dataset: xarray.Dataset
            The dataset to analyze
        lon_name: str
            Name of the lon_name coordinate
        lat_name: str
            Name of the lat_name coordinate

    Returns:
        dict: resolution information including degrees and approximate km
    """
    # Spatial resolution in degrees
    dx_deg = float(dataset[lon_name][1] - dataset[lon_name][0])
    dy_deg = float(dataset[lat_name][1] - dataset[lat_name][0])

    # Approximate midpoint lat_name for scaling lon_name to km
    mid_lat = float(dataset[lat_name].mean())


    # Constants
    km_per_deg_lat = 111  # approximate
    km_per_deg_lon = 111 * np.cos(np.deg2rad(mid_lat))

    # Convert to km
    dx_km = dx_deg * km_per_deg_lon
    dy_km = dy_deg * km_per_deg_lat

    print(f"Spatial resolution: {dx_deg}° lon x {dy_deg}° lat")
    print(f"Approximate spatial resolution:")
    print(f"{dx_km:.2f} km (lon_name) x {dy_km:.2f} km (lat_name) at {mid_lat:.2f}° lat")

    # Temporal resolution (assuming consistent intervals)
    dt = dataset.time[1] - dataset.time[0]
    temporal_resolution_days = pd.to_timedelta(dt.values).days
    print(f"Temporal resolution: {temporal_resolution_days} days")

    return {
        "dx_deg": dx_deg,
        "dy_deg": dy_deg,
        "dx_km": dx_km,
        "dy_km": dy_km,
        "mid_lat": mid_lat
    }


def check_spatial_overlap(data_clim, geometry, lon_name='X', lat_name='Y', mode='total'):
    """
    Check if a shapefile or a single geometry overlaps with a climate DataArray.

    :param data_clim: xarray DataArray
    :param geometry: GeoDataFrame (for mode='total') or shapely geometry (for mode='row')
    :param lon_name: str, name of lon_name coordinate in DataArray
    :param lat_name: str, name of lat_name coordinate in DataArray
    :param mode: 'total' for full shapefile, 'row' for a single geometry
    :return: bool, True if overlap exists
    """
    # Get bounds from DataArray
    da_lon_min = float(data_clim[lon_name].min())
    da_lon_max = float(data_clim[lon_name].max())
    da_lat_min = float(data_clim[lat_name].min())
    da_lat_max = float(data_clim[lat_name].max())

    print("\n📦 DataArray bounds:")
    print(f"  lon_name ({lon_name}): {da_lon_min:.4f} to {da_lon_max:.4f}")
    print(f"  lat_name ({lat_name}):  {da_lat_min:.4f} to {da_lat_max:.4f}")

    # Get geometry bounds
    if mode == 'total':
        bounds = geometry.total_bounds  # (minx, miny, maxx, maxy)
    elif mode == 'row':
        bounds = geometry.bounds  # shapely object
    else:
        raise ValueError("mode must be either 'total' or 'row'")

    g_lon_min, g_lat_min, g_lon_max, g_lat_max = bounds

    print(f"  lon_name: {g_lon_min:.4f} to {g_lon_max:.4f}")
    print(f"  lat_name:  {g_lat_min:.4f} to {g_lat_max:.4f}")

    # Check overlap
    overlap_lon = (g_lon_min <= da_lon_max) and (g_lon_max >= da_lon_min)
    overlap_lat = (g_lat_min <= da_lat_max) and (g_lat_max >= da_lat_min)

    if overlap_lon and overlap_lat:
        print("\n✅ Overlap detected.\n")
        return True
    else:
        print("\n❌ No overlap detected.\n")
        return False


def calculate_spatial_mean_annual(data_climatic, gdf_regions, lat_name='Y', lon_name='X'):
    """
    Calculate spatial and yearly mean of a climate variable per region.

    :param data_climatic: xarray.DataArray
        A 3D DataArray (time, lat, lon) containing the climate variable.
    :param gdf_regions: GeoDataFrame
        A GeoDataFrame with polygons (geometry) and unique ID per region (e.g., "region", "station", etc.)
    :param var_name: str
        Name to assign to the climate variable in output
    :return: DataFrame
        Multi-index DataFrame with time and region ID, showing the monthly and annual mean
    """

    # Ensure spatial metadata is set
    data_climatic = data_climatic.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)
    data_climatic = data_climatic.rio.write_crs("EPSG:4326", inplace=False)

    if not gdf_regions.crs:
        gdf_regions = gdf_regions.set_crs("EPSG:4326")
    else:
        gdf_regions = gdf_regions.to_crs("EPSG:4326")


    check_spatial_overlap(data_climatic, gdf_regions, lon_name=lon_name, lat_name=lat_name, mode='total')
    results = []

    for i, row in gdf_regions.iterrows():
        region_id = row.get("station", f"region_{i}")
        print(f'Region {region_id}')
        if check_spatial_overlap(data_climatic, row.geometry, mode='row', lon_name=lon_name, lat_name=lat_name):
            # proceed with clipping
            clipped = data_climatic.rio.clip([row.geometry.__geo_interface__], gdf_regions.crs, all_touched=True)
        else:
            print(f'No overlap for region: {region_id}')

        # Mean over space (lat, lon) -> results in time series
        regional_mean = clipped.mean(dim=[lat_name, lon_name])
        regional_mean = regional_mean.expand_dims({"region": [region_id]})
        results.append(regional_mean)

    # Combine into one dataset (dims: time, region)
    combined = xr.concat(results, dim="region")

    # Convert to DataFrame
    df = combined.to_dataframe().reset_index()

    return df

def convert_to_yearly_mm_year(df, var_name="Runoff", unit_init="mm/day"):
    """
    Convert monthly values in mm/day to yearly total in mm/year.

    :param df: DataFrame with columns ['time', 'region', var_name] where time is datetime-like.
    :param var_name: Name of the variable column (e.g., 'Runoff').
    :return: DataFrame with yearly runoff totals in mm/year per region.
    """

    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    if unit_init == "mm/day":
        df["days_in_month"] = df["time"].dt.days_in_month
        # mm/month = mm/day * days in month
        df["monthly_total_mm"] = df[var_name] * df["days_in_month"]
    elif unit_init == "mm/month":
        df["monthly_total_mm"] = df[var_name]
    else:
        raise ValueError("unit_init must be either 'mm/day' or 'mm/month'")

    # Sum over year per region
    df_yearly = (
        df.groupby(["year", "region"])["monthly_total_mm"]
        .sum()
        .reset_index()
        .rename(columns={"monthly_total_mm": f"{var_name}_mm_per_year"})
    )

    return df_yearly


def map_grdc_stationbasins_and_subregions(folder, file_stationbasins, file_subregions=None):
    # Load your files
    stationbasins = gpd.read_file(file_stationbasins)
    if file_subregions is not None:
        subregions = gpd.read_file(file_subregions)

        # Ensure CRS is consistent
        if stationbasins.crs != subregions.crs:
            stationbasins = stationbasins.to_crs(subregions.crs)

    # Calculate map center
    map_center = stationbasins.geometry.unary_union.centroid.coords[0][::-1]

    # Initialize folium map
    m = folium.Map(location=map_center, zoom_start=6, tiles='CartoDB positron')
    if file_subregions is not None:
        # Add subregions layer
        folium.GeoJson(
            subregions,
            name="Subregions",
            style_function=lambda feature: {
                "color": "blue",
                "weight": 1.5,
                "fillOpacity": 0.1,
            },
            tooltip=GeoJsonTooltip(fields=subregions.columns[:2].tolist(), aliases=["SUBREGNAME", "RIVERBASIN"])
            # customize fields
        ).add_to(m)

    # Add stations (points or polygons) layer
    folium.GeoJson(
        stationbasins,
        name="Station Catchments",
        style_function=lambda feature: {
            "color": "red",
            "weight": 1,
            "fillOpacity": 0.3,
        },
        tooltip=GeoJsonTooltip(
            fields=["station", "river", "area"],
            aliases=["Station ID", "River", "Area (km²)"],
            sticky=True
        )
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save and display
    m.save(os.path.join(folder, "map_stations_and_subregions.html"))
    print("Map saved to map_stations_and_subregions.html")