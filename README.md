

# Data Summary





# Global Runoff Data Centre (GRDC)

> The Global Runoff Data Centre (GRDC) is an international data centre operating under the auspices of the World
> Meteorological Organization (WMO). Established in 1988 to support the research on global and climate change and
> integrated water resources management, it holds the most substantive collection of quality assured river discharge
> data on global scale.

There is no API available for the GRDC, but the data can be downloaded from the [GRDC Data PORTAL](https://portal.grdc.bafg.de/applications/public.html?publicuser=PublicUser#dataDownload/StationCatalogue).

Unlike ERA5 and GRUN, the GRDC data is not available in a gridded format.
The discharge (streamflow) data is available at station level.

Check out the [FAQ](https://grdc.bafg.de/help/faq/) to define the way to download the data.

# ERA5-Land

## ERA5-Land Monthly Averages

- Covers the period from **January 1950 to 2-3 months before the present**
- ERA5-Land runs at enhanced resolution (9 km) - a regular latitude/longitude grid of 0.1°x0.1° via the CDS catalogue

This document summarizes the units for each variable available in the ERA5-Land **monthly_averaged_reanalysis** dataset via the CDS API.
> Check the [Documentation](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=overview)

---

### 📏 Units for ERA5-Land Monthly Averaged Variables

| Variable                         | Unit     | Description |
|----------------------------------|----------|-------------|
| **2m_temperature**               | Kelvin (K) | Air temperature at 2 meters above the surface. To convert to Celsius, subtract 273.15. |
| **total_precipitation**          | m/day    | Total precipitation accumulated over the day, expressed in meters of water equivalent per day. Multiply by 1000 to convert to mm/day. Multiply by the number of days in the month for total monthly precipitation. |
| **surface_runoff**               | m/day    | Surface runoff accumulated over the day, in meters of water equivalent. Multiply by 1000 for mm/day. Multiply by number of days in the month for monthly total. |
| **snow_depth_water_equivalent**  | m        | Instantaneous depth of snow in meters of water equivalent. |
| **potential_evaporation**        | m/day    | Potential evaporation per day in meters. Multiply by 1000 for mm/day. Multiply by number of days in the month for monthly total. |
| **total_evaporation**            | m/day    | Actual evapotranspiration per day in meters. Multiply by 1000 for mm/day. Multiply by number of days in the month for monthly total. |

---

### 🔄 Notes on Data Interpretation

- **Accumulated Variables**:
    - Values are **daily means in meters/day**.
    - To compute **total for a month**:
      `monthly_total = daily_mean × number_of_days_in_month`
    - Example:
      `0.004 m/day × 30 days = 0.12 m`

- **Instantaneous Variables**:
    - Do **not** need accumulation over time.

---

> For more details, refer to the [ERA5-Land documentation](https://confluence.ecmwf.int/display/CKB/ERA5-Land:+data+documentation).


### Extracting Data from file

#### Understanding `stepType`: Handling Instantaneous vs Accumulated Variables in ERA5-Land Monthly Averages

In ERA5-Land monthly reanalysis data, variables are processed differently depending on whether they are **instantaneous
** or **accumulated**. This affects how they are averaged and stored in GRIB files.

---

##### 🔄 Key `stepType` Values

| `stepType`  | Meaning                                                                                                                                                               |
|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`avgid`** | Monthly **average of hourly data** that was originally **instantaneous** (e.g., 2m temperature).                                                                      |
| **`avgas`** | Monthly **average of hourly data** that was originally **accumulated**, then **scaled** to a rate (e.g., total precipitation → precipitation rate → monthly average). |

---

##### 🧠 Why It Matters

- These different `stepType`s **must be handled separately** when reading GRIB files using tools like `cfgrib`.
- Attempting to read a GRIB file with mixed `stepType`s without filtering will cause errors.

---

##### ✅ How to Extract Data

Use `cfgrib` with `filter_by_keys` to open each type separately:

```python
import xarray as xr

# For instantaneous variables (e.g., 2m_temperature)
ds_instant = xr.open_dataset("data.grib", engine="cfgrib",
                             backend_kwargs={"filter_by_keys": {"stepType": "avgid"}})

# For accumulated variables (e.g., total_precipitation)
ds_accum = xr.open_dataset("data.grib", engine="cfgrib",
                           backend_kwargs={"filter_by_keys": {"stepType": "avgas"}})
```

You can then merge the datasets if they share compatible dimensions.

---

> For more, see the [CDS GRIB-to-netCDF changes](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation).

# GRUN

The dataset contains a gridded global reconstruction of monthly runoff timeseries.
The data are at monthly resolution covering the period 1902-2014 on a **0.5 degrees (WGS84)
grid in units of mm/day**
The data are provided in a NetCDFv4 file, that is downloadable from the GRUN repository.


[GRUN Dataset](https://figshare.com/articles/dataset/GRUN_Global_Runoff_Reconstruction/9228176)
[Publication](https://essd.copernicus.org/articles/11/1655/2019/)

> In-situ streamflow observations from the GSIM dataset are used to train a machine learning algorithm
> that predicts monthly runoff rates based on antecedent precipitation and temperature from the
> Global Soil Wetness Project Phase 3 (GSWP3) meteorological forcing dataset.
