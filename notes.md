# Dataset

```python
# Convert time to unix timestamp
forecast["ds"] = forecast["ds"].astype(int) / 10**9
```

```python
def parse_dataset(dataset):
    df = pandas.DataFrame(dataset)
    print(df.dtypes)
    print(df.head())

    # TODO: those transforms should not be necessary in the future
    df = df.dropna()

    # Test for unix timestamp
    try:
        int(df["ds"][0])
    except ValueError:
        df["ds"] = pandas.to_datetime(df["ds"], utc=True, errors="coerce")
    else:
        df["ds"] = pandas.to_datetime(df["ds"], unit="s", utc=True, errors="coerce")
    df["ds"] = df["ds"].dt.tz_localize(None)
    df["y"] = pandas.to_numeric(df["y"])

    return df
```

# Holidays

```python
if "holidays" in configuration and len(configuration["holidays"]) > 0:
    print("Add country holidays for", configuration["holidays"])
    model = model.add_country_holidays(configuration["holidays"])
```

# Lagged regressors

```python
for lagged_regressor in configuration.get("laggedRegressors", []):
    name = lagged_regressor["dataColumnRef"]
    print("Add lagged regressor", name)
    df[name] = pandas.to_numeric(df[name])
    n_lags = lagged_regressor.get("n_lags", "auto")
    if n_lags != "auto" and n_lags != "scalar":
        n_lags = int(n_lags)
    model = model.add_lagged_regressor(
        name,
        n_lags,
        lagged_regressor.get("regularization", None),
        lagged_regressor.get("normalize", "auto"),
    )
```
