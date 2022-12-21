import pandas


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
