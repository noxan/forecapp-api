def parse_configuration(configuration):
    epochs = configuration.get("training", {}).get("epochs", None)
    # Map "auto" for epochs back to None... else it all fails
    if epochs == "auto":
        epochs = None
    elif epochs:
        epochs = int(epochs)

    forecasts = int(configuration.get("forecasts", 1))

    ar_config = configuration.get("autoRegression", {})
    ar_lags = int(ar_config.get("lags", 0))
    ar_regularization = int(ar_config.get("regularization", 0))

    return epochs, forecasts, ar_lags, ar_regularization
