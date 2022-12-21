def parse_configuration(configuration):
    epochs = configuration.get("training", {}).get("epochs", None)
    # Map "auto" for epochs back to None... else it all fails
    if epochs == "auto":
        epochs = None
    elif epochs:
        epochs = int(epochs)

    forecasts = int(configuration.get("forecasts", 1))

    return epochs, forecasts
