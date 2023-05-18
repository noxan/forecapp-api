import pandas as pd
from app.config import EventItem


def create_event_dataframe(event_name: str, event: EventItem):
    df = pd.DataFrame(
        {
            "event": event_name,
            "ds": pd.to_datetime(event.dates),
        }
    )
    return df
