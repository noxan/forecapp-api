from fastapi import FastAPI
from pydantic import BaseModel


class ModelConfiguration(BaseModel):
    training: bool = False


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/prediction")
def prediction(configuration: ModelConfiguration):
    print(configuration)
    return {"status": "ok", "configuration": configuration}
