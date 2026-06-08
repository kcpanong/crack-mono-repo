from fastapi import FastAPI

from main import (
    process_all_crops
)

app = FastAPI()

@app.get("/")
def root():

    return {
        "status": "online"
    }

@app.post("/process-all")
def process_all():

    result = process_all_crops()

    return {

        "status":
            "success",

        "result":
            result
    }