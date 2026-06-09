from fastapi import FastAPI

from main import (
    process_all_images
)

app = FastAPI()

@app.get("/")
def root():

    return {
        "status": "online"
    }

@app.post("/process-all")
def process_all():

    result = process_all_images()

    return {

        "status":
            "success",

        "result":
            result
    }