from backend.main import (
    app
)

from services.mask_generation import (
    process_all_images
)

from services.image_capture import (
    app as image_capture_app
)

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

app.mount(
    "/capture",
    image_capture_app
)