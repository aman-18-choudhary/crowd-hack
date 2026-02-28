from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

# Store latest AI data here
latest_data = {}

class CrowdData(BaseModel):
    current_density: float
    predicted_density: float
    current_flow: float
    predicted_flow: float
    risk: bool


@app.post("/update")
def update(data: CrowdData):
    global latest_data
    latest_data = {
        **data.dict(),
        "timestamp": datetime.utcnow()
    }
    return {"message": "Updated successfully"}


@app.get("/status")
def status():
    return latest_data