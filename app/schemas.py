from pydantic import BaseModel, Field
from typing import Optional

class PredictRequest(BaseModel):
    city: str = Field(..., examples=["Montreal"])
    no2_ugm3: Optional[float] = Field(None, description="NO2 en µg/m³ (si tu le fournis toi-même)")
    temp_c: Optional[float] = Field(None, description="Température en °C (si tu le fournis)")
    rh: Optional[float] = Field(None, description="Humidité relative en % (si tu le fournis)")

class PredictResponse(BaseModel):
    city: str
    ds: str
    yhat1: float
    inputs: dict
