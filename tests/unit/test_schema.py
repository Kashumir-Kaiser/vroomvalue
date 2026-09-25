import pytest
from pydantic import ValidationError
from apps.api.app.schemas import VehicleInput

BASE = dict(make="Toyota",model="Camry",year=2022,fuel_type="Petrol",transmission="Automatic",engine_size=2.5,mileage=42000,horsepower=203.0,torque=184.0,owners=1,accident_history=0,service_history="Full Service",color="White",body_type="Sedan",drivetrain="FWD",fuel_efficiency=32.0,location="TX")

def test_engine_size_tenth_is_valid():
    assert VehicleInput(**BASE).engine_size == 2.5

def test_engine_size_two_decimals_is_rejected():
    with pytest.raises(ValidationError, match="one decimal place"):
        VehicleInput(**{**BASE, "engine_size": 2.55})