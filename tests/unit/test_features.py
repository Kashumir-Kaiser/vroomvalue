import math

import pandas as pd

from ml.features.build import build_features


def test_feature_math_and_missing_indicators():
    row = pd.DataFrame(
        [
            {
                "Make": "Toyota",
                "Model": "Camry",
                "Year": 2022,
                "Fuel_Type": "Petrol",
                "Transmission": None,
                "Engine_Size": 2.5,
                "Mileage": 42000,
                "Horsepower": 203.0,
                "Torque": 184.0,
                "Owners": 1,
                "Accident_History": None,
                "Service_History": "Full Service",
                "Color": "White",
                "Body_Type": "Sedan",
                "Drivetrain": "FWD",
                "Fuel_Efficiency": 32.0,
                "Location": "TX",
            }
        ]
    )
    features = build_features(row, 2026)

    assert features.loc[0, "vehicle_age"] == 4
    assert features.loc[0, "mileage_per_year"] == 10500
    assert math.isclose(features.loc[0, "log_mileage"], math.log1p(42000))
    assert features.loc[0, "Transmission"] == "MISSING"
    assert features.loc[0, "Transmission_missing"] == 1
    assert features.loc[0, "Accident_History_missing"] == 1
