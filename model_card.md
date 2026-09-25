# VroomValue model card

## Purpose

VroomValue estimates a used vehicle's **Selling_Price** from the supplied 17 predictors. It is decision assistance, not an appraisal guarantee. Displayed amounts use **USD**, mileage uses **miles**, torque uses **lb-ft**, and fuel efficiency uses **MPG / MPGe for electric vehicles** because the source data does not provide explicit unit columns.

## Data

- Source: `data/automobile_dataset.csv` supplied with this project.
- Rows: 5,500.
- Target: `Selling_Price`.
- Reference year frozen into the artifact: 2026.
- `Engine_Size` contract: 0.0–5.7 and at most one decimal place (0.1 increments); 0.0 is valid for electric vehicles.
- Missing values are preserved for numeric CatBoost inputs and represented as explicit `MISSING` categories for categorical fields, with missing indicators for each partially missing field.

## Evaluation design

Duplicate fingerprints are grouped before splitting. Ten percent is a locked interpolation holdout and ten percent is a dedicated calibration set. Raw-price and `log1p(price)` objectives are compared using 5-fold grouped CV on the development rows. The calibration set is not used to choose the point model.

## Current local training result

- Selected objective: **log1p**
- Grouped CV MAE — raw: **$1,063.72**; log1p: **$1,024.37**
- Naive Make-Model-Year baseline MAE: **$2,143.13**
- CatBoost holdout MAE: **$1,116.91**
- MAE improvement over baseline: **47.9%**
- Holdout RMSE: **$1,845.98**
- Holdout R²: **0.9800**
- Holdout WAPE: **9.07%**
- Nominal 80% interval holdout coverage: **81.09%**

### Cold-start Make-Model stress view

- Held-out rows: **1,057**
- Held-out Make-Model groups: **Audi|A4, Audi|Q5, Chevrolet|Equinox, Honda|Civic, Honda|Pilot, Mercedes-Benz|C-Class, Toyota|RAV4, Volkswagen|Atlas**
- Cold-start MAE: **$3,890.55**
- Cold-start WAPE: **32.91%**

Intervals use split-conformal absolute residuals with four prediction-price buckets (Mondrian calibration). This keeps the uncertainty width more appropriate across low- and high-price vehicles. Bucket half-widths are: $117, $1,388, $1,475, $2,758.

## Limitations

The dataset has no listing/sale date or source identifier, so this model cannot measure market-time drift or source bias. `Selling_Price` is treated as the target exactly as named in the supplied data; the source does not establish whether it is a listing price or a verified transaction price. Units/currency are assumptions and are stated in the UI. SHAP explanations describe model behavior, not causal price effects.
