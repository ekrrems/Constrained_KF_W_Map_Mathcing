# Key numbers

- Nominal run: `20261110_100739_osm`
- OSM outage run: `20261110_100846_osm_outage`
- Orthophoto outage run: `20261110_100949_orthophoto_outage`

## Nominal experiment: distance to the digitized evaluation reference (m)

| Method | Mean | Median | RMSE | 95th perc. | RMSE 95% CI | Accepted |
|---|---|---|---|---|---|---|
| Recorded GNSS positions | 0.0794 | 0.0419 | 0.1182 | 0.2795 | [0.0884, 0.1465] | - |
| GNSS/IMU ESKF | 0.0710 | 0.0655 | 0.0816 | 0.1485 | [0.0726, 0.0909] | 0 |
| Two-step output projection | 0.3539 | 0.2348 | 0.4689 | 0.9697 | [0.3676, 0.5556] | 777 |
| Map-heading observation | 0.0716 | 0.0658 | 0.0821 | 0.1492 | [0.0729, 0.0916] | 773 |
| Line-normal constraint | 0.0585 | 0.0500 | 0.0731 | 0.1381 | [0.0633, 0.0832] | 720 |
| Combined lateral position and heading | 0.0546 | 0.0451 | 0.0709 | 0.1384 | [0.0597, 0.0818] | 760 |

## Nominal experiment: heading error (deg)

| Method | Mean abs. | RMSE | 95th perc. | Signed mean |
|---|---|---|---|---|
| GNSS/IMU ESKF | 1.23 | 1.89 | 3.62 | 0.79 |
| Map-heading observation | 0.33 | 0.49 | 1.02 | 0.00 |
| Line-normal constraint | 1.28 | 1.92 | 3.73 | 0.91 |
| Combined lateral position and heading | 0.34 | 0.50 | 1.00 | -0.01 |

## Outage experiment, OSM map: errors against the withheld GNSS

| Method | Window (s) | Horizontal RMSE | Along RMSE | Cross RMSE | Heading RMSE (deg) | Accepted |
|---|---|---|---|---|---|---|
| IMU only (GNSS withheld) | all | 76.95 | 70.64 | 35.30 | 2.37 | 0 |
| IMU only (GNSS withheld) | 101-160 | 61.09 | 9.05 | 60.41 | 1.83 | 0 |
| IMU only (GNSS withheld) | 220-279 | 17.25 | 11.38 | 12.96 | 0.57 | 0 |
| IMU only (GNSS withheld) | 341-399 | 137.70 | 142.03 | 19.43 | 0.87 | 0 |
| IMU only (GNSS withheld) | 460-519 | 49.51 | 25.44 | 42.79 | 4.71 | 0 |
| IMU only (GNSS withheld) | 580-639 | 66.23 | 64.61 | 13.83 | 0.66 | 0 |
| Map-heading observation | all | 73.93 | 64.91 | 38.89 | 1.33 | 211 |
| Map-heading observation | 101-160 | 65.66 | 7.92 | 65.17 | 0.30 | 25 |
| Map-heading observation | 220-279 | 16.97 | 10.55 | 13.28 | 0.24 | 60 |
| Map-heading observation | 341-399 | 131.97 | 135.43 | 15.01 | 2.12 | 46 |
| Map-heading observation | 460-519 | 55.86 | 26.40 | 49.59 | 0.35 | 33 |
| Map-heading observation | 580-639 | 48.90 | 45.69 | 20.81 | 2.06 | 47 |
| Line-normal constraint | all | 38.64 | 38.64 | 0.55 | 1.82 | 299 |
| Line-normal constraint | 101-160 | 12.52 | 12.52 | 0.19 | 2.28 | 60 |
| Line-normal constraint | 220-279 | 21.24 | 21.23 | 0.64 | 1.30 | 60 |
| Line-normal constraint | 341-399 | 81.65 | 81.65 | 0.87 | 1.10 | 59 |
| Line-normal constraint | 460-519 | 15.78 | 15.78 | 0.30 | 2.68 | 60 |
| Line-normal constraint | 580-639 | 5.30 | 5.28 | 0.46 | 0.95 | 60 |
| Combined lateral position and heading | all | 46.13 | 45.93 | 7.30 | 1.06 | 284 |
| Combined lateral position and heading | 101-160 | 50.68 | 50.68 | 0.39 | 0.68 | 59 |
| Combined lateral position and heading | 220-279 | 41.48 | 41.48 | 0.58 | 0.22 | 60 |
| Combined lateral position and heading | 341-399 | 77.37 | 76.78 | 16.42 | 1.03 | 45 |
| Combined lateral position and heading | 460-519 | 19.82 | 19.82 | 0.29 | 1.85 | 60 |
| Combined lateral position and heading | 580-639 | 5.90 | 5.88 | 0.47 | 0.75 | 60 |

## Outage experiment, Orthophoto map: errors against the withheld GNSS

| Method | Window (s) | Horizontal RMSE | Along RMSE | Cross RMSE | Heading RMSE (deg) | Accepted |
|---|---|---|---|---|---|---|
| IMU only (GNSS withheld) | all | 76.95 | 70.64 | 35.30 | 2.37 | 0 |
| IMU only (GNSS withheld) | 101-160 | 61.09 | 9.05 | 60.41 | 1.83 | 0 |
| IMU only (GNSS withheld) | 220-279 | 17.25 | 11.38 | 12.96 | 0.57 | 0 |
| IMU only (GNSS withheld) | 341-399 | 137.70 | 142.03 | 19.43 | 0.87 | 0 |
| IMU only (GNSS withheld) | 460-519 | 49.51 | 25.44 | 42.79 | 4.71 | 0 |
| IMU only (GNSS withheld) | 580-639 | 66.23 | 64.61 | 13.83 | 0.66 | 0 |
| Map-heading observation | all | 74.39 | 65.52 | 40.80 | 1.79 | 197 |
| Map-heading observation | 101-160 | 65.80 | 10.92 | 64.88 | 0.36 | 25 |
| Map-heading observation | 220-279 | 14.51 | 11.26 | 9.15 | 0.34 | 60 |
| Map-heading observation | 341-399 | 135.16 | 140.19 | 25.43 | 0.35 | 44 |
| Map-heading observation | 460-519 | 50.77 | 18.50 | 47.52 | 0.89 | 30 |
| Map-heading observation | 580-639 | 49.93 | 38.43 | 33.29 | 3.50 | 38 |
| Line-normal constraint | all | 81.50 | 82.53 | 0.57 | 4.41 | 272 |
| Line-normal constraint | 101-160 | 21.94 | 21.94 | 0.14 | 2.07 | 57 |
| Line-normal constraint | 220-279 | 44.01 | 44.01 | 0.13 | 0.81 | 60 |
| Line-normal constraint | 341-399 | 161.37 | 163.91 | 0.99 | 3.39 | 56 |
| Line-normal constraint | 460-519 | 57.82 | 58.02 | 0.30 | 8.35 | 54 |
| Line-normal constraint | 580-639 | 41.59 | 41.66 | 0.72 | 2.89 | 45 |
| Combined lateral position and heading | all | 65.80 | 67.75 | 10.36 | 2.14 | 239 |
| Combined lateral position and heading | 101-160 | 43.02 | 43.02 | 0.16 | 0.33 | 54 |
| Combined lateral position and heading | 220-279 | 37.44 | 37.44 | 0.15 | 0.32 | 60 |
| Combined lateral position and heading | 341-399 | 123.50 | 128.67 | 23.10 | 0.47 | 44 |
| Combined lateral position and heading | 460-519 | 42.18 | 42.15 | 1.01 | 4.19 | 43 |
| Combined lateral position and heading | 580-639 | 39.32 | 39.55 | 3.00 | 1.83 | 38 |

## Nominal experiment by track geometry

| Class | Epochs | Cross-track RMSE (cm): GNSS / ESKF / heading / line normal / combined | Heading RMSE (deg): ESKF / heading / line normal / combined |
|---|---|---|---|
| straight | 338 | 4.1 / 6.2 / 6.3 / 5.3 / 5.0 | 1.57 / 0.45 / 1.60 / 0.47 |
| curve | 228 | 17.4 / 10.0 / 10.1 / 9.2 / 9.2 | 2.27 / 0.54 / 2.31 / 0.55 |
| standstill | 111 | 13.4 / 9.0 / 9.1 / 8.0 / 7.4 | nan / nan / nan / nan |
