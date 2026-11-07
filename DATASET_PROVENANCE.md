# Dataset provenance audit

## Local recording: 2025-01-30, set 13

- GNSS epochs: 780 at 1 Hz
- IMU samples: 234,159 at 300 Hz
- duration: approximately 779 s (12.98 min)
- GNSS trajectory length: approximately 8.93 km
- maximum one-second GNSS displacement: approximately 21.64 m/s (77.9 km/h)
- UTM zone: 32N
- receiver modes: 779 epochs with mode 4 and one epoch with mode 5
- digitized reference: 8,150 nodes and approximately 8.15 km polyline length

## Relation to Frolow, Zhang, and Schwieger (2026)

The paper, *Precise Train Positioning With Unscented Kalman Filter and Low-Cost
Sensors*, describes the ConMoRAIL measurement campaign on an approximately
9 km railway route with seven stations, speeds up to 80 km/h, 12–13 minute
drives, RTK GNSS, and a 300 Hz IMU. Set 13 matches all of these independent
characteristics and is therefore very likely from the same measurement
campaign and dataset family.

The public paper does not expose the local filename `set_13` or a drive-to-file
table. The exact statement that this file is one of the paper's evaluated
drives cannot be proven from the publication alone. Before the thesis calls it
the identical evaluation drive, obtain confirmation from the supervisors or a
campaign metadata table. Until then, use the wording “a recording from the
same ConMoRAIL measurement campaign” rather than “the dataset used in the
paper.”

There is also a reference-data difference that must not be hidden. The paper
describes a true-orthophoto trajectory digitized at roughly 5 m spacing,
whereas the supplied `Ref_Intermetric_UTM.txt` contains about 8,150 nodes over
8.15 km, roughly 1 m spacing. It may be a densified or separately digitized
version, but its origin cannot be inferred from the filename alone.

The paper compares an ESKF with a UKF. It reports comparable results for both;
the adaptive stochastic observation model improves both by approximately
2–3 cm to an RMS of about 0.26 m. The UKF is therefore not established as the
more accurate method for this campaign.

## Observation uncertainties used by this implementation

The railway reader now follows the paper's selected horizontal position
standard deviations: 0.10 m for RTK fixed, 0.50 m for RTK float, and 5.0 m for
the standalone/fallback case. The covariance used by the update therefore
changes epoch by epoch when the receiver solution class changes. This is the
meaning of the paper's *adaptive stochastic model*: the observation covariance
is selected from the RTK-fixed, RTK-float, and standalone classes instead of
remaining constant. It is not an online learning algorithm. This is a
documented starting stochastic model, not proof that the filter is
statistically consistent. The mapping currently assumes that the exported
dataset codes mean mode 4 = RTK fixed and mode 5 = RTK float; that code mapping
must be confirmed from the Intermetric/u-blox export metadata before final
reporting. Both codes and all standard deviations are command-line parameters.

The paper derives GNSS velocity from GNSS positions and investigates temporal
and spatial correlation models. Those derived-velocity observations are not
used in the thesis configuration because the required correlation model is not
available here. The railway baseline uses a three-dimensional GNSS position
observation. Velocity remains part of the filter state and is corrected
indirectly through its covariance with position.

## IMU streams

The measurement platform in the paper carried two ASC IMU 7 units, one on the
car body and one on the bogie. The paper explicitly states that only the
car-body IMU was used by the positioning filters; the bogie IMU was reserved
for train-dynamics work because its vibration level was too high for this
localization experiment. The supplied MATLAB pair exposes a single `sensor`
IMU stream. The thesis should therefore describe the implemented input as one
300 Hz IMU stream and must not claim that two IMUs were fused.

## Evaluation reference

`Ref_Intermetric_UTM.txt` is the primary digitized thesis reference used by the
runner. It supplies projected reference nodes and is never used in the filter
update or map association. Its perpendicular-error sign is defined by the
stored point order: positive means left of the oriented reference line.

The paper's quoted 20 cm value is the true-orthophoto pixel resolution, not a
constant trajectory error or an offset to subtract from filter results.

These railway uncertainty values are not transferred to KITTI. KITTI uses the
per-packet OXTS position-accuracy field for its position covariance. The OXTS
velocity fields are part of the RT3003 fused GPS/IMU navigation solution and
are retained only for diagnostics, not used as an independent GNSS velocity
update in the IMU-driven ESKF.

Primary publication: Rudolf Frolow, Li Zhang, and Volker Schwieger, “Precise
Train Positioning With Unscented Kalman Filter and Low-Cost Sensors,” IEEE Open
Journal of Intelligent Transportation Systems, volume 7, pages 304–312, 2026,
doi:10.1109/OJITS.2026.3652748.
