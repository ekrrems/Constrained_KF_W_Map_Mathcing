# Input data

The code reads all inputs from this folder:

| File | Content |
|---|---|
| `imu_300_30012025_13.mat` | IMU of the recording (30 January 2025, set 13), 300 Hz |
| `gnss_300_30012025_13.mat` | GNSS of the same recording, 1 Hz |
| `track.db` | OSM track map: nodes with UTM 32N coordinates and ordered ways (SQLite) |
| `Ref_Nodes.txt` | Track nodes digitized from orthophotos (orthophoto map) |
| `Ref_Intermetric_UTM.txt` | Evaluation reference polyline (track centre line) |
| `dop20_thesis_panels/` | Three DOP20 orthophoto excerpts for figure 18 (`tools/download_dop20_panels.py`) |

These files are not part of the Git repository.
