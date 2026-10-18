# Dataset: PhysioNet/Computing in Cardiology Challenge 2019

* **Name / version:** Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019, version 1.0.0 (Reyna et al., 2019/2020). DOI: 10.13026/v64v-d857
* **Official page:** https://physionet.org/content/challenge-2019/1.0.0/
* **Access:** Open Access, Creative Commons Attribution 4.0 International. No registration, credentialing or data-use agreement is required (only MIMIC-IV, the planned external-validation set, is credentialed).
* **Files needed:** `training/training_setA/*.psv` (20,336 files) and `training/training_setB/*.psv` (20,000 files); about 255 MB uncompressed. Each file is one ICU stay: 40 hourly variables + `SepsisLabel`, pipe-separated.
* **Download:** `scripts/download_physionet2019.sh data/physionet2019` (official AWS Open Data mirror `s3://physionet-open/challenge-2019/1.0.0/` if the AWS CLI is installed, otherwise `wget -r` from physionet.org). Then: `python -c "from acmta_sepsis.data import build_cache; build_cache('data/physionet2019', 'data/physionet2019.npz')"`.
* **Expected structure:** `data/physionet2019/training_setA/p000001.psv ...`, `data/physionet2019/training_setB/p100001.psv ...`
* **Checksums:** `docs/physionet2019_sha256_manifest.txt` lists the SHA-256 of all 40,336 files used for the reported runs; the download script verifies a fresh download against it.

## Provenance of the copy used for the reported runs (read this)

The execution sandbox could not reach physionet.org, AWS S3 or Google Cloud Storage. The files were therefore obtained
from two independent public GitHub repositories that redistribute the CC BY 4.0 training set
(`shayaan-ahm3d/sepsis` and `ANSR1226/EarlySepsis`) and verified as follows:

1. The two copies are byte-identical for all 40,336 files (identical git blob hashes).
2. Two files (p000001.psv from set A; p100047.psv, a septic stay, from set B) were fetched from the official
   physionet.org server and matched row by row.
3. The data reproduce the published characteristics: 40,336 stays, 1,552,210 hourly rows, 2,932 septic stays
   (set A 1,790; set B 1,142), prevalence 7.27 %.

PhysioNet does not publish per-file checksums for this project, so byte-level identity with the official server is
established only for the spot-checked files. Re-running `scripts/download_physionet2019.sh` against the official
source and checking the manifest removes this residual uncertainty.
