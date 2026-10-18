#!/usr/bin/env bash
# Download the PhysioNet/Computing in Cardiology Challenge 2019 training data (v1.0.0, open access, CC BY 4.0)
# from the OFFICIAL sources and verify it against the manifest used for the reported runs.
# Usage: scripts/download_physionet2019.sh <target_dir>      (needs ~260 MB of disk)
set -euo pipefail
DEST=${1:-data/physionet2019}
mkdir -p "$DEST"
if command -v aws >/dev/null 2>&1; then
  # Official AWS Open Data mirror operated by PhysioNet (no credentials needed)
  aws s3 sync --no-sign-request s3://physionet-open/challenge-2019/1.0.0/training/ "$DEST/"
else
  # Official PhysioNet web server (recursive download of the two training folders)
  wget -r -N -c -np -nH --cut-dirs=4 -R "index.html*" -P "$DEST" https://physionet.org/files/challenge-2019/1.0.0/training/
fi
# expected layout: $DEST/training_setA/p000001.psv ... (20,336 files) and $DEST/training_setB/p100001.psv ... (20,000 files)
nA=$(ls "$DEST"/training_setA/*.psv | wc -l); nB=$(ls "$DEST"/training_setB/*.psv | wc -l)
echo "training_setA: $nA files (expected 20336); training_setB: $nB files (expected 20000)"
# verify every file against the SHA-256 manifest of the copy used for the reported results
( cd "$DEST" && sha256sum -c --quiet "$(dirname "$0")/../docs/physionet2019_sha256_manifest.txt" ) && echo "SHA-256 manifest: all 40,336 files identical to the copy used for the reported runs"
