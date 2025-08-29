#!/bin/bash
set -euo pipefail

BASE="/scratch/na00078/projects/IPTA_MDC2/mdc2/group2/dataset_2"
PARDIR="$BASE/par"
TIMDIR="$BASE/tim"
CLEAN="$BASE/clean"

# Make clean output directory
mkdir -p "$CLEAN"

# Loop through all .par files
for par in "$PARDIR"/*.par; do
    psrname=$(basename "$par" .par)
    tim="$TIMDIR/$psrname.tim"

    if [[ ! -f "$tim" ]]; then
        echo "⚠️ No tim file for $psrname, skipping"
        continue
    fi

    cleanpar="$CLEAN/${psrname}_clean.par"
    cleantim="$CLEAN/${psrname}_clean.tim"

    echo "Cleaning $psrname ..."
    tempo2 -f "$par" "$tim" -newpar "$cleanpar" -out "$cleantim" > "$CLEAN/${psrname}.log" 2>&1
done

echo "✅ Cleaning complete. Cleaned files are in: $CLEAN/"

