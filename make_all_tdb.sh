#!/usr/bin/env bash
set -euo pipefail

BASE="/scratch/na00078/projects/IPTA_MDC2/mdc2/group2/dataset_2"
PAR_DIR="$BASE/par"
TIM_DIR="$BASE/tim"
CLEAN_DIR="$BASE/par_clean"
FIXED_DIR="$BASE/par_fixed"     # pre-cleaned before tcb2tdb
TDB_DIR="$BASE/par_tdb"
LOG_DIR="$BASE/tdb_logs"

mkdir -p "$CLEAN_DIR" "$FIXED_DIR" "$TDB_DIR" "$LOG_DIR"

if ! command -v tempo2 >/dev/null; then echo "tempo2 not in PATH"; exit 1; fi
if ! command -v tcb2tdb >/dev/null; then echo "tcb2tdb not in PATH"; exit 1; fi

echo "Found $(ls -1 "$PAR_DIR"/*.par | wc -l) .par files in: $PAR_DIR"
echo "TIM files expected in: $TIM_DIR"
echo

# preclean() takes a tempo2-cleaned par and writes a fixed version:
#  - keep only first NE_SW line
#  - drop DILATEFREQ lines
#  - force TIMEEPH FB90 (PINT only supports FB90)
#  - ensure UNITS TCB present (tcb2tdb expects TCB input)
#  - if POSEPOCH/DMEPOCH missing, set them to PEPOCH if present
preclean() {
  inpar="$1"
  outpar="$2"
  awk '
    BEGIN{ nesw_seen=0; have_posepoch=0; have_dmepoch=0; have_pepoch=0; have_unitts=0; }
    # track PEPOCH/POSEPOCH/DMEPOCH
    /^PEPOCH[[:space:]]/ {have_pepoch=1}
    /^POSEPOCH[[:space:]]/ {have_posepoch=1}
    /^DMEPOCH[[:space:]]/  {have_dmepoch=1}
    /^UNITS[[:space:]]/    {have_unitts=1}

    # drop DILATEFREQ entirely
    /^DILATEFREQ[[:space:]]/ {next}

    # keep only first NE_SW
    /^NE_SW[[:space:]]/ { if(nesw_seen) next; nesw_seen=1 }

    # normalize TIMEEPH to FB90
    /^TIMEEPH[[:space:]]/ { print "TIMEEPH FB90"; next }

    { print }
    END {
      if (!have_unitts) print "UNITS TCB"
    }
  ' "$inpar" > "$outpar"

  # If posepoch/dmepoch missing but pepoch exists: add them
  if grep -qE '^PEPOCH[[:space:]]' "$outpar"; then
    pepoch=$(awk '/^PEPOCH[[:space:]]/ {print $2; exit}' "$outpar")
    grep -qE '^POSEPOCH[[:space:]]' "$outpar" || echo "POSEPOCH $pepoch" >> "$outpar"
    grep -qE '^DMEPOCH[[:space:]]'  "$outpar" || echo "DMEPOCH $pepoch"  >> "$outpar"
  fi
}

for par in "$PAR_DIR"/*.par; do
  psr=$(basename "$par" .par)
  tim="$TIM_DIR/$psr.tim"
  clean="$CLEAN_DIR/${psr}_clean.par"
  fixed="$FIXED_DIR/${psr}_fixed.par"
  tdb="$TDB_DIR/${psr}_tdb.par"
  log_clean="$LOG_DIR/${psr}_clean.log"
  log_tcb="$LOG_DIR/${psr}_tcb2tdb.log"

  echo "=== Processing $psr ==="

  if [[ ! -f "$tim" ]]; then
    echo "⚠️  Missing TIM: $tim"
    continue
  fi

  if [[ -f "$tdb" ]]; then
    echo "✅ Already converted: $(basename "$tdb") (skipping)"
    continue
  fi

  echo "🛠️  tempo2 clean -> $(basename "$clean")"
  if ! tempo2 -f "$par" "$tim" -nonobs -newpar "$clean" &> "$log_clean"; then
    echo "❌ tempo2 clean FAILED (see $log_clean)"
    continue
  fi

  echo "🧹 pre-clean for PINT compatibility -> $(basename "$fixed")"
  preclean "$clean" "$fixed"

  echo "🧭 tcb2tdb -> $(basename "$tdb")"
  if ! tcb2tdb "$fixed" "$tdb" &> "$log_tcb"; then
    echo "❌ tcb2tdb FAILED (see $log_tcb)"; echo
    continue
  fi

  if [[ -f "$tdb" ]]; then
    echo "✅ Wrote $tdb"
  else
    echo "⛔ Still missing $tdb (check logs)"
  fi
  echo
done

echo "Done. TDB parfiles in: $TDB_DIR"
echo "Logs in: $LOG_DIR"
echo "Counts: $(ls -1 "$TDB_DIR"/*.par 2>/dev/null | wc -l) converted."
