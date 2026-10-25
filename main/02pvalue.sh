#!/bin/bash
set -euo pipefail

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00config.sh"

# Input paths
DATA_DIR="$BASE_OUTPUT_DIR/splits"

# Output paths
OUT_DIR="$BASE_OUTPUT_DIR/glm"
POPS=("hk" "kr" "jp" "sg")

mkdir -p "$OUT_DIR"

echo "$(date) === Script started ==="
echo "Input: $DATA_DIR"
echo "Output: $OUT_DIR"

# --- Thread detection ---
if [[ -z "${NUM_THREADS:-}" ]]; then
    NUM_THREADS=4
    if command -v nproc >/dev/null 2>&1; then
        NUM_THREADS=$(nproc --all)
        echo "Detected logical cores (nproc): $NUM_THREADS"
    elif command -v sysctl >/dev/null 2>&1; then
        if threads=$(sysctl -n hw.logicalcpu 2>/dev/null); then
            NUM_THREADS="$threads"
            echo "Detected logical cores (sysctl): $NUM_THREADS"
        fi
    else
        echo "Warning: Could not detect number of logical cores. Defaulting to $NUM_THREADS threads."
    fi
else
    echo "Using NUM_THREADS from environment: $NUM_THREADS"
fi
export NUM_THREADS

# Unified phenotype + covariate files
# Note: We include sg in COVAR_ALL for ML use, but GWAS won't use it
PHENO_ALL="$OUT_DIR/pheno_all.txt"
COVAR_ALL="$OUT_DIR/covar_all.txt"

printf "FID IID PHENO\n" > "$PHENO_ALL"
printf "FID IID P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 hk kr jp sg\n" > "$COVAR_ALL"

for pop in "${POPS[@]}"; do
    PHENO_PATH="$BASE_DATA_DIR/${pop}/Phenotype-PCV-tAMD-controls.ped"

    case "$pop" in
        hk) SUFFIX="HK"; dummies="1 0 0 0" ;;
        kr) SUFFIX="";    dummies="0 1 0 0" ;;
        jp) SUFFIX="";    dummies="0 0 1 0" ;;
        sg) SUFFIX="SG"; dummies="0 0 0 1" ;;
    esac

    if [[ -s "$PHENO_PATH" ]]; then
        awk -v OFS=" " -v sfx="$SUFFIX" 'NR>1 {
            fid = $1 sfx
            iid = $2 sfx
            print fid, iid, $6
        }' "$PHENO_PATH" >> "$PHENO_ALL"

        awk -v OFS=" " -v sfx="$SUFFIX" -v d="$dummies" 'NR>1 {
            fid = $1 sfx
            iid = $2 sfx
            line = fid " " iid
            for (i=7; i<=16; i++) line = line " " $i
            n = split(d, arr, " ")
            for (j=1; j<=n; j++) line = line " " arr[j]
            print line
        }' "$PHENO_PATH" >> "$COVAR_ALL"
    else
        echo "$(date) ⚠ Missing phenotype file for $pop: $PHENO_PATH" | tee -a "$OUT_DIR/missing_files.log"
    fi
done

echo "$(date) ✓ Unified phenotype file built: $PHENO_ALL ($(wc -l < "$PHENO_ALL") lines)"
echo "$(date) ✓ Unified covariate file built: $COVAR_ALL ($(wc -l < "$COVAR_ALL") lines)"
echo "$(date) Note: COVAR_ALL contains sg column for ML use, but GWAS uses dummy coding only"

# Run GWAS for each configuration and repetition
for gwas_cfg in "${GWAS_CONFIGS[@]}"; do
    CFG_NAME="${gwas_cfg%%:*}"
    COVAR_SPEC="${gwas_cfg#*:}"
    COVAR_SPEC="${COVAR_SPEC%:*}"
    
    MODEL_OUTDIR="$OUT_DIR/${CFG_NAME}"
    mkdir -p "$MODEL_OUTDIR"
    
    echo "$(date) === GWAS Configuration: $CFG_NAME ==="
    
    # Validate no one-hot encoding in GWAS
    if [[ "$COVAR_SPEC" == *"sg"* ]]; then
        echo "$(date) ERROR: One-hot encoding detected in GWAS config! This causes collinearity."
        echo "$(date) Config: $CFG_NAME, Spec: $COVAR_SPEC"
        exit 1
    fi

    for rep in {1..15}; do
        echo "$(date) --- Repetition $rep, Config $CFG_NAME ---"
        TRAIN_BFILE="$DATA_DIR/stacked_train_rep${rep}"
        OUT_PREFIX="$MODEL_OUTDIR/train_assoc_rep${rep}"

        if [[ ! -f "${TRAIN_BFILE}.bed" ]]; then
            echo "$(date) ⚠ Missing PLINK bfile for rep $rep: ${TRAIN_BFILE}.*" | tee -a "$OUT_DIR/missing_files.log"
            continue
        fi

        echo "$(date) → Running GWAS for rep $rep, config $CFG_NAME"
        
        if [[ -z "$COVAR_SPEC" ]]; then
            # No covariates
            plink2 \
              --bfile "$TRAIN_BFILE" \
              --pheno "$PHENO_ALL" \
              --pheno-name PHENO \
              --glm hide-covar allow-no-covars \
              --freq \
              --threads "$NUM_THREADS" \
              --out "$OUT_PREFIX"
        else
            # With covariates (dummy coding only, no one-hot)
            plink2 \
              --bfile "$TRAIN_BFILE" \
              --pheno "$PHENO_ALL" \
              --pheno-name PHENO \
              --covar "$COVAR_ALL" \
              --covar-name "$COVAR_SPEC" \
              --covar-variance-standardize \
              --glm hide-covar \
              --freq \
              --threads "$NUM_THREADS" \
              --out "$OUT_PREFIX"
        fi

        echo "$(date) ✓ GWAS completed for rep $rep, config $CFG_NAME → ${OUT_PREFIX}"
    done
done

echo "$(date) === All GWAS configurations and repetitions finished ==="
echo "$(date) Configurations run: ${GWAS_CONFIGS[*]}"