#!/bin/bash

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00config.sh"

# Input paths
DATA_DIR="$BASE_OUTPUT_DIR/splits"
STAT_BASE_DIR="$BASE_OUTPUT_DIR/glm"

# Output paths
OUT_BASE_DIR="$BASE_OUTPUT_DIR/clumped"
POPS=("hk" "kr" "jp" "sg")

mkdir -p "$OUT_BASE_DIR"
echo "=== Script started at $(date) ==="
echo "Input data: $DATA_DIR"
echo "Input stats: $STAT_BASE_DIR"
echo "Output: $OUT_BASE_DIR"

# --- Thread Detection ---
if [[ -z "${NUM_THREADS:-}" ]]; then
    NUM_THREADS=4
    if command -v nproc >/dev/null 2>&1; then
        NUM_THREADS=$(nproc --all)
    elif command -v sysctl >/dev/null 2>&1 && sysctl -n hw.logicalcpu >/dev/null 2>&1; then
        NUM_THREADS=$(sysctl -n hw.logicalcpu)
    fi
else
    echo "Using NUM_THREADS from environment: $NUM_THREADS"
fi
export NUM_THREADS
echo "Threads: $NUM_THREADS"

find_bfile() {
    local pop="$1"
    local split="$2"
    local rep="$3"
    local data_dir="$4"

    local candidates=(
        "${data_dir}/${pop}_${split}_rep${rep}"
        "${data_dir}/${split}_${pop}_rep${rep}"
    )

    for cand in "${candidates[@]}"; do
        if [[ -f "${cand}.bed" ]]; then
            echo "$cand"
            return 0
        fi
    done
    return 1
}

R2_VALUES=(0.10 0.30 0.60 0.80)
PVAL_THRESHOLDS=(1e-3 1e-5 1e-7 1e-8)

for gwas_cfg in "${GWAS_CONFIGS[@]}"; do
    GWAS_NAME="${gwas_cfg%%:*}"
    STAT_DIR="$STAT_BASE_DIR/${GWAS_NAME}"
    CLUMPED_DIR="$OUT_BASE_DIR/${GWAS_NAME}"
    mkdir -p "$CLUMPED_DIR"
    
    MANIFEST="$CLUMPED_DIR/clump_manifest.tsv"
    echo -e "timestamp\trep\tr2\tpval\tsplit\tpop\tsamples\tsnps\tstatus" > "$MANIFEST"
    
    echo "$(date) === Processing GWAS config: $GWAS_NAME ==="

    for rep in {1..15}; do
        TRAIN_BFILE="$DATA_DIR/stacked_train_rep${rep}"
        GWAS_FILE="$STAT_DIR/train_assoc_rep${rep}.PHENO.glm.logistic.hybrid"

        if [[ ! -f "$TRAIN_BFILE.bed" || ! -f "$GWAS_FILE" ]]; then
            echo "$(date) ERROR: Missing input for rep $rep, config $GWAS_NAME"
            continue
        fi

        for r2 in "${R2_VALUES[@]}"; do
            for pval in "${PVAL_THRESHOLDS[@]}"; do
                r2_suffix="r2${r2//./}"
                pval_suffix="p${pval//[e.-]/}"
                suffix="${r2_suffix}_${pval_suffix}"

                OUT_PREFIX="$CLUMPED_DIR/clumped_rep${rep}_${suffix}"
                plink2 --bfile "$TRAIN_BFILE" \
                       --clump "$GWAS_FILE" \
                       --clump-p1 "$pval" \
                       --clump-p2 5e-2 \
                       --clump-r2 "$r2" \
                       --clump-kb 500 \
                       --threads "$NUM_THREADS" \
                       --out "$OUT_PREFIX"

                CLUMPED="$OUT_PREFIX.clumps"
                if [[ ! -f "$CLUMPED" ]]; then
                    echo "$(date) ERROR: Clumping failed rep=$rep r2=$r2 pval=$pval config=$GWAS_NAME"
                    echo -e "$(date)\t$rep\t$r2\t$pval\t-\t-\t-\t-\tfail" >> "$MANIFEST"
                    continue
                fi

                SNP_IDS="$OUT_PREFIX.snps.txt"
                awk 'NR>1 && $3 != "" {print $3}' "$CLUMPED" > "$SNP_IDS"
                NSNPS=$(wc -l < "$SNP_IDS")

                # Align stacked datasets
                for split in train val test; do
                    INPUT="$DATA_DIR/stacked_${split}_rep${rep}"
                    OUTPUT="$CLUMPED_DIR/stacked_${split}_rep${rep}_${suffix}"

                    if [[ ! -f "$INPUT.bed" ]]; then
                        echo "$(date) ERROR: Missing stacked $split rep=$rep"
                        continue
                    fi

                    if plink2 --bfile "$INPUT" --extract "$SNP_IDS" \
                              --make-bed --freq --threads "$NUM_THREADS" \
                              --out "$OUTPUT"; then
                        NSAMPLES=$(wc -l < "$OUTPUT.fam")
                        echo -e "$(date)\t$rep\t$r2\t$pval\t$split\tall\t$NSAMPLES\t$NSNPS\tsuccess" >> "$MANIFEST"
                    else
                        echo -e "$(date)\t$rep\t$r2\t$pval\t$split\tall\t-\t-\tfail" >> "$MANIFEST"
                    fi
                done

                # Align per-population datasets
                for pop in "${POPS[@]}"; do
                    for split in train val test; do
                        POP_INPUT=$(find_bfile "$pop" "$split" "$rep" "$DATA_DIR")
                        if [[ -z "$POP_INPUT" ]]; then
                            echo "$(date) WARNING: No input for $pop $split rep=$rep"
                            echo -e "$(date)\t$rep\t$r2\t$pval\t$split\t$pop\t-\t-\tmissing" >> "$MANIFEST"
                            continue
                        fi

                        POP_OUTPUT="$CLUMPED_DIR/${pop}_${split}_rep${rep}_${suffix}"
                        if plink2 --bfile "$POP_INPUT" --extract "$SNP_IDS" \
                                  --make-bed --freq --threads "$NUM_THREADS" \
                                  --out "$POP_OUTPUT"; then
                            NSAMPLES=$(wc -l < "$POP_OUTPUT.fam")
                            echo -e "$(date)\t$rep\t$r2\t$pval\t$split\t$pop\t$NSAMPLES\t$NSNPS\tsuccess" >> "$MANIFEST"
                        else
                            echo -e "$(date)\t$rep\t$r2\t$pval\t$split\t$pop\t-\t-\tfail" >> "$MANIFEST"
                        fi
                    done
                done

                rm -f "$SNP_IDS"
            done
        done
    done
done

echo "=== Finished at $(date) ==="