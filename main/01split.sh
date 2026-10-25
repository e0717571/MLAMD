#!/bin/bash

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00config.sh"

# Input paths
SRC_BFILE="$BASE_DATA_DIR/S3_SG_HK_JP_KR_forAnalysis_nodup"

# Output paths
OUT_DIR="$BASE_OUTPUT_DIR/splits"
POPS=("hk" "kr" "jp" "sg")

mkdir -p "$OUT_DIR"

echo "$(date) === Script started ==="
echo "Input: $SRC_BFILE"
echo "Output: $OUT_DIR"

# QC on full dataset
plink --bfile "$SRC_BFILE" \
      --geno 0.05 --mind 0.05 --maf 0.01 \
      --make-bed --freq \
      --out "$OUT_DIR/full_qc"

# Step 0: KEEP + base subsets
FAM_FILE="$OUT_DIR/full_qc.fam"
for pop in "${POPS[@]}"; do
    PHENO_PATH="$BASE_DATA_DIR/${pop}/Phenotype-PCV-tAMD-controls.ped"
    KEEP_FILE="$OUT_DIR/keep_${pop}.txt"
    case "$pop" in
        hk) SUFFIX="HK" ;;
        sg) SUFFIX="SG" ;;
        *)  SUFFIX=""  ;;
    esac
    awk -v sfx="$SUFFIX" 'NR>1 && NF>=2 {
        if (sfx != "") { print $1 sfx, $2 sfx } else { print $1, $2 }
    }' "$PHENO_PATH" > "$KEEP_FILE"

    plink --bfile "$OUT_DIR/full_qc" --keep "$KEEP_FILE" \
          --make-bed --freq \
          --out "$OUT_DIR/${pop}_base"

    echo "$(date) ✓ Base subset created: $OUT_DIR/${pop}_base.* ($(wc -l < $OUT_DIR/${pop}_base.fam) samples)"
done

# --- Step 1a: Stratified random splits per population and replicate ---
TRAIN_FRAC=80   # percent
VAL_FRAC=10     # percent
TEST_FRAC=10    # percent
SEED=42

SPLIT_MANIFEST="$OUT_DIR/split_manifest.tsv"
echo -e "timestamp\trep\tpop\tseed\ttrain_cases\tval_cases\ttest_cases\ttrain_ctrl\tval_ctrl\ttest_ctrl\ttotal\tpheno_path" > "$SPLIT_MANIFEST"

for rep in $(seq 1 15); do
    echo "$(date) → Generating stratified splits for rep $rep"
    REP_SEED=$((SEED + rep))

    for pop in "${POPS[@]}"; do
        PHENO_PATH="$BASE_DATA_DIR/${pop}/Phenotype-PCV-tAMD-controls.ped"
        if [[ ! -s "$PHENO_PATH" ]]; then
            echo "$(date) ⚠ Missing phenotype file for $pop: $PHENO_PATH" | tee -a "$OUT_DIR/missing_files.log"
            continue
        fi
        case "$pop" in
            hk) SUFFIX="HK" ;;
            sg) SUFFIX="SG" ;;
            *)  SUFFIX="" ;;
        esac
        CASES="$OUT_DIR/${pop}_cases_rep${rep}.txt"
        CTRLS="$OUT_DIR/${pop}_controls_rep${rep}.txt"
        awk -v sfx="$SUFFIX" 'NR>1 && $6==2 {
            if (sfx != "") { print $1 sfx, $2 sfx } else { print $1, $2 }
        }' "$PHENO_PATH" > "$CASES"
        awk -v sfx="$SUFFIX" 'NR>1 && $6==1 {
            if (sfx != "") { print $1 sfx, $2 sfx } else { print $1, $2 }
        }' "$PHENO_PATH" > "$CTRLS"

        SHUFFLED_CASES="$OUT_DIR/${pop}_cases_shuffled_rep${rep}.txt"
        SHUFFLED_CTRLS="$OUT_DIR/${pop}_controls_shuffled_rep${rep}.txt"
        awk -v seed="$REP_SEED" 'BEGIN{srand(seed)} {print rand(), $0}' "$CASES" | sort -k1,1n | cut -d' ' -f2- > "$SHUFFLED_CASES"
        awk -v seed="$REP_SEED" 'BEGIN{srand(seed+100)} {print rand(), $0}' "$CTRLS" | sort -k1,1n | cut -d' ' -f2- > "$SHUFFLED_CTRLS"

        NCASE=$(wc -l < "$SHUFFLED_CASES")
        NCTRL=$(wc -l < "$SHUFFLED_CTRLS")
        NCASE_TRAIN=$(( NCASE * TRAIN_FRAC / 100 ))
        NCASE_VAL=$(( NCASE * VAL_FRAC / 100 ))
        NCASE_TEST=$(( NCASE - NCASE_TRAIN - NCASE_VAL ))
        NCTRL_TRAIN=$(( NCTRL * TRAIN_FRAC / 100 ))
        NCTRL_VAL=$(( NCTRL * VAL_FRAC / 100 ))
        NCTRL_TEST=$(( NCTRL - NCTRL_TRAIN - NCTRL_VAL ))

        TRAIN_FILE="$OUT_DIR/${pop}_train_ids_rep${rep}.txt"
        VAL_FILE="$OUT_DIR/${pop}_val_ids_rep${rep}.txt"
        TEST_FILE="$OUT_DIR/${pop}_test_ids_rep${rep}.txt"

        head -n "$NCASE_TRAIN" "$SHUFFLED_CASES" > "$TRAIN_FILE"
        head -n "$NCTRL_TRAIN" "$SHUFFLED_CTRLS" >> "$TRAIN_FILE"

        head -n $((NCASE_TRAIN+NCASE_VAL)) "$SHUFFLED_CASES" | tail -n "$NCASE_VAL" > "$VAL_FILE"
        head -n $((NCTRL_TRAIN+NCTRL_VAL)) "$SHUFFLED_CTRLS" | tail -n "$NCTRL_VAL" >> "$VAL_FILE"

        tail -n "$NCASE_TEST" "$SHUFFLED_CASES" > "$TEST_FILE"
        tail -n "$NCTRL_TEST" "$SHUFFLED_CTRLS" >> "$TEST_FILE"

        TOTAL=$((NCASE+NCTRL))
        echo "$(date) ✓ $pop rep=$rep seed=$REP_SEED → train_cases=$NCASE_TRAIN val_cases=$NCASE_VAL test_cases=$NCASE_TEST train_ctrl=$NCTRL_TRAIN val_ctrl=$NCTRL_VAL test_ctrl=$NCTRL_TEST total=$TOTAL"
        echo -e "$(date)\t$rep\t$pop\t$REP_SEED\t$NCASE_TRAIN\t$NCASE_VAL\t$NCASE_TEST\t$NCTRL_TRAIN\t$NCTRL_VAL\t$NCTRL_TEST\t$TOTAL\t$PHENO_PATH" >> "$SPLIT_MANIFEST"
    done
done

echo "$(date) === Stratified random splitting completed; manifest at $SPLIT_MANIFEST ==="

# Step 1b: combine populations and call PLINK once per split
for rep in $(seq 1 15); do
    echo "$(date) → Creating stacked datasets for rep $rep"
    for split in train val test; do
        STACK="$OUT_DIR/stacked_${split}_rep${rep}.txt"
        : > "$STACK"
        for pop in "${POPS[@]}"; do
            cat "$OUT_DIR/${pop}_${split}_ids_rep${rep}.txt" >> "$STACK"
        done
        echo "$(date) → $split combined: $(wc -l < $STACK) samples"

        plink --bfile "$OUT_DIR/full_qc" --keep "$STACK" \
              --make-bed --freq \
              --out "$OUT_DIR/stacked_${split}_rep${rep}"

        echo "$(date) ✓ PLINK extraction done: $split"
    done
done

# Step 1c: per-population train/val/test datasets
for rep in $(seq 1 15); do
    echo "$(date) → Creating per-population datasets for rep $rep"
    for pop in "${POPS[@]}"; do
        for split in train val test; do
            SPLIT_FILE="$OUT_DIR/${pop}_${split}_ids_rep${rep}.txt"
            if [[ ! -s "$SPLIT_FILE" ]]; then
                echo "$(date) ⚠ Missing split file for $pop $split rep=$rep"
                continue
            fi

            OUT_PREFIX="$OUT_DIR/${pop}_${split}_rep${rep}"
            plink --bfile "$OUT_DIR/full_qc" --keep "$SPLIT_FILE" \
                  --make-bed --freq \
                  --out "$OUT_PREFIX"

            echo "$(date) ✓ PLINK extraction done: $pop $split rep=$rep ($(wc -l < ${OUT_PREFIX}.fam) samples)"
        done
    done
done

echo "$(date) === syncing NFS ==="
sync
sleep 5
echo "$(date) === Script finished ==="