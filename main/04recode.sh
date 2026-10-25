#!/bin/bash

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00config.sh"

# Input paths
CLUMPED_BASE_DIR="$BASE_OUTPUT_DIR/clumped"
PHENO_FILE="$BASE_OUTPUT_DIR/glm/pheno_all.txt"
COVAR_FILE="$BASE_OUTPUT_DIR/glm/covar_all.txt"

# Output paths
ML_OUT_BASE_DIR="$BASE_OUTPUT_DIR/ml_input"
POPS=("hk" "kr" "jp" "sg")

R2_VALUES=(0.10 0.30 0.60 0.80)
PVAL_THRESHOLDS=(1e-3 1e-5 1e-7 1e-8)

mkdir -p "$ML_OUT_BASE_DIR"

# Verify directory is writable
if [ ! -w "$ML_OUT_BASE_DIR" ]; then
    echo "$(date) ERROR: Output directory not writable: $ML_OUT_BASE_DIR"
    exit 1
fi
echo "=== Script started at $(date) ==="
echo "Input clumped: $CLUMPED_BASE_DIR"
echo "Input phenotypes: $PHENO_FILE"
echo "Input covariates: $COVAR_FILE"
echo "Output: $ML_OUT_BASE_DIR"
echo ""
echo "Note: One-hot encoding (X_geno_pc_onehot) is only used in ML, not GWAS"
echo "      ML algorithms handle collinearity via regularization or tree splits"

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

if [ ! -f "$PHENO_FILE" ]; then
    echo "$(date) ERROR: Phenotype file missing: $PHENO_FILE"
    exit 1
fi

if [ ! -f "$COVAR_FILE" ]; then
    echo "$(date) ERROR: Covariate file missing: $COVAR_FILE"
    exit 1
fi

for gwas_cfg in "${GWAS_CONFIGS[@]}"; do
    GWAS_NAME="${gwas_cfg%%:*}"
    CLUMPED_DIR="$CLUMPED_BASE_DIR/${GWAS_NAME}"
    
    echo "$(date) === Processing GWAS config: $GWAS_NAME ==="

    for ml_cfg in "${ML_CONFIGS[@]}"; do
        ML_OUT_DIR="$ML_OUT_BASE_DIR/${GWAS_NAME}/${ml_cfg}"
        mkdir -p "$ML_OUT_DIR"
        
        echo "$(date) === ML config: $ml_cfg ==="

        for rep in {1..15}; do
            echo "$(date) === Starting repetition $rep ==="
            for r2 in "${R2_VALUES[@]}"; do
                for pval in "${PVAL_THRESHOLDS[@]}"; do
                    r2_suffix="r2${r2//./}"
                    pval_suffix="p${pval//[e.-]/}"
                    suffix="${r2_suffix}_${pval_suffix}"
                    
                    # --- Stacked datasets ---
                    for split in train val test; do
                        INPUT_BFILE="$CLUMPED_DIR/stacked_${split}_rep${rep}_${suffix}"
                        
                        if [ ! -f "$INPUT_BFILE.bed" ]; then
                            echo "$(date) ERROR: Input bfile missing for $split, rep $rep, suffix $suffix"
                            continue
                        fi
                        
                        RAW_OUT="$ML_OUT_DIR/tmp_stacked_${split}_rep${rep}_${suffix}"
                        X_GENOTYPES="$ML_OUT_DIR/tmp_X_genotypes_stacked_${split}_rep${rep}_${suffix}.txt"
                        X_FILE="$ML_OUT_DIR/X_stacked_${split}_rep${rep}_${suffix}.txt"
                        Y_FILE="$ML_OUT_DIR/y_stacked_${split}_rep${rep}_${suffix}.txt"
                        HEADER_FILE="$ML_OUT_DIR/header_stacked_${split}_rep${rep}_${suffix}.txt"

                        plink2 --bfile "$INPUT_BFILE" --recode A --threads "$NUM_THREADS" --out "$RAW_OUT"
                        
                        if [ ! -f "$RAW_OUT.raw" ]; then
                            echo "$(date) ERROR: Failed to export .raw for $split, rep $rep, suffix $suffix"
                            continue
                        fi

                        tail -n +2 "$RAW_OUT.raw" | awk '{for (i=7;i<=NF;i++) printf (i==7?"":" ") $i; print ""}' > "$X_GENOTYPES"
                        awk 'NR==FNR {pheno[$1" "$2]=$3; next} {print pheno[$1" "$2]}' "$PHENO_FILE" "$INPUT_BFILE.fam" > "$Y_FILE"

                        case "$ml_cfg" in
                            X_geno)
                                head -n 1 "$RAW_OUT.raw" | awk '{for (i=7;i<=NF;i++) printf (i==7?"":" ") $i; print ""}' > "$HEADER_FILE"
                                mv "$X_GENOTYPES" "$X_FILE"
                                ;;
                            X_geno_pc)
                                head -n 1 "$RAW_OUT.raw" | awk '{for (i=7;i<=NF;i++) printf (i==7?"":" ") $i; print ""}' > "$HEADER_FILE"
                                printf " P1 P2 P3 P4 P5 P6 P7 P8 P9 P10\n" >> "$HEADER_FILE"
                                
                                PC_FILE="$ML_OUT_DIR/tmp_pc_${split}_rep${rep}_${suffix}.txt"
                                awk 'NR==FNR {pc[$1" "$2]=$0; next} 
                                     {key=$1" "$2; if (key in pc) print pc[key]}' \
                                    "$COVAR_FILE" "$INPUT_BFILE.fam" | \
                                    cut -d' ' -f3-12 > "$PC_FILE"
                                paste -d' ' "$X_GENOTYPES" "$PC_FILE" > "$X_FILE"
                                rm -f "$PC_FILE"
                                ;;
                            X_geno_dummy)
                                head -n 1 "$RAW_OUT.raw" | awk '{for (i=7;i<=NF;i++) printf (i==7?"":" ") $i; print ""}' > "$HEADER_FILE"
                                printf " hk kr jp\n" >> "$HEADER_FILE"
                                
                                DUMMY_FILE="$ML_OUT_DIR/tmp_dummy_${split}_rep${rep}_${suffix}.txt"
                                awk 'NR==FNR {info[$1" "$2]=$0; next} 
                                     {key=$1" "$2; if (key in info) print info[key]}' \
                                    "$COVAR_FILE" "$INPUT_BFILE.fam" | \
                                    cut -d' ' -f13-15 > "$DUMMY_FILE"
                                paste -d' ' "$X_GENOTYPES" "$DUMMY_FILE" > "$X_FILE"
                                rm -f "$DUMMY_FILE"
                                ;;
                            X_geno_onehot)
                                head -n 1 "$RAW_OUT.raw" | awk '{for (i=7;i<=NF;i++) printf (i==7?"":" ") $i; print ""}' > "$HEADER_FILE"
                                printf " hk kr jp sg\n" >> "$HEADER_FILE"
                                
                                ONEHOT_FILE="$ML_OUT_DIR/tmp_onehot_${split}_rep${rep}_${suffix}.txt"
                                awk 'NR==FNR {info[$1" "$2]=$0; next} 
                                     {key=$1" "$2; if (key in info) print info[key]}' \
                                    "$COVAR_FILE" "$INPUT_BFILE.fam" | \
                                    awk '{print $13, $14, $15, $16}' > "$ONEHOT_FILE"
                                paste -d' ' "$X_GENOTYPES" "$ONEHOT_FILE" > "$X_FILE"
                                rm -f "$ONEHOT_FILE"
                                ;;
                            X_geno_pc_dummy)
                                head -n 1 "$RAW_OUT.raw" | awk '{for (i=7;i<=NF;i++) printf (i==7?"":" ") $i; print ""}' > "$HEADER_FILE"
                                printf " P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 hk kr jp\n" >> "$HEADER_FILE"
                                
                                PC_DUMMY_FILE="$ML_OUT_DIR/tmp_pc_dummy_${split}_rep${rep}_${suffix}.txt"
                                awk 'NR==FNR {info[$1" "$2]=$0; next} 
                                     {key=$1" "$2; if (key in info) print info[key]}' \
                                    "$COVAR_FILE" "$INPUT_BFILE.fam" | \
                                    cut -d' ' -f3-15 > "$PC_DUMMY_FILE"
                                paste -d' ' "$X_GENOTYPES" "$PC_DUMMY_FILE" > "$X_FILE"
                                rm -f "$PC_DUMMY_FILE"
                                ;;
                            X_geno_pc_onehot)
                                head -n 1 "$RAW_OUT.raw" | awk '{for (i=7;i<=NF;i++) printf (i==7?"":" ") $i; print ""}' > "$HEADER_FILE"
                                printf " P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 hk kr jp sg\n" >> "$HEADER_FILE"
                                
                                PC_ONEHOT_FILE="$ML_OUT_DIR/tmp_pc_onehot_${split}_rep${rep}_${suffix}.txt"
                                awk 'NR==FNR {info[$1" "$2]=$0; next} 
                                     {key=$1" "$2; if (key in info) print info[key]}' \
                                    "$COVAR_FILE" "$INPUT_BFILE.fam" | \
                                    awk '{print $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16}' > "$PC_ONEHOT_FILE"
                                paste -d' ' "$X_GENOTYPES" "$PC_ONEHOT_FILE" > "$X_FILE"
                                rm -f "$PC_ONEHOT_FILE"
                                ;;
                        esac

                        if [ -s "$X_FILE" ] && [ -s "$Y_FILE" ]; then
                            num_samples=$(wc -l < "$Y_FILE")
                            num_features=$(wc -w < "$HEADER_FILE")
                            echo "$(date) ✓ ML inputs created for stacked $split: X ($num_samples samples, $num_features features)"
                        else
                            echo "$(date) ERROR: Failed to create X or y for $split, rep $rep, suffix $suffix"
                        fi

                        rm -f "$RAW_OUT.raw" "$RAW_OUT.log" "$X_GENOTYPES"
                    done

                    # --- Per-population datasets (similar pattern) ---
                    for pop in "${POPS[@]}"; do
                        for split in train val test; do
                            INPUT_BFILE_P="$CLUMPED_DIR/${pop}_${split}_rep${rep}_${suffix}"
                            
                            [ ! -f "${INPUT_BFILE_P}.bed" ] && { 
                                echo "$(date) WARNING: No bfile for $pop $split rep $rep"; 
                                continue; 
                            }

                            RAW_OUT_P="$ML_OUT_DIR/tmp_${pop}_${split}_rep${rep}_${suffix}"
                            X_GENOTYPES_P="$ML_OUT_DIR/tmp_X_genotypes_${pop}_${split}_rep${rep}_${suffix}.txt"
                            X_FILE_P="$ML_OUT_DIR/X_${pop}_${split}_rep${rep}_${suffix}.txt"
                            Y_FILE_P="$ML_OUT_DIR/y_${pop}_${split}_rep${rep}_${suffix}.txt"
                            HEADER_FILE_P="$ML_OUT_DIR/header_${pop}_${split}_rep${rep}_${suffix}.txt"

                            plink2 --bfile "$INPUT_BFILE_P" --recode A --threads "$NUM_THREADS" --out "$RAW_OUT_P"
                            [ ! -f "${RAW_OUT_P}.raw" ] && continue

                            tail -n +2 "${RAW_OUT_P}.raw" | awk '{for (i=7;i<=NF;i++) printf (i==7?"":" ") $i; print ""}' > "$X_GENOTYPES_P"
                            awk 'NR==FNR {pheno[$1" "$2]=$3; next} {print pheno[$1" "$2]}' "$PHENO_FILE" "${INPUT_BFILE_P}.fam" > "$Y_FILE_P"

                            # Same case statement as stacked
                            case "$ml_cfg" in
                                X_geno)
                                    head -n 1 "${RAW_OUT_P}.raw" | awk '{for (i=7;i<=NF;i++) printf (i==7?"":" ") $i; print ""}' > "$HEADER_FILE_P"
                                    mv "$X_GENOTYPES_P" "$X_FILE_P"
                                    ;;
                                X_geno_pc)
                                    head -n 1 "${RAW_OUT_P}.raw" | awk '{for (i=7;i<=NF;i++) printf (i==7?"":" ") $i; print ""}' > "$HEADER_FILE_P"
                                    printf " P1 P2 P3 P4 P5 P6 P7 P8 P9 P10\n" >> "$HEADER_FILE_P"
                                    
                                    PC_FILE_P="$ML_OUT_DIR/tmp_pc_${pop}_${split}_rep${rep}_${suffix}.txt"
                                    awk 'NR==FNR {pc[$1" "$2]=$0; next} 
                                         {key=$1" "$2; if (key in pc) print pc[key]}' \
                                        "$COVAR_FILE" "${INPUT_BFILE_P}.fam" | \
                                        cut -d' ' -f3-12 > "$PC_FILE_P"
                                    paste -d' ' "$X_GENOTYPES_P" "$PC_FILE_P" > "$X_FILE_P"
                                    rm -f "$PC_FILE_P"
                                    ;;
                                X_geno_dummy)
                                    head -n 1 "${RAW_OUT_P}.raw" | awk '{for (i=7;i<=NF;i++) printf (i==7?"":" ") $i; print ""}' > "$HEADER_FILE_P"
                                    printf " hk kr jp\n" >> "$HEADER_FILE_P"
                                    
                                    DUMMY_FILE_P="$ML_OUT_DIR/tmp_dummy_${pop}_${split}_rep${rep}_${suffix}.txt"
                                    awk 'NR==FNR {info[$1" "$2]=$0; next} 
                                         {key=$1" "$2; if (key in info) print info[key]}' \
                                        "$COVAR_FILE" "${INPUT_BFILE_P}.fam" | \
                                        cut -d' ' -f13-15 > "$DUMMY_FILE_P"
                                    paste -d' ' "$X_GENOTYPES_P" "$DUMMY_FILE_P" > "$X_FILE_P"
                                    rm -f "$DUMMY_FILE_P"
                                    ;;
                                X_geno_onehot)
                                    head -n 1 "${RAW_OUT_P}.raw" | awk '{for (i=7;i<=NF;i++) printf (i==7?"":" ") $i; print ""}' > "$HEADER_FILE_P"
                                    printf " hk kr jp sg\n" >> "$HEADER_FILE_P"
                                    
                                    ONEHOT_FILE_P="$ML_OUT_DIR/tmp_onehot_${pop}_${split}_rep${rep}_${suffix}.txt"
                                    awk 'NR==FNR {info[$1" "$2]=$0; next} 
                                         {key=$1" "$2; if (key in info) print info[key]}' \
                                        "$COVAR_FILE" "${INPUT_BFILE_P}.fam" | \
                                        awk '{print $13, $14, $15, $16}' > "$ONEHOT_FILE_P"
                                    paste -d' ' "$X_GENOTYPES_P" "$ONEHOT_FILE_P" > "$X_FILE_P"
                                    rm -f "$ONEHOT_FILE_P"
                                    ;;
                                X_geno_pc_dummy)
                                    head -n 1 "${RAW_OUT_P}.raw" | awk '{for (i=7;i<=NF;i++) printf (i==7?"":" ") $i; print ""}' > "$HEADER_FILE_P"
                                    printf " P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 hk kr jp\n" >> "$HEADER_FILE_P"
                                    
                                    PC_DUMMY_FILE_P="$ML_OUT_DIR/tmp_pc_dummy_${pop}_${split}_rep${rep}_${suffix}.txt"
                                    awk 'NR==FNR {info[$1" "$2]=$0; next} 
                                         {key=$1" "$2; if (key in info) print info[key]}' \
                                        "$COVAR_FILE" "${INPUT_BFILE_P}.fam" | \
                                        cut -d' ' -f3-15 > "$PC_DUMMY_FILE_P"
                                    paste -d' ' "$X_GENOTYPES_P" "$PC_DUMMY_FILE_P" > "$X_FILE_P"
                                    rm -f "$PC_DUMMY_FILE_P"
                                    ;;
                                X_geno_pc_onehot)
                                    head -n 1 "${RAW_OUT_P}.raw" | awk '{for (i=7;i<=NF;i++) printf (i==7?"":" ") $i; print ""}' > "$HEADER_FILE_P"
                                    printf " P1 P2 P3 P4 P5 P6 P7 P8 P9 P10 hk kr jp sg\n" >> "$HEADER_FILE_P"
                                    
                                    PC_ONEHOT_FILE_P="$ML_OUT_DIR/tmp_pc_onehot_${pop}_${split}_rep${rep}_${suffix}.txt"
                                    awk 'NR==FNR {info[$1" "$2]=$0; next} 
                                         {key=$1" "$2; if (key in info) print info[key]}' \
                                        "$COVAR_FILE" "${INPUT_BFILE_P}.fam" | \
                                        awk '{print $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16}' > "$PC_ONEHOT_FILE_P"
                                    paste -d' ' "$X_GENOTYPES_P" "$PC_ONEHOT_FILE_P" > "$X_FILE_P"
                                    rm -f "$PC_ONEHOT_FILE_P"
                                    ;;
                            esac

                            rm -f "${RAW_OUT_P}.raw" "${RAW_OUT_P}.log" "$X_GENOTYPES_P"
                        done
                    done
                done
            done
        done
    done
done

echo "=== Script finished at $(date) ==="