#!/bin/bash

# === DIRECTORY CONFIGURATION ===
export BASE_DATA_DIR="${BASE_DATA_DIR:-$HOME/scratch/gwas-data/plink}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$HOME/nas/magnus/ml_res}"

# Create output directory structure
mkdir -p "$BASE_OUTPUT_DIR"/{splits,glm,clumped,ml_input,models/{lasso,mlp,rf,xgb},logs}

# === COVARIATE CONFIGURATIONS ===

# GWAS configurations (Stage 2) - NO one-hot encoding due to collinearity in GLM
declare -a GWAS_CONFIGS=(
    "geno_only::"
    "geno_pc:P1,P2,P3,P4,P5,P6,P7,P8,P9,P10:"
    "geno_dummy:hk,kr,jp:"
    "geno_pc_dummy:P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,hk,kr,jp:"
)

# ML predictor configurations (Stage 4+) - one-hot is fine for ML algorithms
declare -a ML_CONFIGS=(
    "X_geno"
    "X_geno_pc"
    "X_geno_dummy"
    "X_geno_onehot"
    "X_geno_pc_dummy"
    "X_geno_pc_onehot"
)

export GWAS_CONFIGS
export ML_CONFIGS

echo "=== Configuration Matrix ==="
echo "GWAS configs: ${#GWAS_CONFIGS[@]} (geno_only, geno_dummy, geno_pc, geno_pc_dummy)"
echo "ML configs: ${#ML_CONFIGS[@]} (X_geno, X_geno_pc, X_geno_dummy, X_geno_onehot, X_geno_pc_dummy, X_geno_pc_onehot)"
echo "Total combinations: $((${#GWAS_CONFIGS[@]} * ${#ML_CONFIGS[@]}))"
echo ""
echo "Data directory: $BASE_DATA_DIR"
echo "Output directory: $BASE_OUTPUT_DIR"