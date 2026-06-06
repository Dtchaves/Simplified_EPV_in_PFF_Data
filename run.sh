#!/bin/bash
#SBATCH --job-name=Soccermap                                                    # Nome do job
##SBATCH --output=/sonic_home/diogochaves/CausalEPV/results/logs/%j.out        # Caminho legado
##SBATCH --error=/sonic_home/diogochaves/CausalEPV/results/logs/%j.err         # Caminho legado
#SBATCH --output=/home/joaomarcostomaz/causal/Simplified_EPV_in_PFF_Data/results/logs/%j.out
#SBATCH --error=/home/joaomarcostomaz/causal/Simplified_EPV_in_PFF_Data/results/logs/%j.err
#SBATCH -N 1                                                                  # Número de nós
#SBATCH --nodelist=gorgona5                                                   # Solicitar nó específico
#SBATCH --time=120:00:00                                                      # Tempo máximo

# ------------ PATH CONFIGS ----------
# SOURCE_DIR="/sonic_home/diogochaves/CausalEPV"
SOURCE_DIR="/home/joaomarcostomaz/causal/Simplified_EPV_in_PFF_Data"

# VENV_DIR="/sonic_home/diogochaves/.venv"
VENV_DIR="/home/joaomarcostomaz/causal/Simplified_EPV_in_PFF_Data/.venv"

# Mudar para o diretório
cd "$SOURCE_DIR"

mkdir -p "$SOURCE_DIR/results/logs"

# Ativar o ambiente virtual Python
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$SOURCE_DIR/src:${PYTHONPATH:-}"

# ------------ MODELOS A EXECUTAR ----------
# MODELS=(
#     "src/Pass/Pass_epv_success/main.py"
#     "src/Pass/Pass_epv_missed/main.py"
#     "src/Pass/Pass_selection_probability/main.py"
#     "src/Pass/Pass_sucess_probability/main.py"
# )

MODELS=(
    "src/Pass/Pass_sucess_probability/main.py"
    # "src/Pass/Pass_selection_probability/main.py"
    # "src/Pass/Pass_epv_success/main.py"
    # "src/Pass/Pass_epv_missed/main.py"
    # "src/BallDrive/main.py"
    # "src/Shot/main.py"
    # "src/ActionSelection/main.py"
)

echo "========== Iniciando execução dos modelos =========="

STATUS=0

# Executar cada modelo
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "Executando: $MODEL"
    python3 "$MODEL" test
    if [ $? -eq 0 ]; then
        echo "✓ $MODEL completado com sucesso"
    else
        echo "✗ ERRO ao executar $MODEL"
        STATUS=1
    fi
done

echo ""
echo "========== Execução finalizada =========="
exit $STATUS
