#!/bin/bash
#SBATCH --job-name=Soccermap                                                    # Nome do job
#SBATCH --output=/sonic_home/diogochaves/CausalEPV/results/logs/%j.out         # Saída de logs
#SBATCH --error=/sonic_home/diogochaves/CausalEPV/results/logs/%j.err          # Saída de erro 
#SBATCH -N 1                                                                  # Número de nós
#SBATCH --nodelist=gorgona5                                                   # Solicitar nó específico
#SBATCH --time=120:00:00                                                      # Tempo máximo

# ------------ PATH CONFIGS ----------
SOURCE_DIR="/sonic_home/diogochaves/CausalEPV"

# Definir o diretório do ambiente virtual
VENV_DIR="/sonic_home/diogochaves/.venv"

# Mudar para o diretório
cd $SOURCE_DIR

# Ativar o ambiente virtual Python
source $VENV_DIR/bin/activate

# ------------ MODELOS A EXECUTAR ----------
MODELS=(
    "src/Pass/Pass_epv_success/main.py"
    "src/Pass/Pass_epv_missed/main.py"
    "src/Pass/Pass_selection_probability/main.py"
    "src/Pass/Pass_sucess_probability/main.py"
)

echo "========== Iniciando execução dos modelos =========="

# Executar cada modelo
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "Executando: $MODEL"
    python3 "$MODEL" train
    if [ $? -eq 0 ]; then
        echo "✓ $MODEL completado com sucesso"
    else
        echo "✗ ERRO ao executar $MODEL"
    fi
done

echo ""
echo "========== Execução finalizada =========="
