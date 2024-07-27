#!/bin/bash
#SBATCH --job-name=Soccermap                                                    # Nome do job
#SBATCH --output=/home_cerberus/disk2/diogochaves/FUTEBOL/Simplified_EPV_in_PFF_Data/results/logs/%j.out # Saída de logs
#SBATCH --error=/home_cerberus/disk2/diogochaves/FUTEBOL/Simplified_EPV_in_PFF_Data/results/metrics/%j.err    # Saída de erro 
#SBATCH -N 1                                                                  # Número de nós
#SBATCH --nodelist=gorgona4                                                   # Solicitar nó específico, passado como argumento
#SBATCH --time=90:00:00                                                       # Tempo máximo

# ------------ PATH CONFIGS ----------
# Inicialmente, definir SOURCE_DIR para um valor padrão para evitar erros
SOURCE_DIR="/home_cerberus/disk2/diogochaves/FUTEBOL/Simplified_EPV_in_PFF_Data"

# ----------- SCRIPT PARAMETERS ----------
SCRIPTS_FILE=$1  # Nome do arquivo TXT contendo os scripts a serem execuxtados
PROJECT_CHOICE=$2    # Novo argumento para controlar SOURCE_DIR


# Definir o diretório do ambiente virtual com base no nó alocado
VENV_DIR="/scratch/diogochaves/venv_diogochaves"

# Mudar para o diretório\
cd $SOURCE_DIR

module load python3.12.1

# Ativar o ambiente virtual Python
source $VENV_DIR/bin/activate

echo $SCRIPTS_FILE
# Verifica se o arquivo de scripts existe
if [ -f "$SCRIPTS_FILE" ]; then

    # Lê o arquivo e executa cada script
    while IFS= read -r SCRIPT; do
    
        # Vai para pasta certa do projeto
        if [ ! -z "$PROJECT_CHOICE" ]; then
            cd $PROJECT_CHOICE
        fi

        echo "Executando script"
        # Executa script
        echo $SCRIPT
        python3 $SCRIPT

    done < "$SCRIPTS_FILE"
else
    echo "Arquivo de scripts não existe: $SCRIPTS_FILE"
    exit 1
fi

# Desativar o ambiente virtual
deactivate
