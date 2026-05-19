# CausalEPV - Run Script & Output Paths Documentation

## Overview
O arquivo `run.sh` foi configurado para executar todos os 4 modelos Pass do CausalEPV de forma automatizada e organizada.

## Models Executados

### 1. **Pass_epv_success**
- **Localização**: `src/Pass/Pass_epv_success/`
- **Tipo**: EPV (Expected Passing Value) para passes bem-sucedidos
- **Entrada de dados**: `data/passes/` (treino com passes completados - "C")
- **Modo**: train (pode ser expandido para train_test)

### 2. **Pass_epv_missed**
- **Localização**: `src/Pass/Pass_epv_missed/`
- **Tipo**: EPV para passes falhados
- **Entrada de dados**: `data/passes/` (treino com passes falhados)
- **Modo**: train (pode ser expandido para train_test)

### 3. **Pass_selection_probability**
- **Localização**: `src/Pass/Pass_selection_probability/`
- **Tipo**: Probabilidade de seleção de passe
- **Entrada de dados**: `data/passes/`
- **Modo**: train

### 4. **Pass_sucess_probability**
- **Localização**: `src/Pass/Pass_sucess_probability/`
- **Tipo**: Probabilidade de sucesso de passe
- **Entrada de dados**: `data/passes/`
- **Modo**: train

---

## ✅ Verificação de Diretórios de Saída

### Modelos Salvos (*.pt files)
```
results/
├── models/
│   ├── pass_epv_success/
│   │   └── Pass_epv_success.pt          ← Modelo EPV para passes bem-sucedidos
│   ├── pass_epv_missed/
│   │   └── Pass_epv_missed.pt           ← Modelo EPV para passes falhados
│   ├── Pass_selection_probability.pt    ← Modelo de seleção
│   └── Pass_success_probability.pt      ← Modelo de probabilidade de sucesso
```

**Path Absoluto**: `/sonic_home/diogochaves/CausalEPV/results/models/`

### Loss Plots
```
results/
└── loss/
    ├── Pass_epv_success_loss.png
    ├── Pass_epv_missed_loss.png
    ├── Pass_selection_probability_loss.png
    └── Pass_success_probability_loss_loss.png
```

**Path Absoluto**: `/sonic_home/diogochaves/CausalEPV/results/loss/`

### Métricas & Outros
```
results/
├── metrics/                 ← Arquivo de métricas de treino/validação
├── heatmaps/               ← Mapas de calor de predições
└── [outros outputs]
```

**Paths Absolutos**:
- Métricas: `/sonic_home/diogochaves/CausalEPV/results/metrics/`
- Heatmaps: `/sonic_home/diogochaves/CausalEPV/results/heatmaps/`

---

## 🔍 How Results Are Saved

Cada trainer implementa a mesma lógica de path resolution:

```python
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# Exemplo: /sonic_home/diogochaves/CausalEPV

def _resolve_repo_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)

# Paths relativos são convertidos:
# "results/models/pass_epv_success" → "/sonic_home/diogochaves/CausalEPV/results/models/pass_epv_success"
```

### Default Save Paths por Modelo:

| Modelo | Diretório de Modelos | Diretório de Loss |
|--------|----------------------|-------------------|
| Pass_epv_success | `results/models/pass_epv_success` | `results/loss` |
| Pass_epv_missed | `results/models/pass_epv_missed` | `results/loss` |
| Pass_selection_probability | `results/models` | `results/loss` |
| Pass_sucess_probability | `results/models` | `results/loss` |

---

## 📋 Pre-Execution Checks Performed

O `run.sh` executa os seguintes checks antes de rodar os modelos:

✅ Verifica se diretório do projeto existe  
✅ Verifica se diretório `src/Pass` existe  
✅ **Cria automaticamente** todos os diretórios necessários em `results/`:
- `results/models/`
- `results/models/pass_epv_success/`
- `results/models/pass_epv_missed/`
- `results/loss/`
- `results/metrics/`
- `results/heatmaps/`

✅ Valida presença de dados em `data/passes/`  
✅ Valida presença de eventos em `data/raw/event/`  
✅ Ativa venv (se disponível em `/scratch/diogochaves/venv_diogochaves`)

---

## 🚀 Como Usar

### Executar todos os modelos (interativo):
```bash
bash run.sh
```

### Submeter como job SLURM (recomendado para HPC):
```bash
sbatch run.sh
```

O script irá automaticamente:
```bash
# Pass_epv_success
python src/Pass/Pass_epv_success/main.py train

# Pass_epv_missed
python src/Pass/Pass_epv_missed/main.py train

# Pass_selection_probability
python src/Pass/Pass_selection_probability/main.py train

# Pass_sucess_probability
python src/Pass/Pass_sucess_probability/main.py train
```

### Modos disponíveis:
- `train` - Apenas treina o modelo
- `test` - Apenas testa o modelo
- `train_test` - Treina e depois testa

---

## 📊 Output Summary

O `run.sh` fornece ao final:

1. **Resumo de Execução**: Lista quais modelos completaram com sucesso/falharam
2. **Verificação de Outputs**: Confirma quais arquivos foram criados
3. **Árvore de Diretórios**: Mostra a estrutura completa de `results/`
4. **Exit Code**: Retorna 1 se algum modelo falhou, 0 se tudo sucedeu

---

## 📋 SBATCH Headers Configurados

O script contém os seguintes headers SLURM para execução em HPC:

```bash
#SBATCH --job-name=CausalEPV_Models              # Nome do job
#SBATCH --output=results/logs/%j.out             # Logs de stdout
#SBATCH --error=results/logs/%j.err              # Logs de stderr
#SBATCH -N 1                                     # 1 nó
#SBATCH --time=120:00:00                         # Limite de 120 horas
```

**Submissão**: `sbatch run.sh`

Logs de execução serão salvos em:
- `results/logs/<jobid>.out` - Output padrão
- `results/logs/<jobid>.err` - Erros

---

1. **Caminhos de Dados**: Todos os dataloaders foram corrigidos para usar paths absolutos a partir de `REPO_ROOT`
   - `data_directory="data/passes"` → Resolvido para `/sonic_home/diogochaves/CausalEPV/data/passes/`
   - `reward_event_directory="data/raw/event"` → Resolvido para `/sonic_home/diogochaves/CausalEPV/data/raw/event/`

2. **Criação Automática de Diretórios**: O `run.sh` cria automaticamente todos os diretórios necessários antes de executar os treinos

3. **Logs de Execução**: Cada modelo imprime logs durante treino/teste, facilitando debugging

4. **Tratamento de Erros**: Se um modelo falhar, o `run.sh` continua com os próximos, mas retorna código de erro ao final

---

## 📁 Estrutura Esperada de Resultados

Após execução bem-sucedida:

```
results/
├── models/
│   ├── pass_epv_success/
│   │   └── Pass_epv_success.pt
│   ├── pass_epv_missed/
│   │   └── Pass_epv_missed.pt
│   ├── Pass_selection_probability.pt
│   └── Pass_success_probability.pt
├── loss/
│   ├── Pass_epv_success_loss.png
│   ├── Pass_epv_missed_loss.png
│   ├── Pass_selection_probability_loss.png
│   └── Pass_success_probability_loss_loss.png
├── metrics/
└── heatmaps/
```

Total de arquivos esperados: **4 modelos + 4 plots de loss + outros outputs**

---

## 🔧 Troubleshooting

### Se um modelo falhar:
1. Verifique se `data/passes/` tem arquivos
2. Verifique se `data/raw/event/` tem dados de eventos
3. Verifique permissões em `results/` directory
4. Veja o output do modelo específico para detalhes do erro

### Se diretórios não forem criados:
1. Verifique permissões de escrita em `/sonic_home/diogochaves/CausalEPV/results/`
2. Execute manualmente: `mkdir -p /sonic_home/diogochaves/CausalEPV/results/{models/{pass_epv_success,pass_epv_missed},loss,metrics,heatmaps}`

---

**Última Atualização**: 2026-05-11
