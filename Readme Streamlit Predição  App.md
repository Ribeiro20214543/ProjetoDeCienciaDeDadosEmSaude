# Sistema de Predição de Doença Cardíaca

## 🫀 Sobre o Projeto

Este é um sistema web interativo desenvolvido com Streamlit para predição de risco de doença cardíaca baseado em características clínicas do paciente. O sistema utiliza um modelo de Machine Learning (SVM) treinado com técnica SMOTE para fornecer predições precisas.

## 🚀 Como Executar

### Pré-requisitos

1. Python 3.8 ou superior instalado
2. Os arquivos do modelo treinado:
   - `preprocessor.joblib`
   - `svm_model_smote.joblib`

### Instalação

1. Clone ou baixe os arquivos do projeto
2. Instale as dependências:
```bash
pip install -r requirements.txt
```

### Execução

Execute o seguinte comando no terminal:
```bash
streamlit run predicao_cardiaca_app.py
```

O sistema será aberto automaticamente no seu navegador em `http://localhost:8501`

## 📋 Como Usar

1. **Preencha os dados do paciente** no formulário à esquerda:
   - Dados numéricos: idade, pressão arterial, colesterol, etc.
   - Dados categóricos: sexo, tipo de dor no peito, etc.

2. **Clique em "Realizar Predição"** para processar os dados

3. **Analise os resultados** na coluna direita:
   - Probabilidade de risco em percentual
   - Classificação de risco (Alto/Baixo)
   - Recomendações médicas

## 🔧 Funcionalidades

- **Interface intuitiva** com formulário organizado
- **Validação de dados** com limites apropriados
- **Visualização clara** dos resultados
- **Recomendações** baseadas no nível de risco
- **Glossário** de termos médicos na sidebar
- **Cache** para carregamento eficiente dos modelos

## ⚠️ Importante

Este sistema é uma ferramenta de **apoio à decisão médica** e **não substitui a consulta médica profissional**. Sempre procure orientação médica adequada.

## 🛠️ Arquivos Necessários

- `predicao_cardiaca_app.py` - Aplicação principal
- `preprocessor.joblib` - Pré-processador de dados treinado
- `svm_model_smote.joblib` - Modelo SVM treinado
- `requirements.txt` - Dependências do projeto

## 📊 Dados de Entrada

O sistema aceita as seguintes variáveis:

### Numéricas:
- **age**: Idade do paciente
- **trestbps**: Pressão arterial em repouso (mmHg)
- **chol**: Colesterol sérico (mg/dl)
- **thalch**: Frequência cardíaca máxima alcançada
- **oldpeak**: Depressão do ST induzida por exercício

### Categóricas:
- **sex**: Sexo (Male/Female)
- **dataset**: Base de dados de origem
- **cp**: Tipo de dor no peito
- **fbs**: Açúcar no sangue em jejum > 120 mg/dl
- **restecg**: Resultados do eletrocardiograma em repouso
- **exang**: Angina induzida por exercício
- **slope**: Inclinação do segmento ST

## 🎯 Modelo

- **Algoritmo**: SVM (Support Vector Machine)
- **Técnica de balanceamento**: SMOTE
- **Pré-processamento**: Normalização + One-Hot Encoding
"""
