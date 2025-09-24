<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# README_Streamlit.md

# 💓 Análise de Doença Cardíaca - Classificação Binária com Balanceamento e Otimização

Este projeto implementa uma aplicação interativa em Streamlit para análise e classificação binária de doença cardíaca, comparando duas abordagens principais de balanceamento de classes: **Ponderação de Classe** e **Oversampling com SMOTE**. A aplicação utiliza múltiplos classificadores, com opção de otimização de hiperparâmetros para maximizar a performance preditiva.

***

## Funcionalidades

- Carregamento e processamento de dados de entradas para classificação binária (0 = sem doença, 1 = presença de doença).
- Comparação de abordagens de balanceamento de classes:
    - Ponderação de classes para lidar com desbalanceamento.
    - Oversampling usando SMOTE para balancear as classes sinteticamente.
- Treinamento e otimização automática dos modelos usando GridSearchCV com validação cruzada.
- Avaliação de múltiplos classificadores populares:
    - Random Forest, SVM, KNN, Regressão Logística, Decision Tree, XGBoost, Gradient Boosting, MLP.
- Visualização interativa dos resultados, incluindo:
    - Distribuição das classes antes e depois do balanceamento.
    - Resultados da otimização (melhores hiperparâmetros e scores ROC-AUC).
    - Métricas de validação cruzada (Acurácia, AUC, F1-Score).
    - Matrizes de confusão com sensibilidade e especificidade.
    - Curvas ROC para comparação dos modelos.
    - Tabela comparativa final dos modelos com principais métricas.
- Recomendações médicas baseadas nos resultados, destacando a sensibilidade do melhor modelo.

***

## Requisitos

- Python 3.8+
- Bibliotecas:
    - streamlit
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - xgboost
    - scikit-learn
    - imbalanced-learn (para SMOTE)

***

## Como usar

1. Clone o repositório.
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Garanta que os arquivos de dados `heart_train_processed.csv` e `heart_test_processed.csv` estejam na mesma pasta do script.
4. Execute a aplicação Streamlit:

```bash
streamlit run modelagem_app.py
```

5. Use a barra lateral para configurar a abordagem de balanceamento, ativar/desativar otimização de hiperparâmetros e definir os folds para validação.
6. Clique em "Iniciar Avaliação" para rodar a análise completa.

***

## Estrutura do código

- **carregar_e_converter_dados()**: Carrega os dados e prepara a variável target binária.
- **obter_classificadores_e_parametros_binarios()**: Define os classificadores e seus grids de hiperparâmetros para cada abordagem.
- **aplicar_smote_binario()**: Aplica SMOTE para balanceamento das classes.
- **otimizar_com_gridsearch()**: Realiza otimização dos hiperparâmetros via GridSearchCV.
- **executar_validacao_cruzada_binaria()**: Executa validação cruzada para avaliação robusta dos modelos.
- **treinar_e_avaliar_binario()**: Treina e avalia os modelos finalizados com dados de teste.
- Funções de visualização e métricas são usadas para análise detalhada dos resultados.

***

## Classificação Binária

- Classe 0: Ausência de doença cardíaca
- Classe 1: Presença de doença cardíaca
- Métricas importantes:
    - AUC (Área sob a curva ROC)
    - Sensibilidade (Recall)
    - Especificidade
- Relevância médica na priorização da sensibilidade para garantir a detecção dos casos positivos.

***

## Contatos e Contribuições

Contribuições são bem-vindas! Abra uma issue ou envie um pull request para melhorias.

***


