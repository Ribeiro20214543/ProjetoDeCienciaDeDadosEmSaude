
# 💓 Análise Binária de Doença Cardíaca

Aplicação web interativa desenvolvida com Streamlit para análise e comparação de múltiplos classificadores na tarefa de detecção binária de doença cardíaca. O foco é comparar abordagens de balanceamento de classes (ponderação e oversampling com SMOTE) aliadas a otimização de hiperparâmetros para identificar o melhor modelo preditivo.

***

## Funcionalidades

- Carregamento e pré-processamento dos dados para classificação binária (0 = sem doença, 1 = presença de doença).
- Implementação de diversos classificadores populares: Random Forest, SVM, KNN, Logistic Regression, Decision Tree, XGBoost, Gradient Boosting, MLP, Perceptron, LinearSVC, SGDClassifier, RidgeClassifier e PassiveAggressiveClassifier.
- Comparação entre duas abordagens de balanceamento:
    - Ponderação de classe automática nos classificadores.
    - Oversampling usando SMOTE para reamostragem de classe minoritária.
- Otimização de hiperparâmetros via GridSearchCV com múltiplos folds configuráveis.
- Validação cruzada estratificada com métricas completas: acurácia, AUC (area under ROC curve), F1-score.
- Avaliação final em conjunto de teste com geração de métricas (acurácia, precisão, recall, F1, AUC, matriz de confusão).
- Visualizações gráficas interativas:
    - Distribuição das classes original e pós-balanceamento.
    - Resultados da otimização.
    - Métricas da validação cruzada.
    - Matrizes de confusão.
    - Curvas ROC comparativas.
    - Tabela comparativa final dos modelos.
- Relatórios para download em formatos texto e Markdown contendo resultados da avaliação, matrizes de confusão e melhores hiperparâmetros.
- Recomendações explicativas voltadas a interpretação médica do desempenho do melhor modelo, com foco em sensibilidade (recall).

***

## Requisitos

- Python 3.8 ou superior
- Bibliotecas Python:
    - streamlit
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scikit-learn
    - imbalanced-learn
    - xgboost
- Arquivos de dados:
    - `heart_train_processed.csv`
    - `heart_test_processed.csv`

***

## Como executar

1. Certifique-se de ter instalado as dependências utilizando:
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

2. Coloque os arquivos `heart_train_processed.csv` e `heart_test_processed.csv` no mesmo diretório do script.
3. Execute a aplicação Streamlit com:
```bash
streamlit run modelagem_app.py
```

4. Acesse o app via navegador no endereço exibido no terminal (geralmente http://localhost:8501).
5. Use a barra lateral para configurar:
    - Abordagem de balanceamento (ponderação de classe ou SMOTE)
    - Uso ou não da otimização de hiperparâmetros
    - Número de folds para otimização e validação final
6. Clique em "Iniciar Avaliação" para executar os processos de treinamento, validação e avaliação.
7. Visualize os resultados gráficos e métricas detalhadas.
8. Baixe relatórios resumidos ou completos para análise offline.

***

## Sobre a Classificação Binária

- Classe 0: Ausência de doença cardíaca.
- Classe 1: Presença de doença cardíaca.
- AUC: Área sob a curva ROC (0.5 = aleatório, 1.0 = perfeito).
- Sensibilidade (Recall): Percentual dos casos de doença corretamente detectados.
- Especificidade: Percentual dos casos saudáveis corretamente identificados.
- Na aplicação médica, prioriza-se alta sensibilidade para reduzir falsos negativos (pacientes doentes não detectados).

***

## Estrutura do Código

- Carregamento e processamento dos dados com conversão para formato binário.
- Definição de classificadores e seus grids de hiperparâmetros para otimização.
- Funções para aplicação de SMOTE.
- Otimização via GridSearchCV com pipelines padrão (normalização + classificador).
- Validação cruzada estratificada com cálculo de métricas detalhadas.
- Treinamento final e avaliação dos modelos no conjunto teste.
- Visualização e exibição dos resultados e gráficos no Streamlit.
- Geração e download de relatórios completos em texto e markdown.

***

## Contato

Este projeto foi desenvolvido para fins acadêmicos e pesquisa sobre classificação e balanceamento de dados para detecção de doenças cardíacas. Dúvidas e sugestões podem ser encaminhadas ao responsável pelo repositório.

***

### Nota

O arquivo original foi criado e editado no Google Colab e convertido para aplicação Streamlit para visualização interativa e análise detalhada de modelos de machine learning para doenças cardíacas.

***

# Licença

Livre para uso acadêmico e de pesquisa.


