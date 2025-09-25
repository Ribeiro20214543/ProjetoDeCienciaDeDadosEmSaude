# Relatório Completo de Avaliação de Modelos

Gerado em: 2025-09-25 09:27:48

**Abordagem de Balanceamento:** Ponderação de Classe

**Otimização de Hiperparâmetros:** Sim

---

## Resultados Finais no Conjunto de Teste

| Modelo                      |   Acurácia |   Precisão |   Recall |   F1-Score |      AUC |
|:----------------------------|-----------:|-----------:|---------:|-----------:|---------:|
| SVM                         |   0.842391 |   0.834862 | 0.892157 |   0.862559 | 0.925514 |
| RidgeClassifier             |   0.842391 |   0.841121 | 0.882353 |   0.861244 | 0.924199 |
| LinearSVC                   |   0.836957 |   0.815789 | 0.911765 |   0.861111 | 0.923123 |
| Logistic Regression         |   0.847826 |   0.849057 | 0.882353 |   0.865385 | 0.922764 |
| XGBoost                     |   0.831522 |   0.851485 | 0.843137 |   0.847291 | 0.919715 |
| Gradient Boosting           |   0.826087 |   0.836538 | 0.852941 |   0.84466  | 0.918699 |
| SGDClassifier               |   0.842391 |   0.823009 | 0.911765 |   0.865116 | 0.917384 |
| Random Forest               |   0.847826 |   0.855769 | 0.872549 |   0.864078 | 0.910689 |
| KNN                         |   0.842391 |   0.834862 | 0.892157 |   0.862559 | 0.899868 |
| MLP                         |   0.815217 |   0.826923 | 0.843137 |   0.834951 | 0.87494  |
| Decision Tree               |   0.798913 |   0.835052 | 0.794118 |   0.81407  | 0.870696 |
| Perceptron                  |   0.75     |   0.77451  | 0.77451  |   0.77451  | 0.828551 |
| PassiveAggressiveClassifier |   0.728261 |   0.75     | 0.764706 |   0.757282 | 0.794357 |



## Resultados da Validação Cruzada

| Modelo                      |   Acurácia Média (CV) |   AUC Médio (CV) |   F1-Score Médio (CV) |
|:----------------------------|----------------------:|-----------------:|----------------------:|
| Logistic Regression         |              0.801609 |         0.890262 |              0.816109 |
| SVM                         |              0.797536 |         0.888861 |              0.81248  |
| RidgeClassifier             |              0.798878 |         0.88844  |              0.81276  |
| LinearSVC                   |              0.811133 |         0.888254 |              0.830835 |
| XGBoost                     |              0.813863 |         0.880682 |              0.830947 |
| Random Forest               |              0.807079 |         0.876498 |              0.827242 |
| Gradient Boosting           |              0.808421 |         0.874768 |              0.830267 |
| KNN                         |              0.801636 |         0.868928 |              0.826629 |
| SGDClassifier               |              0.800303 |         0.867187 |              0.816734 |
| MLP                         |              0.760857 |         0.840822 |              0.784983 |
| PassiveAggressiveClassifier |              0.747288 |         0.825135 |              0.76812  |
| Perceptron                  |              0.745882 |         0.816327 |              0.765638 |
| Decision Tree               |              0.773092 |         0.801171 |              0.791953 |



## Melhores Hiperparâmetros

### Random Forest

```json
{'classifier__max_depth': 5, 'classifier__n_estimators': 100}
```

### SVM

```json
{'classifier__C': 0.1, 'classifier__kernel': 'linear'}
```

### KNN

```json
{'classifier__n_neighbors': 7, 'classifier__weights': 'distance'}
```

### Logistic Regression

```json
{'classifier__C': 0.1, 'classifier__solver': 'liblinear'}
```

### Decision Tree

```json
{'classifier__max_depth': 5, 'classifier__min_samples_split': 5}
```

### XGBoost

```json
{'classifier__learning_rate': 0.1, 'classifier__max_depth': 3, 'classifier__n_estimators': 50}
```

### Gradient Boosting

```json
{'classifier__learning_rate': 0.1, 'classifier__max_depth': 3, 'classifier__n_estimators': 50}
```

### MLP

```json
{'classifier__alpha': 0.001, 'classifier__hidden_layer_sizes': (50,)}
```

### Perceptron

```json
{'classifier__eta0': 0.0001, 'classifier__max_iter': 1000}
```

### LinearSVC

```json
{'classifier__C': 0.1}
```

### SGDClassifier

```json
{'classifier__alpha': 0.001, 'classifier__loss': 'log_loss'}
```

### RidgeClassifier

```json
{'classifier__alpha': 10.0}
```

### PassiveAggressiveClassifier

```json
{'classifier__C': 0.5, 'classifier__max_iter': 1000}
```

