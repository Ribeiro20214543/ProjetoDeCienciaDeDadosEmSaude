# Relatório Completo de Avaliação de Modelos

Gerado em: 2025-09-25 09:26:45

**Abordagem de Balanceamento:** Oversampling com SMOTE

**Otimização de Hiperparâmetros:** Sim

---

##Resultados Finais no Conjunto de Teste

| Modelo                      |   Acurácia |   Precisão |   Recall |   F1-Score |      AUC |
|:----------------------------|-----------:|-----------:|---------:|-----------:|---------:|
| SVM                         |   0.847826 |   0.836364 | 0.901961 |   0.867925 | 0.926829 |
| Gradient Boosting           |   0.831522 |   0.84466  | 0.852941 |   0.84878  | 0.925335 |
| SGDClassifier               |   0.831522 |   0.865979 | 0.823529 |   0.844221 | 0.923242 |
| XGBoost                     |   0.847826 |   0.842593 | 0.892157 |   0.866667 | 0.922645 |
| RidgeClassifier             |   0.847826 |   0.849057 | 0.882353 |   0.865385 | 0.92121  |
| LinearSVC                   |   0.842391 |   0.841121 | 0.882353 |   0.861244 | 0.919895 |
| Logistic Regression         |   0.842391 |   0.847619 | 0.872549 |   0.859903 | 0.918341 |
| Random Forest               |   0.826087 |   0.824074 | 0.872549 |   0.847619 | 0.914156 |
| KNN                         |   0.847826 |   0.87     | 0.852941 |   0.861386 | 0.896102 |
| MLP                         |   0.836957 |   0.827273 | 0.892157 |   0.858491 | 0.885342 |
| PassiveAggressiveClassifier |   0.826087 |   0.864583 | 0.813725 |   0.838384 | 0.878527 |
| Perceptron                  |   0.788043 |   0.846154 | 0.754902 |   0.797927 | 0.86956  |
| Decision Tree               |   0.782609 |   0.803922 | 0.803922 |   0.803922 | 0.857425 |



##Resultados da Validação Cruzada

| Modelo                      |   Acurácia Média (CV) |   AUC Médio (CV) |   F1-Score Médio (CV) |
|:----------------------------|----------------------:|-----------------:|----------------------:|
| XGBoost                     |              0.81817  |         0.898733 |              0.818516 |
| Random Forest               |              0.834144 |         0.897341 |              0.835274 |
| KNN                         |              0.820617 |         0.894986 |              0.820544 |
| Gradient Boosting           |              0.825532 |         0.893964 |              0.825463 |
| Logistic Regression         |              0.807067 |         0.889807 |              0.807482 |
| RidgeClassifier             |              0.807089 |         0.888839 |              0.805996 |
| LinearSVC                   |              0.805832 |         0.888293 |              0.80514  |
| SVM                         |              0.804605 |         0.887371 |              0.806506 |
| SGDClassifier               |              0.803454 |         0.878764 |              0.80493  |
| MLP                         |              0.799697 |         0.863407 |              0.801522 |
| Decision Tree               |              0.770264 |         0.829863 |              0.772881 |
| Perceptron                  |              0.755525 |         0.823517 |              0.749819 |
| PassiveAggressiveClassifier |              0.745641 |         0.821328 |              0.735081 |



## Melhores Hiperparâmetros

### Random Forest

```json
{'classifier__max_depth': 10, 'classifier__n_estimators': 100}
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
