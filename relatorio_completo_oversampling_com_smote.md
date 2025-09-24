# Relatório Completo de Avaliação de Modelos

Gerado em: 2025-09-24 20:08:27

**Abordagem de Balanceamento:** Oversampling com SMOTE

**Otimização de Hiperparâmetros:** Sim

---

## 🏆 Resultados Finais no Conjunto de Teste

| Modelo                      |   Acurácia |   Precisão |   Recall (Sensibilidade) |   F1-Score |      AUC |
|:----------------------------|-----------:|-----------:|-------------------------:|-----------:|---------:|
| SVM                         |   0.847826 |   0.836364 |                 0.901961 |   0.867925 | 0.926829 |
| Gradient Boosting           |   0.815217 |   0.84     |                 0.823529 |   0.831683 | 0.925693 |
| XGBoost                     |   0.847826 |   0.842593 |                 0.892157 |   0.866667 | 0.922645 |
| RidgeClassifier             |   0.847826 |   0.849057 |                 0.882353 |   0.865385 | 0.92121  |
| LinearSVC                   |   0.842391 |   0.841121 |                 0.882353 |   0.861244 | 0.919895 |
| Logistic Regression         |   0.842391 |   0.847619 |                 0.872549 |   0.859903 | 0.918341 |
| SGDClassifier               |   0.847826 |   0.855769 |                 0.872549 |   0.864078 | 0.916069 |
| KNN                         |   0.853261 |   0.871287 |                 0.862745 |   0.866995 | 0.905069 |
| Random Forest               |   0.831522 |   0.838095 |                 0.862745 |   0.850242 | 0.900167 |
| MLP                         |   0.836957 |   0.827273 |                 0.892157 |   0.858491 | 0.886657 |
| PassiveAggressiveClassifier |   0.804348 |   0.823529 |                 0.823529 |   0.823529 | 0.879842 |
| Perceptron                  |   0.788043 |   0.846154 |                 0.754902 |   0.797927 | 0.86956  |
| Decision Tree               |   0.777174 |   0.80198  |                 0.794118 |   0.79803  | 0.853001 |



## 🔄 Resultados da Validação Cruzada

| Modelo                      |   Acurácia Média (CV) |   AUC Médio (CV) |   F1-Score Médio (CV) |
|:----------------------------|----------------------:|-----------------:|----------------------:|
| XGBoost                     |              0.81817  |         0.898733 |              0.818516 |
| Random Forest               |              0.831667 |         0.898351 |              0.834305 |
| KNN                         |              0.829228 |         0.897446 |              0.828211 |
| Gradient Boosting           |              0.82304  |         0.894724 |              0.822893 |
| Logistic Regression         |              0.807067 |         0.889807 |              0.807482 |
| SGDClassifier               |              0.807074 |         0.889565 |              0.807532 |
| RidgeClassifier             |              0.807089 |         0.888839 |              0.805996 |
| LinearSVC                   |              0.805832 |         0.888293 |              0.80514  |
| SVM                         |              0.804605 |         0.887371 |              0.806506 |
| MLP                         |              0.798485 |         0.865102 |              0.799604 |
| PassiveAggressiveClassifier |              0.772696 |         0.846619 |              0.776518 |
| Decision Tree               |              0.776399 |         0.841124 |              0.778804 |
| Perceptron                  |              0.755525 |         0.823517 |              0.749819 |



## 🔍 Matrizes de Confusão (Teste)

Formato: [[Verdadeiro Negativo, Falso Positivo], [Falso Negativo, Verdadeiro Positivo]]

### Random Forest

```
[[65 17]
 [14 88]]
```

### SVM

```
[[64 18]
 [10 92]]
```

### KNN

```
[[69 13]
 [14 88]]
```

### Logistic Regression

```
[[66 16]
 [13 89]]
```

### Decision Tree

```
[[62 20]
 [21 81]]
```

### XGBoost

```
[[65 17]
 [11 91]]
```

### Gradient Boosting

```
[[66 16]
 [18 84]]
```

### MLP

```
[[63 19]
 [11 91]]
```

### Perceptron

```
[[68 14]
 [25 77]]
```

### LinearSVC

```
[[65 17]
 [12 90]]
```

### SGDClassifier

```
[[67 15]
 [13 89]]
```

### RidgeClassifier

```
[[66 16]
 [12 90]]
```

### PassiveAggressiveClassifier

```
[[64 18]
 [18 84]]
```



## 🔧 Melhores Hiperparâmetros

### Random Forest

```json
{'classifier__max_depth': None, 'classifier__min_samples_split': 5, 'classifier__n_estimators': 100}
```

### SVM

```json
{'classifier__C': 0.1, 'classifier__gamma': 'scale', 'classifier__kernel': 'linear'}
```

### KNN

```json
{'classifier__n_neighbors': 9, 'classifier__weights': 'distance'}
```

### Logistic Regression

```json
{'classifier__C': 0.1, 'classifier__solver': 'liblinear'}
```

### Decision Tree

```json
{'classifier__max_depth': 5, 'classifier__min_samples_split': 10}
```

### XGBoost

```json
{'classifier__learning_rate': 0.1, 'classifier__max_depth': 3, 'classifier__n_estimators': 50}
```

### Gradient Boosting

```json
{'classifier__learning_rate': 0.01, 'classifier__max_depth': 3, 'classifier__n_estimators': 200}
```

### MLP

```json
{'classifier__alpha': 0.01, 'classifier__hidden_layer_sizes': (50,)}
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
{'classifier__alpha': 0.01, 'classifier__loss': 'log_loss'}
```

### RidgeClassifier

```json
{'classifier__alpha': 10.0}
```

### PassiveAggressiveClassifier

```json
{'classifier__C': 0.1, 'classifier__max_iter': 1000}
```
