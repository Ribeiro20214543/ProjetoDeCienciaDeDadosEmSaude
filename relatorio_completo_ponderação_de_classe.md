# Relatório Completo de Avaliação de Modelos

Gerado em: 2025-09-24 20:05:36

**Abordagem de Balanceamento:** Ponderação de Classe

**Otimização de Hiperparâmetros:** Sim

---

## 🏆 Resultados Finais no Conjunto de Teste

| Modelo                      |   Acurácia |   Precisão |   Recall (Sensibilidade) |   F1-Score |      AUC |
|:----------------------------|-----------:|-----------:|-------------------------:|-----------:|---------:|
| SVM                         |   0.842391 |   0.834862 |                 0.892157 |   0.862559 | 0.925514 |
| Gradient Boosting           |   0.831522 |   0.84466  |                 0.852941 |   0.84878  | 0.925155 |
| RidgeClassifier             |   0.842391 |   0.841121 |                 0.882353 |   0.861244 | 0.924199 |
| Logistic Regression         |   0.847826 |   0.849057 |                 0.882353 |   0.865385 | 0.922764 |
| SGDClassifier               |   0.853261 |   0.850467 |                 0.892157 |   0.870813 | 0.922645 |
| LinearSVC                   |   0.831522 |   0.814159 |                 0.901961 |   0.855814 | 0.922047 |
| XGBoost                     |   0.831522 |   0.851485 |                 0.843137 |   0.847291 | 0.919715 |
| KNN                         |   0.836957 |   0.827273 |                 0.892157 |   0.858491 | 0.914036 |
| Random Forest               |   0.847826 |   0.862745 |                 0.862745 |   0.862745 | 0.907341 |
| PassiveAggressiveClassifier |   0.798913 |   0.857143 |                 0.764706 |   0.80829  | 0.877331 |
| MLP                         |   0.815217 |   0.826923 |                 0.843137 |   0.834951 | 0.87494  |
| Decision Tree               |   0.798913 |   0.835052 |                 0.794118 |   0.81407  | 0.870696 |
| Perceptron                  |   0.75     |   0.77451  |                 0.77451  |   0.77451  | 0.828551 |



## 🔄 Resultados da Validação Cruzada

| Modelo                      |   Acurácia Média (CV) |   AUC Médio (CV) |   F1-Score Médio (CV) |
|:----------------------------|----------------------:|-----------------:|----------------------:|
| Logistic Regression         |              0.801609 |         0.890262 |              0.816109 |
| LinearSVC                   |              0.812493 |         0.889439 |              0.831945 |
| SGDClassifier               |              0.796176 |         0.889404 |              0.810824 |
| SVM                         |              0.797536 |         0.888861 |              0.81248  |
| RidgeClassifier             |              0.798878 |         0.88844  |              0.81276  |
| XGBoost                     |              0.813863 |         0.880682 |              0.830947 |
| Gradient Boosting           |              0.81389  |         0.879137 |              0.836696 |
| Random Forest               |              0.807069 |         0.876943 |              0.825391 |
| KNN                         |              0.803006 |         0.872538 |              0.828161 |
| PassiveAggressiveClassifier |              0.782625 |         0.841288 |              0.799521 |
| MLP                         |              0.760857 |         0.840822 |              0.784983 |
| Perceptron                  |              0.745882 |         0.816327 |              0.765638 |
| Decision Tree               |              0.771732 |         0.810049 |              0.78926  |



## 🔍 Matrizes de Confusão (Teste)

Formato: [[Verdadeiro Negativo, Falso Positivo], [Falso Negativo, Verdadeiro Positivo]]

### Random Forest

```
[[68 14]
 [14 88]]
```

### SVM

```
[[64 18]
 [11 91]]
```

### KNN

```
[[63 19]
 [11 91]]
```

### Logistic Regression

```
[[66 16]
 [12 90]]
```

### Decision Tree

```
[[66 16]
 [21 81]]
```

### XGBoost

```
[[67 15]
 [16 86]]
```

### Gradient Boosting

```
[[66 16]
 [15 87]]
```

### MLP

```
[[64 18]
 [16 86]]
```

### Perceptron

```
[[59 23]
 [23 79]]
```

### LinearSVC

```
[[61 21]
 [10 92]]
```

### SGDClassifier

```
[[66 16]
 [11 91]]
```

### RidgeClassifier

```
[[65 17]
 [12 90]]
```

### PassiveAggressiveClassifier

```
[[69 13]
 [24 78]]
```



## 🔧 Melhores Hiperparâmetros

### Random Forest

```json
{'classifier__max_depth': 5, 'classifier__min_samples_split': 5, 'classifier__n_estimators': 100}
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
{'classifier__alpha': 0.001, 'classifier__hidden_layer_sizes': (50,)}
```

### Perceptron

```json
{'classifier__eta0': 0.0001, 'classifier__max_iter': 1000}
```

### LinearSVC

```json
{'classifier__C': 0.01}
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
