import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import warnings
import datetime

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import (
    LogisticRegression, Perceptron, SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, roc_curve, average_precision_score
)

warnings.filterwarnings('ignore')


# Configuração da página
st.set_page_config(
    page_title="Análise Binária de Doença Cardíaca",
    page_icon="💓",
    layout="wide"
)


st.title("💓 Análise de Doença Cardíaca")
st.markdown("### Comparação de Abordagens de Balanceamento com Otimização")
st.markdown("**Classes**: 0 (Sem doença) vs 1 (Presença de doença)")
st.markdown("---")


@st.cache_data
def carregar_e_converter_dados():
    """Carrega dados e converte para classificação binária"""
    try:
        train_df = pd.read_csv('heart_train_processed.csv')
        test_df = pd.read_csv('heart_test_processed.csv')

        # Converter para binário (0 = sem doença, 1+ = com doença)
        train_df['num_reag'] = (train_df['num_reag'] > 0).astype(int)
        test_df['num_reag'] = (test_df['num_reag'] > 0).astype(int)

        # Separar features e target
        X_train = train_df.drop('num_reag', axis=1)
        y_train = train_df['num_reag']
        X_test = test_df.drop('num_reag', axis=1)
        y_test = test_df['num_reag']

        return X_train, y_train, X_test, y_test
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return None, None, None, None


def obter_classificadores_e_parametros_binarios(abordagem, usar_otimizacao=True):
    """Define classificadores e grids para classificação binária"""

    base_classificadores = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'MLP': MLPClassifier(random_state=42, max_iter=1000),
        'Perceptron': Perceptron(random_state=42),
        'LinearSVC': LinearSVC(random_state=42, dual='auto'),
        'SGDClassifier': SGDClassifier(random_state=42),
        'RidgeClassifier': RidgeClassifier(random_state=42),
        'PassiveAggressiveClassifier': PassiveAggressiveClassifier(random_state=42)
    }

    base_param_grids = {
        'Random Forest': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [5, 10, None],
            'classifier__min_samples_split': [2, 5]
        },
        'SVM': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__kernel': ['rbf', 'linear'],
            'classifier__gamma': ['scale', 'auto']
        },
        'KNN': {
            'classifier__n_neighbors': [3, 5, 7, 9],
            'classifier__weights': ['uniform', 'distance']
        },
        'Logistic Regression': {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__solver': ['liblinear', 'lbfgs']
        },
        'Decision Tree': {
            'classifier__max_depth': [5, 10, 15, None],
            'classifier__min_samples_split': [2, 5, 10]
        },
        'XGBoost': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        },
        'Gradient Boosting': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        },
        'MLP': {
            'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'classifier__alpha': [0.0001, 0.001, 0.01]
        },
        'Perceptron': {
            'classifier__eta0': [0.0001, 0.001, 0.01],
            'classifier__max_iter': [1000, 2000]
        },
        'LinearSVC': {
            'classifier__C': [0.01, 0.1, 1.0, 10.0]
        },
        'SGDClassifier': {
            'classifier__loss': ['hinge', 'log_loss', 'perceptron'],
            'classifier__alpha': [0.0001, 0.001, 0.01]
        },
        'RidgeClassifier': {
            'classifier__alpha': [0.1, 1.0, 10.0, 100.0]
        },
        'PassiveAggressiveClassifier': {
            'classifier__C': [0.1, 0.5, 1.0],
            'classifier__max_iter': [1000, 2000]
        }
    }

    classificadores = base_classificadores.copy()
    param_grids = base_param_grids.copy() if usar_otimizacao else None

    if abordagem == "Ponderação de Classe":
        classificadores['Random Forest'].set_params(class_weight='balanced')
        classificadores['SVM'].set_params(class_weight='balanced')
        classificadores['Logistic Regression'].set_params(class_weight='balanced')
        classificadores['Decision Tree'].set_params(class_weight='balanced')
        classificadores['Perceptron'].set_params(class_weight='balanced')
        classificadores['SGDClassifier'].set_params(class_weight='balanced')
        classificadores['RidgeClassifier'].set_params(class_weight='balanced')
        classificadores['PassiveAggressiveClassifier'].set_params(class_weight='balanced')

        if 'y_train' in st.session_state:
            pos_weight = (st.session_state.y_train == 0).sum() / (st.session_state.y_train == 1).sum()
            classificadores['XGBoost'].set_params(scale_pos_weight=pos_weight)

    if not usar_otimizacao:
        classificadores['Random Forest'].set_params(n_estimators=100)
        classificadores['KNN'].set_params(n_neighbors=5)
        classificadores['XGBoost'].set_params(n_estimators=100)
        classificadores['Gradient Boosting'].set_params(n_estimators=100)
        classificadores['MLP'].set_params(hidden_layer_sizes=(100,))

    return classificadores, param_grids


def aplicar_smote_binario(X_train, y_train):
    """Aplica SMOTE para classificação binária"""
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_reamostrado, y_reamostrado = smote.fit_resample(X_train, y_train)
    return X_reamostrado, y_reamostrado


def otimizar_com_gridsearch(classificadores, param_grids, X_train, y_train, cv_folds=3):
    """Otimiza hiperparâmetros usando GridSearchCV"""
    modelos_otimizados = {}
    resultados_otimizacao = {}

    progress_bar = st.progress(0)
    progress_text = st.empty()

    for i, (nome, classificador) in enumerate(classificadores.items()):
        progress_text.text(f'Otimizando: {nome}')

        try:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', classificador)
            ])

            # Modelos sem predict_proba precisam de uma métrica de pontuação diferente
            models_without_proba = ['RidgeClassifier', 'PassiveAggressiveClassifier', 'LinearSVC', 'Perceptron']
            scoring_metric = 'accuracy' if nome in models_without_proba else 'roc_auc'

            grid_search = GridSearchCV(
                pipeline,
                param_grids[nome],
                cv=cv_folds,
                scoring=scoring_metric,
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X_train, y_train)

            modelos_otimizados[nome] = grid_search.best_estimator_
            resultados_otimizacao[nome] = {
                'best_score': grid_search.best_score_,
                'best_params': grid_search.best_params_,
                'cv_results': grid_search.cv_results_
            }

        except Exception as e:
            st.error(f"Erro na otimização de {nome}: {e}")
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', classificador)
            ])
            modelos_otimizados[nome] = pipeline
            resultados_otimizacao[nome] = {
                'best_score': 0,
                'best_params': {},
                'cv_results': {}
            }

        progress_bar.progress((i + 1) / len(classificadores))

    progress_text.empty()
    progress_bar.empty()

    return modelos_otimizados, resultados_otimizacao


def executar_validacao_cruzada_binaria(modelos, X_train, y_train, cv_folds=5):
    """Validação cruzada com métricas para classificação binária"""
    resultados_cv = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    progress_bar = st.progress(0)
    progress_text = st.empty()

    for i, (nome, modelo) in enumerate(modelos.items()):
        progress_text.text(f'Validação cruzada: {nome}')

        try:
            accuracy_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

            try:
                auc_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            except Exception:
                # Fallback para modelos sem predict_proba
                auc_scores = np.array([0.0] * cv.get_n_splits())

            f1_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)

            resultados_cv[nome] = {
                'Accuracy_Mean': accuracy_scores.mean(),
                'Accuracy_Std': accuracy_scores.std(),
                'AUC_Mean': auc_scores.mean(),
                'AUC_Std': auc_scores.std(),
                'F1_Mean': f1_scores.mean(),
                'F1_Std': f1_scores.std(),
                'Scores': {
                    'accuracy': accuracy_scores,
                    'auc': auc_scores,
                    'f1': f1_scores
                }
            }
        except Exception as e:
            st.error(f"Erro na validação cruzada de {nome}: {e}")
            resultados_cv[nome] = {
                'Accuracy_Mean': 0, 'Accuracy_Std': 0,
                'AUC_Mean': 0, 'AUC_Std': 0,
                'F1_Mean': 0, 'F1_Std': 0,
                'Scores': {'accuracy': [0]*cv_folds, 'auc': [0]*cv_folds, 'f1': [0]*cv_folds}
            }

        progress_bar.progress((i + 1) / len(modelos))

    progress_text.empty()
    progress_bar.empty()

    return resultados_cv


def treinar_e_avaliar_binario(modelos, X_train, y_train, X_test, y_test):
    """Treina e avalia modelos para classificação binária"""
    resultados = {}
    modelos_treinados = {}

    progress_bar = st.progress(0)
    progress_text = st.empty()

    for i, (nome, modelo) in enumerate(modelos.items()):
        progress_text.text(f'Avaliando: {nome}')

        try:
            modelo.fit(X_train, y_train)
            modelos_treinados[nome] = modelo

            y_pred = modelo.predict(X_test)

            y_scores = None
            # Tenta obter probabilidades ou, se não for possível, scores de decisão
            if hasattr(modelo, "predict_proba"):
                y_scores = modelo.predict_proba(X_test)[:, 1]
            elif hasattr(modelo, "decision_function"):
                y_scores = modelo.decision_function(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

            if y_scores is not None:
                auc = roc_auc_score(y_test, y_scores)
                avg_precision = average_precision_score(y_test, y_scores)
            else:
                auc = 0.5
                avg_precision = 0.5

            resultados[nome] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'AUC': auc,
                'Avg_Precision': avg_precision,
                'Predictions': y_pred,
                'Probabilities': y_scores, # Armazena scores ou probabilidades
                'Confusion_Matrix': confusion_matrix(y_test, y_pred)
            }

        except Exception as e:
            st.error(f"Erro ao treinar {nome}: {e}")

        progress_bar.progress((i + 1) / len(modelos))

    progress_text.empty()
    progress_bar.empty()

    return resultados, modelos_treinados


def exibir_distribuicao_classes_binaria(y_train_original, y_train_processado, abordagem):
    """Exibe distribuição das classes binárias"""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuição Original")
        contagens_originais = y_train_original.value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(8, 6))
        barras = ax.bar(['Sem Doença (0)', 'Com Doença (1)'], contagens_originais.values,
                            color=['lightblue', 'lightcoral'])
        ax.set_ylabel('Número de Amostras')
        ax.set_title('Distribuição Original das Classes')

        for bar, count in zip(barras, contagens_originais.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha='center', va='bottom', fontweight='bold')

        total = sum(contagens_originais.values)
        for i, (bar, count) in enumerate(zip(barras, contagens_originais.values)):
            pct = (count/total)*100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'{pct:.1f}%', ha='center', va='center', color='white', fontweight='bold')

        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader(f"Após {abordagem}")
        contagens_processadas = pd.Series(y_train_processado).value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(8, 6))
        barras = ax.bar(['Sem Doença (0)', 'Com Doença (1)'], contagens_processadas.values,
                            color=['lightblue', 'lightcoral'])
        ax.set_ylabel('Número de Amostras')
        ax.set_title(f'Distribuição Após {abordagem}')

        for bar, count in zip(barras, contagens_processadas.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha='center', va='bottom', fontweight='bold')

        total = sum(contagens_processadas.values)
        for i, (bar, count) in enumerate(zip(barras, contagens_processadas.values)):
            pct = (count/total)*100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'{pct:.1f}%', ha='center', va='center', color='white', fontweight='bold')

        st.pyplot(fig)
        plt.close()


def exibir_resultados_otimizacao(resultados_otimizacao):
    """Exibe resultados da otimização"""
    st.subheader("🔧 Resultados da Otimização")

    opt_df = pd.DataFrame({
        'Modelo': list(resultados_otimizacao.keys()),
        'Melhor Score (CV)': [results['best_score'] for results in resultados_otimizacao.values()]
    })

    opt_df = opt_df.sort_values('Melhor Score (CV)', ascending=False)
    opt_df['Melhor Score (CV)'] = opt_df['Melhor Score (CV)'].round(4)

    st.dataframe(
        opt_df.style.highlight_max(subset=['Melhor Score (CV)'], color='lightgreen'),
        use_container_width=True
    )

    st.subheader("📋 Melhores Hiperparâmetros")

    tabs = st.tabs(list(resultados_otimizacao.keys()))
    for tab, (model_name, results) in zip(tabs, resultados_otimizacao.items()):
        with tab:
            if results['best_params']:
                st.json(results['best_params'])
            else:
                st.write("Nenhum parâmetro otimizado")


def exibir_resultados_cv_binario(resultados_cv):
    """Exibe resultados da validação cruzada para classificação binária"""
    st.subheader("📊 Validação Cruzada - Métricas Múltiplas")

    cv_df = pd.DataFrame({
        'Modelo': list(resultados_cv.keys()),
        'Acurácia': [results['Accuracy_Mean'] for results in resultados_cv.values()],
        'AUC': [results['AUC_Mean'] for results in resultados_cv.values()],
        'F1-Score': [results['F1_Mean'] for results in resultados_cv.values()],
        'Acc_Std': [results['Accuracy_Std'] for results in resultados_cv.values()],
        'AUC_Std': [results['AUC_Std'] for results in resultados_cv.values()],
        'F1_Std': [results['F1_Std'] for results in resultados_cv.values()]
    })

    cv_df = cv_df.sort_values('AUC', ascending=False)
    for col in ['Acurácia', 'AUC', 'F1-Score', 'Acc_Std', 'AUC_Std', 'F1_Std']:
        cv_df[col] = cv_df[col].round(4)

    display_cols = ['Modelo', 'Acurácia', 'AUC', 'F1-Score']
    st.dataframe(
        cv_df[display_cols].style.highlight_max(subset=['Acurácia', 'AUC', 'F1-Score'], color='lightgreen'),
        use_container_width=True
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    bars1 = ax1.bar(cv_df['Modelo'], cv_df['Acurácia'], yerr=cv_df['Acc_Std'],
                    capsize=3, color='skyblue', alpha=0.7)
    ax1.set_title('Acurácia')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)

    bars2 = ax2.bar(cv_df['Modelo'], cv_df['AUC'], yerr=cv_df['AUC_Std'],
                    capsize=3, color='lightcoral', alpha=0.7)
    ax2.set_title('ROC-AUC')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)

    bars3 = ax3.bar(cv_df['Modelo'], cv_df['F1-Score'], yerr=cv_df['F1_Std'],
                    capsize=3, color='lightgreen', alpha=0.7)
    ax3.set_title('F1-Score')
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def exibir_matrizes_confusao_binaria(resultados):
    """Exibe matrizes de confusão para classificação binária"""
    st.subheader("🔍 Matrizes de Confusão")

    modelos = list(resultados.keys())
    num_modelos = len(modelos)

    cols = st.columns(min(num_modelos, 4))

    for i, model_name in enumerate(modelos):
        with cols[i % 4]:
            cm = resultados[model_name]['Confusion_Matrix']
            fig, ax = plt.subplots(figsize=(5, 4))

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Sem Doença', 'Com Doença'],
                        yticklabels=['Sem Doença', 'Com Doença'])
            ax.set_title(f'{model_name}')
            ax.set_xlabel('Predição')
            ax.set_ylabel('Real')

            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            st.pyplot(fig)
            st.caption(f"Sensibilidade: {sensitivity:.3f} | Especificidade: {specificity:.3f}")
            plt.close()


def exibir_curvas_roc(resultados, y_test):
    """Exibe curvas ROC para todos os modelos"""
    st.subheader("📈 Curvas ROC")

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.get_cmap('tab20', len(resultados))

    for i, (nome, resultado) in enumerate(resultados.items()):
        if resultado['Probabilities'] is not None:
            fpr, tpr, _ = roc_curve(y_test, resultado['Probabilities'])
            auc = resultado['AUC']
            ax.plot(fpr, tpr, color=colors(i), linewidth=2,
                    label=f'{nome} (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Classificador Aleatório')

    ax.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
    ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)')
    ax.set_title('Curvas ROC - Comparação de Modelos')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def exibir_resultados_comparativos_binarios(resultados):
    """Tabela comparativa para classificação binária"""
    st.subheader("🏆 Comparação Final dos Modelos")

    results_df = pd.DataFrame({
        'Modelo': list(resultados.keys()),
        'Acurácia': [resultados[model]['Accuracy'] for model in resultados.keys()],
        'Precisão': [resultados[model]['Precision'] for model in resultados.keys()],
        'Recall (Sensibilidade)': [resultados[model]['Recall'] for model in resultados.keys()],
        'F1-Score': [resultados[model]['F1-Score'] for model in resultados.keys()],
        'AUC': [resultados[model]['AUC'] for model in resultados.keys()]
    })

    for col in ['Acurácia', 'Precisão', 'Recall (Sensibilidade)', 'F1-Score', 'AUC']:
        results_df[col] = results_df[col].round(4)

    results_df = results_df.sort_values('AUC', ascending=False)

    styled_df = results_df.style.highlight_max(
        subset=['Acurácia', 'Precisão', 'Recall (Sensibilidade)', 'F1-Score', 'AUC'],
        color='lightgreen'
    )

    st.dataframe(styled_df, use_container_width=True)

    best_model = results_df.iloc[0]['Modelo']
    best_auc = results_df.iloc[0]['AUC']
    best_recall = results_df.iloc[0]['Recall (Sensibilidade)']

    st.success(f"🏆 **Melhor Modelo (AUC)**: {best_model} (AUC: {best_auc:.4f})")

    if best_recall < 0.8:
        st.warning(f"⚠️ **Atenção Médica**: O melhor modelo tem recall de {best_recall:.3f} "
                    f"para detectar doença cardíaca. Considere otimizar para maior sensibilidade.")
    else:
        st.info(f"✅ **Boa Sensibilidade**: O modelo detecta {best_recall*100:.1f}% dos casos de doença.")

    return results_df


def gerar_relatorio_cv_texto(resultados_cv):
    """Gera um relatório de texto simples dos resultados da validação cruzada."""
    report_lines = ["Relatório de Validação Cruzada\n", "="*35, "\n\n"]

    cv_df = pd.DataFrame({
        'Modelo': list(resultados_cv.keys()),
        'Acurácia Média': [results['Accuracy_Mean'] for results in resultados_cv.values()],
        'AUC Médio': [results['AUC_Mean'] for results in resultados_cv.values()],
        'F1-Score Médio': [results['F1_Mean'] for results in resultados_cv.values()],
        'Acc Desv. Padrão': [results['Accuracy_Std'] for results in resultados_cv.values()],
        'AUC Desv. Padrão': [results['AUC_Std'] for results in resultados_cv.values()],
        'F1 Desv. Padrão': [results['F1_Std'] for results in resultados_cv.values()]
    }).sort_values('AUC Médio', ascending=False)

    report_lines.append(cv_df.to_string(index=False))
    report_lines.append("\n\n--- Fim do Relatório ---")

    return "\n".join(report_lines)


def gerar_relatorio_completo_md(resultados_teste, resultados_cv, resultados_otimizacao, abordagem, usar_otimizacao):
    """Gera um relatório completo em formato Markdown."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        f"# Relatório Completo de Avaliação de Modelos\n",
        f"Gerado em: {now}\n",
        f"**Abordagem de Balanceamento:** {abordagem}\n",
        f"**Otimização de Hiperparâmetros:** {'Sim' if usar_otimizacao else 'Não'}\n",
        "---\n"
    ]

    # Resultados Finais
    results_df = pd.DataFrame({
        'Modelo': list(resultados_teste.keys()),
        'Acurácia': [r['Accuracy'] for r in resultados_teste.values()],
        'Precisão': [r['Precision'] for r in resultados_teste.values()],
        'Recall (Sensibilidade)': [r['Recall'] for r in resultados_teste.values()],
        'F1-Score': [r['F1-Score'] for r in resultados_teste.values()],
        'AUC': [r['AUC'] for r in resultados_teste.values()]
    }).sort_values('AUC', ascending=False)
    report_lines.append("## 🏆 Resultados Finais no Conjunto de Teste\n")
    report_lines.append(results_df.to_markdown(index=False))
    report_lines.append("\n\n")

    # Resultados da Validação Cruzada
    cv_df = pd.DataFrame({
        'Modelo': list(resultados_cv.keys()),
        'Acurácia Média (CV)': [r['Accuracy_Mean'] for r in resultados_cv.values()],
        'AUC Médio (CV)': [r['AUC_Mean'] for r in resultados_cv.values()],
        'F1-Score Médio (CV)': [r['F1_Mean'] for r in resultados_cv.values()]
    }).sort_values('AUC Médio (CV)', ascending=False)
    report_lines.append("## 🔄 Resultados da Validação Cruzada\n")
    report_lines.append(cv_df.to_markdown(index=False))
    report_lines.append("\n\n")

    # Matrizes de Confusão
    report_lines.append("## 🔍 Matrizes de Confusão (Teste)\n")
    report_lines.append("Formato: [[Verdadeiro Negativo, Falso Positivo], [Falso Negativo, Verdadeiro Positivo]]\n")
    for nome, resultado in resultados_teste.items():
        cm = resultado['Confusion_Matrix']
        report_lines.append(f"### {nome}\n")
        report_lines.append(f"```\n{cm}\n```\n")
    report_lines.append("\n")

    # Hiperparâmetros
    if usar_otimizacao and resultados_otimizacao:
        report_lines.append("## 🔧 Melhores Hiperparâmetros\n")
        for nome, resultado in resultados_otimizacao.items():
            report_lines.append(f"### {nome}\n")
            params_str = str(resultado['best_params'])
            report_lines.append(f"```json\n{params_str}\n```\n")

    return "\n".join(report_lines)


def principal():
    X_train, y_train, X_test, y_test = carregar_e_converter_dados()

    if X_train is None:
        st.stop()

    if 'y_train' not in st.session_state:
        st.session_state.y_train = y_train

    st.sidebar.header("📋 Configurações")
    st.sidebar.subheader("Informações dos Dados")
    st.sidebar.write(f"Treinamento: {X_train.shape[0]} × {X_train.shape[1]}")
    st.sidebar.write(f"Teste: {X_test.shape[0]} × {X_test.shape[1]}")

    class_dist = y_train.value_counts().sort_index()
    st.sidebar.write("**Distribuição (Treinamento):**")
    st.sidebar.write(f"Sem doença (0): {class_dist[0]} ({class_dist[0]/len(y_train)*100:.1f}%)")
    st.sidebar.write(f"Com doença (1): {class_dist[1]} ({class_dist[1]/len(y_train)*100:.1f}%)")

    abordagem = st.sidebar.selectbox(
        "Abordagem de Balanceamento:",
        ["Ponderação de Classe", "Oversampling com SMOTE"]
    )

    usar_otimizacao = st.sidebar.checkbox("Otimizar Hiperparâmetros", value=True)

    if usar_otimizacao:
        cv_folds_opt = st.sidebar.slider("Folds para Otimização", 3, 5, 3)

    cv_folds_final = st.sidebar.slider("Folds para Validação Final", 3, 10, 5)

    if st.button("🚀 Iniciar Avaliação", type="primary"):
        resultados_otimizacao = {}
        with st.spinner("Preparando avaliação..."):

            if abordagem == "Oversampling com SMOTE":
                X_train_processado, y_train_processado = aplicar_smote_binario(X_train, y_train)
                st.success("✅ SMOTE aplicado - classes balanceadas")
            else:
                X_train_processado, y_train_processado = X_train, y_train
                st.success("✅ Ponderação de classe configurada")

            st.header("📊 Distribuição das Classes")
            exibir_distribuicao_classes_binaria(y_train, y_train_processado, abordagem)

            classificadores, param_grids = obter_classificadores_e_parametros_binarios(abordagem, usar_otimizacao)

            if usar_otimizacao and param_grids:
                st.header("🔧 Otimização de Hiperparâmetros")
                with st.spinner("Otimizando..."):
                    modelos_otimizados, resultados_otimizacao = otimizar_com_gridsearch(
                        classificadores, param_grids, X_train_processado, y_train_processado, cv_folds_opt
                    )
                exibir_resultados_otimizacao(resultados_otimizacao)
                modelos_a_usar = modelos_otimizados
            else:
                st.info("Usando configurações padrão")
                modelos_a_usar = {}
                for nome, classificador in classificadores.items():
                    modelos_a_usar[nome] = Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', classificador)
                    ])

            st.header("🔄 Validação Cruzada")
            with st.spinner("Validação cruzada..."):
                resultados_cv = executar_validacao_cruzada_binaria(
                    modelos_a_usar, X_train_processado, y_train_processado, cv_folds_final
                )
            exibir_resultados_cv_binario(resultados_cv)

            st.header("📈 Avaliação Final")
            with st.spinner("Avaliando modelos..."):
                resultados_teste, modelos_treinados = treinar_e_avaliar_binario(
                    modelos_a_usar, X_train_processado, y_train_processado, X_test, y_test
                )

            exibir_matrizes_confusao_binaria(resultados_teste)
            exibir_curvas_roc(resultados_teste, y_test)
            results_df = exibir_resultados_comparativos_binarios(resultados_teste)

            st.header("💡 Recomendações Médicas")

            best_model = results_df.iloc[0]['Modelo']
            best_auc = results_df.iloc[0]['AUC']
            best_recall = results_df.iloc[0]['Recall (Sensibilidade)']
            best_precision = results_df.iloc[0]['Precisão']

            col1, col2 = st.columns(2)

            with col1:
                st.metric("🏆 Melhor Modelo", best_model)
                st.metric("📊 AUC", f"{best_auc:.4f}")
                st.metric("🎯 Sensibilidade", f"{best_recall:.4f}")

            with col2:
                st.metric("🔍 Precisão", f"{best_precision:.4f}")

                if best_recall >= 0.85:
                    st.success("✅ Excelente detecção de doença")
                elif best_recall >= 0.75:
                    st.info("✅ Boa detecção de doença")
                else:
                    st.warning("⚠️ Considere ajustar para maior sensibilidade")

        st.markdown("---")
        st.header("📄 Relatórios para Download")

        col1_dl, col2_dl = st.columns(2)

        with col1_dl:
            # Gerar relatório de validação cruzada
            report_cv_txt = gerar_relatorio_cv_texto(resultados_cv)
            st.download_button(
                label="📥 Baixar Relatório de Validação Cruzada",
                data=report_cv_txt,
                file_name="relatorio_validacao_cruzada.txt",
                mime="text/plain"
            )

        with col2_dl:
            # Gerar relatório completo
            report_completo_md = gerar_relatorio_completo_md(
                resultados_teste,
                resultados_cv,
                resultados_otimizacao,
                abordagem,
                usar_otimizacao
            )
            st.download_button(
                label="📥 Baixar Relatório Completo",
                data=report_completo_md,
                file_name=f"relatorio_completo_{abordagem.replace(' ', '_').lower()}.md",
                mime="text/markdown"
            )


        st.markdown("---")
        st.markdown("### 📝 Sobre a Classificação Binária")
        st.markdown("""
        - **Classe 0**: Ausência de doença cardíaca
        - **Classe 1**: Presença de doença cardíaca
        - **AUC**: Área sob a curva ROC (0.5 = aleatório, 1.0 = perfeito)
        - **Sensibilidade (Recall)**: % de casos de doença corretamente detectados
        - **Especificidade**: % de casos saudáveis corretamente identificados
        - **Aplicação Médica**: Priorizar alta sensibilidade para não perder casos positivos
        """)


if __name__ == "__main__":
    principal()