
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import warnings

# Configurar warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Configuração da página
st.set_page_config(
    page_title="Predição de Doença Cardíaca",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache para carregar modelos
@st.cache_resource
def carregar_modelos():
    """Carregar o pré-processador e o modelo"""
    try:
        preprocessor = joblib.load("preprocessor.joblib")
        model = joblib.load("svm_model_smote.joblib")
        return preprocessor, model
    except FileNotFoundError as e:
        st.error(f"Erro ao carregar modelos: {e}")
        st.error("Certifique-se de que os arquivos preprocessor.joblib e svm_model_smote.joblib estão no diretório correto.")
        return None, None

def main():
    # Título principal
    st.title("🫀 Sistema de Predição de Doença Cardíaca")
    st.markdown("---")

    # Descrição
    st.markdown("""
    Este sistema utiliza um modelo de Machine Learning (SVM) para prever o risco de doença cardíaca 
    baseado em características clínicas do paciente.
    """)

    # Carregar modelos
    preprocessor, model = carregar_modelos()

    if preprocessor is None or model is None:
        st.stop()

    # Layout em duas colunas
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📋 Dados do Paciente")

        # Criar formulário
        with st.form("patient_form"):
            # Dados numéricos
            st.subheader("Dados Numéricos")
            col_num1, col_num2, col_num3 = st.columns(3)

            with col_num1:
                age = st.number_input("Idade", min_value=1, max_value=120, value=50, step=1)
                trestbps = st.number_input("Pressão Arterial em Repouso (mmHg)", 
                                         min_value=50, max_value=250, value=120, step=1)

            with col_num2:
                chol = st.number_input("Colesterol Sérico (mg/dl)", 
                                     min_value=100, max_value=600, value=200, step=1)
                thalch = st.number_input("Frequência Cardíaca Máxima", 
                                       min_value=60, max_value=220, value=150, step=1)

            with col_num3:
                oldpeak = st.number_input("Depressão do ST induzida por exercício", 
                                        min_value=0.0, max_value=10.0, value=1.0, step=0.1)

            st.markdown("---")

            # Dados categóricos
            st.subheader("Dados Categóricos")
            col_cat1, col_cat2 = st.columns(2)

            with col_cat1:
                sex = st.selectbox("Sexo", ["Male", "Female"])
                dataset = st.selectbox("Dataset", ["Cleveland", "Hungarian", "Switzerland", "VA"])
                cp = st.selectbox("Tipo de Dor no Peito", 
                                ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
                fbs = st.selectbox("Açúcar no Sangue em Jejum > 120 mg/dl", [True, False])

            with col_cat2:
                restecg = st.selectbox("Resultados do ECG em Repouso", 
                                     ["normal", "st-t abnormality", "lv hypertrophy"])
                exang = st.selectbox("Angina Induzida por Exercício", [True, False])
                slope = st.selectbox("Inclinação do Segmento ST", 
                                   ["upsloping", "flat", "downsloping"])

            # Botão de predição
            submitted = st.form_submit_button("🔍 Realizar Predição", use_container_width=True)

        # Processar predição
        if submitted:
            # Preparar dados
            dados_paciente = {
                'age': [age],
                'trestbps': [trestbps],
                'chol': [chol],
                'thalch': [thalch],
                'oldpeak': [oldpeak],
                'sex': [sex],
                'dataset': [dataset],
                'cp': [cp],
                'fbs': [fbs],
                'restecg': [restecg],
                'exang': [exang],
                'slope': [slope]
            }

            # Ordem das colunas
            ordem_colunas = [
                'age', 'trestbps', 'chol', 'thalch', 'oldpeak',
                'sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope'
            ]

            # Criar DataFrame
            paciente_df = pd.DataFrame(dados_paciente, columns=ordem_colunas)

            try:
                # Transformar dados
                paciente_transformado = preprocessor.transform(paciente_df)

                # Fazer predição
                predicao = model.predict(paciente_transformado)
                probabilidades = model.predict_proba(paciente_transformado)
                prob_doenca = probabilidades[0][1]

                # Mostrar resultados na coluna 2
                with col2:
                    st.header("📊 Resultado da Predição")

                    # Métrica de probabilidade
                    st.metric(
                        label="Probabilidade de Risco",
                        value=f"{prob_doenca * 100:.2f}%"
                    )

                    # Resultado categórico
                    if predicao[0] == 1:
                        st.error("⚠️ **ALTO RISCO** de doença cardíaca detectado")
                        st.markdown("**Recomendação:** Procure um cardiologista imediatamente.")
                    else:
                        st.success("✅ **BAIXO RISCO** de doença cardíaca")
                        st.markdown("**Recomendação:** Mantenha hábitos saudáveis e consultas regulares.")

                    # Barra de progresso visual
                    st.markdown("### Nível de Risco")
                    if prob_doenca < 0.3:
                        st.progress(prob_doenca / 0.3 * 0.33)
                        st.markdown("🟢 **Baixo Risco**")
                    elif prob_doenca < 0.7:
                        st.progress(0.33 + (prob_doenca - 0.3) / 0.4 * 0.34)
                        st.markdown("🟡 **Risco Moderado**")
                    else:
                        st.progress(0.67 + (prob_doenca - 0.7) / 0.3 * 0.33)
                        st.markdown("🔴 **Alto Risco**")

                # Mostrar dados do paciente
                st.markdown("---")
                st.subheader("📋 Resumo dos Dados Inseridos")

                # Criar colunas para exibir os dados de forma organizada
                col_dados1, col_dados2 = st.columns(2)

                with col_dados1:
                    st.markdown("**Dados Numéricos:**")
                    st.write(f"• Idade: {age} anos")
                    st.write(f"• Pressão Arterial: {trestbps} mmHg")
                    st.write(f"• Colesterol: {chol} mg/dl")
                    st.write(f"• Freq. Cardíaca Máx: {thalch} bpm")
                    st.write(f"• Depressão ST: {oldpeak}")

                with col_dados2:
                    st.markdown("**Dados Categóricos:**")
                    st.write(f"• Sexo: {sex}")
                    st.write(f"• Tipo de Dor: {cp}")
                    st.write(f"• Açúcar Alto: {'Sim' if fbs else 'Não'}")
                    st.write(f"• Angina por Exercício: {'Sim' if exang else 'Não'}")
                    st.write(f"• Inclinação ST: {slope}")

            except Exception as e:
                st.error(f"Erro na predição: {e}")

    # Sidebar com informações
    with st.sidebar:
        st.header("ℹ️ Informações")

        st.markdown("""
        ### Sobre o Sistema

        Este sistema utiliza um modelo SVM (Support Vector Machine) treinado com técnica SMOTE 
        para balanceamento de dados.

        ### Como usar:
        1. Preencha todos os campos do formulário
        2. Clique em "Realizar Predição"
        3. Analise o resultado e as recomendações

        ### ⚠️ Importante
        Este sistema é apenas uma ferramenta de apoio. 
        **Não substitui a consulta médica profissional.**
        """)

        st.markdown("---")
        st.markdown("### 📚 Glossário")

        with st.expander("Termos Médicos"):
            st.markdown("""
            - **Angina**: Dor no peito por falta de oxigênio no coração
            - **ECG**: Eletrocardiograma 
            - **ST**: Segmento do ECG
            - **Colesterol Sérico**: Nível de colesterol no sangue
            - **mmHg**: Milímetros de mercúrio (pressão)
            - **mg/dl**: Miligramas por decilitro
            """)

if __name__ == "__main__":
    main()
