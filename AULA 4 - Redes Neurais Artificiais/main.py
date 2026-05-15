import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# --- CONFIGURAÇÃO DA INTERFACE ---
st.set_page_config(page_title="Análise de Regressão Linear", layout="centered")



st.title("📘 Predição de Desempenho Acadêmico")
st.markdown("""
Este script demonstra a implementação de uma **Regressão Linear Simples** utilizando `scikit-learn`.
O modelo analisa a correlação entre horas de estudo e a nota final obtida.
""")

# --- 1. CONJUNTO DE DADOS ---
# Representação dos dados fornecidos para o treinamento
estudos = pd.DataFrame({
    'notas': [1, 2, 4, 6, 8, 10],
    'horas': [2, 4, 5, 7, 9, 10]
})

# --- 2. MODELAGEM (SCIKIT-LEARN) ---
# O Scikit-learn exige que a variável independente (X) seja uma matriz (2D)
X = estudos[['horas']]  # Variável Independente (Feature)
y = estudos['notas']    # Variável Dependente (Target)

# Inicialização e ajuste (treinamento) do modelo
modelo = LinearRegression()
modelo.fit(X, y)

# --- 3. INTERATIVIDADE COM STREAMLIT ---
st.sidebar.header("Parâmetros de Entrada")
horas_estudo = st.sidebar.slider("Horas de Estudo Diário:", 0.0, 12.0, 5.0)

# Realizando a predição baseada no input do usuário
# Convertemos o input em um array 2D para o método .predict()
nota_prevista = modelo.predict([[horas_estudo]])[0]
nota_prevista = np.clip(nota_prevista, 0, 10) # Garante que a nota fique entre 0 e 10

# --- 4. EXIBIÇÃO DE RESULTADOS ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Dados Base")
    st.dataframe(estudos)

with col2:
    st.subheader("Resultado da Predição")
    st.metric(label="Nota Estimada", value=f"{nota_prevista:.2f}")
    st.info(f"Para {horas_estudo} horas, a nota esperada é aproximadamente {nota_prevista:.1f}.")

# --- 5. REPRESENTAÇÃO GRÁFICA ---
st.subheader("Visualização do Modelo")
fig, ax = plt.subplots()

# Scatter plot: Pontos reais do DataFrame
ax.scatter(estudos['horas'], estudos['notas'], color='blue', label='Dados Históricos')

# Plot da linha de regressão (tendência teórica)
x_model = np.linspace(0, 12, 100).reshape(-1, 1)
y_model = modelo.predict(x_model)
ax.plot(x_model, y_model, color='red', linestyle='--', label='Reta de Regressão')

# Destaque do ponto predito pelo usuário
ax.scatter([horas_estudo], [nota_prevista], color='green', s=100, label='Sua Predição', zorder=5)

ax.set_xlabel("Horas de Estudo")
ax.set_ylabel("Nota Final")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# --- 6. EXPLICAÇÃO ACADÊMICA ---
with st.expander("Ver Detalhes Matemáticos"):
    st.write(f"**Coeficiente Angular (β1):** {modelo.coef_[0]:.4f}")
    st.write(f"**Intercepto (β0):** {modelo.intercept_:.4f}")
    st.latex(rf"f(x) = {modelo.coef_[0]:.2f}x + ({modelo.intercept_:.2f})")
    st.write("A regressão linear busca a linha que minimiza a soma dos quadrados dos resíduos.")