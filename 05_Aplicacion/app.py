import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime

st.set_page_config(page_title="CO2 Planta de Acero", layout="wide")

PROMEDIOS_HISTORICOS = {
    "Light_Load": 0.004,
    "Medium_Load": 0.008,
    "Maximum_Load": 0.016,
}

LOAD_TYPE_LABELS = {
    "Carga Ligera": "Light_Load",
    "Carga Media": "Medium_Load",
    "Carga Maxima": "Maximum_Load",
}


@st.cache_resource
def cargar_pipeline():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta = os.path.join(base, "03_Modelos", "02_Produccion", "pipeline_produccion_B_min_v1.joblib")
    return joblib.load(ruta)


def calcular_nsm():
    now = datetime.now()
    nsm = now.hour * 3600 + now.minute * 60 + now.second
    return np.sin(2 * np.pi * nsm / 86400), np.cos(2 * np.pi * nsm / 86400)


def predecir(pipeline, load_type, leading_pf, lagging_pf, nsm_sin, nsm_cos):
    enc = pipeline["encoder"]
    scaler = pipeline["scaler"]
    target_scaler = pipeline["target_scaler"]
    modelo = pipeline["modelo"]

    cat_input = pd.DataFrame({"Load_Type": [load_type]})
    cat_ohe = enc.transform(cat_input)

    num_input = pd.DataFrame({
        "Leading_Current_Power_Factor": [leading_pf],
        "Lagging_Current_Power_Factor": [lagging_pf],
        "NSM_sin": [nsm_sin],
        "NSM_cos": [nsm_cos],
    })
    num_scaled = scaler.transform(num_input)

    X = np.hstack([cat_ohe, num_scaled])
    y_std = modelo.predict(X)
    y_tco2 = target_scaler.inverse_transform(y_std.reshape(-1, 1)).ravel()
    return float(y_tco2[0])


def predecir_batch(pipeline, df):
    enc = pipeline["encoder"]
    scaler = pipeline["scaler"]
    target_scaler = pipeline["target_scaler"]
    modelo = pipeline["modelo"]

    cat_ohe = enc.transform(df[["Load_Type"]])

    num_scaled = scaler.transform(
        df[["Leading_Current_Power_Factor", "Lagging_Current_Power_Factor", "NSM_sin", "NSM_cos"]]
    )

    X = np.hstack([cat_ohe, num_scaled])
    y_std = modelo.predict(X)
    y_tco2 = target_scaler.inverse_transform(y_std.reshape(-1, 1)).ravel()
    return y_tco2


def navegar_a(pagina):
    st.session_state["pagina"] = pagina


if "pagina" not in st.session_state:
    st.session_state["pagina"] = "Portada"

with st.sidebar:
    st.title("Navegacion")
    st.button("Portada", on_click=navegar_a, args=["Portada"], width='stretch')
    st.button("Prediccion Individual", on_click=navegar_a, args=["Individual"], width='stretch')
    st.button("Prediccion por Dataset", on_click=navegar_a, args=["Dataset"], width='stretch')
    st.divider()
    st.caption("Modelo: pipeline_produccion_B_min_v1")
    st.caption("Planta de Acero — 2018")

pagina = st.session_state["pagina"]

# ---------- PORTADA ----------
if pagina == "Portada":
    st.title("Prediccion de Emisiones CO2 — Planta de Acero")
    st.markdown(
        """
        Esta aplicacion permite anticipar las emisiones de CO2 de una planta de acero coreana
        **antes de que ocurra el consumo energetico**, brindando al operador informacion
        para tomar decisiones operacionales que reduzcan la huella de carbono.
        """
    )

    st.subheader("Objetivo de negocio")
    st.markdown(
        """
        Optimizar la planificacion operacional del turno seleccionando el tipo de carga
        y ajustando los factores de potencia para minimizar las emisiones de CO2 proyectadas,
        antes de comprometer recursos en planta.
        """
    )

    st.subheader("Desempeno del modelo")
    col1, col2, col3 = st.columns(3)
    col1.metric("R²", "0.8715", help="Proporcion de varianza explicada en el conjunto de validacion")
    col2.metric("RMSE", "0.0058 tCO2", help="Error cuadratico medio en validacion")
    col3.metric("MAE", "0.0029 tCO2", help="Error absoluto medio en validacion")

    st.subheader("Variables de entrada")
    st.markdown(
        """
        El modelo requiere unicamente **3 inputs del operador** mas la hora del sistema:

        | Variable | Descripcion | Fuente |
        |---|---|---|
        | Tipo de Carga | Regimen operacional del turno | Operador |
        | Factor de Potencia Adelantado | Eficiencia del suministro de energia | Operador |
        | Factor de Potencia Retrasado | Eficiencia del consumo de energia | Operador |
        | NSM (sin/cos) | Posicion ciclica en el dia de 86 400 s | Hora del sistema |

        La hora del sistema se captura automaticamente — el operador no necesita ingresarla.
        """
    )

    st.divider()
    col_a, col_b = st.columns(2)
    col_a.button("Ir a Prediccion Individual", on_click=navegar_a, args=["Individual"], width='stretch')
    col_b.button("Ir a Prediccion por Dataset", on_click=navegar_a, args=["Dataset"], width='stretch')

# ---------- PREDICCION INDIVIDUAL ----------
elif pagina == "Individual":
    st.button("← Volver a la Portada", on_click=navegar_a, args=("Portada",))
    st.title("Prediccion Individual de CO2")
    st.markdown("Ingresa los parametros operacionales para obtener la prediccion de emisiones del proximo intervalo.")

    pipeline = cargar_pipeline()

    with st.form("form_prediccion"):
        st.subheader("Parametros operacionales")

        label_seleccionado = st.selectbox(
            "Tipo de carga",
            options=list(LOAD_TYPE_LABELS.keys()),
            help="Regimen de operacion del turno",
        )

        col1, col2 = st.columns(2)
        with col1:
            leading_pf = st.number_input(
                "Factor de Potencia Adelantado (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=0.1,
                help="Eficiencia del suministro de energia. Rango tipico: 60–100",
            )
        with col2:
            lagging_pf = st.number_input(
                "Factor de Potencia Retrasado (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=0.1,
                help="Eficiencia del consumo de energia. Rango tipico: 60–100",
            )

        now_display = datetime.now()
        st.info(
            f"Hora del sistema: **{now_display.strftime('%H:%M:%S')}** — "
            "El parametro NSM se calcula automaticamente a partir de esta hora."
        )

        submitted = st.form_submit_button("Predecir", width='stretch')

    if submitted:
        load_type = LOAD_TYPE_LABELS[label_seleccionado]
        nsm_sin, nsm_cos = calcular_nsm()
        prediccion = predecir(pipeline, load_type, leading_pf, lagging_pf, nsm_sin, nsm_cos)
        promedio = PROMEDIOS_HISTORICOS[load_type]

        st.divider()
        st.subheader("Resultado")

        col_r1, col_r2 = st.columns([1, 2])

        with col_r1:
            delta_val = prediccion - promedio
            delta_str = f"{delta_val:+.4f} tCO2 vs. promedio"
            st.metric(
                label="CO2 predicho",
                value=f"{prediccion:.4f} tCO2",
                delta=delta_str,
                delta_color="inverse",
            )

        with col_r2:
            color_barra = "#2ecc71" if prediccion < promedio else "#e67e22"
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=[prediccion, promedio],
                y=["Prediccion actual", f"Promedio historico ({label_seleccionado})"],
                orientation="h",
                marker_color=[color_barra, "#95a5a6"],
                text=[f"{prediccion:.4f} tCO2", f"{promedio:.4f} tCO2"],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="Comparativa vs. promedio historico",
                xaxis_title="tCO2",
                margin=dict(l=10, r=80, t=40, b=30),
                height=200,
                showlegend=False,
            )
            st.plotly_chart(fig_bar, width='stretch')

        st.subheader("Indicadores de factor de potencia")
        col_g1, col_g2 = st.columns(2)

        def gauge_fp(valor, titulo):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=valor,
                title={"text": titulo},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2c3e50"},
                    "steps": [
                        {"range": [0, 60], "color": "#e74c3c"},
                        {"range": [60, 80], "color": "#e67e22"},
                        {"range": [80, 100], "color": "#2ecc71"},
                    ],
                },
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=20))
            return fig

        with col_g1:
            st.plotly_chart(gauge_fp(leading_pf, "Factor de Potencia Adelantado"), width='stretch')
        with col_g2:
            st.plotly_chart(gauge_fp(lagging_pf, "Factor de Potencia Retrasado"), width='stretch')

# ---------- PREDICCION POR DATASET ----------
elif pagina == "Dataset":
    st.button("← Volver a la Portada", on_click=navegar_a, args=("Portada",))
    st.title("Prediccion por Dataset")
    st.markdown(
        """
        Carga un archivo CSV con los parametros operacionales de multiples intervalos
        para generar predicciones en lote.

        **Columnas requeridas en el CSV:**
        `Load_Type`, `Leading_Current_Power_Factor`, `Lagging_Current_Power_Factor`

        Los valores de `Load_Type` deben ser exactamente: `Light_Load`, `Medium_Load`, `Maximum_Load`.

        El parametro NSM se calculara automaticamente con la hora del sistema al momento de la carga.
        """
    )

    pipeline = cargar_pipeline()

    archivo = st.file_uploader("Selecciona el archivo CSV", type=["csv"])

    if archivo is not None:
        df_raw = pd.read_csv(archivo)

        columnas_requeridas = {"Load_Type", "Leading_Current_Power_Factor", "Lagging_Current_Power_Factor"}
        columnas_faltantes = columnas_requeridas - set(df_raw.columns)

        if columnas_faltantes:
            st.error(f"El archivo no contiene las siguientes columnas requeridas: {', '.join(columnas_faltantes)}")
        else:
            st.subheader("Vista previa del archivo cargado")
            st.dataframe(df_raw.head(5), width='stretch')

            if st.button("Generar Predicciones", width='stretch'):
                nsm_sin, nsm_cos = calcular_nsm()
                df_modelo = df_raw[["Load_Type", "Leading_Current_Power_Factor", "Lagging_Current_Power_Factor"]].copy()
                df_modelo["NSM_sin"] = nsm_sin
                df_modelo["NSM_cos"] = nsm_cos

                predicciones = predecir_batch(pipeline, df_modelo)

                df_resultado = df_raw.copy()
                df_resultado["CO2_predicho_tCO2"] = predicciones.round(6)
                df_resultado["CO2_predicho_tCO2"] = df_resultado["CO2_predicho_tCO2"].clip(lower=0).round(6)

                st.subheader("Predicciones generadas")
                st.dataframe(df_resultado, width='stretch')

                st.subheader("Distribucion de CO2 predicho por tipo de carga")
                load_order = ["Light_Load", "Medium_Load", "Maximum_Load"]
                categorias_presentes = [c for c in load_order if c in df_resultado["Load_Type"].unique()]

                fig_hist = px.histogram(
                    df_resultado,
                    x="CO2_predicho_tCO2",
                    color="Load_Type",
                    barmode="overlay",
                    opacity=0.75,
                    category_orders={"Load_Type": categorias_presentes},
                    labels={
                        "CO2_predicho_tCO2": "CO2 predicho (tCO2)",
                        "Load_Type": "Tipo de carga",
                    },
                    title="Distribucion de emisiones predichas",
                )
                fig_hist.update_layout(margin=dict(l=20, r=20, t=50, b=40))
                st.plotly_chart(fig_hist, width='stretch')

                st.download_button(
                    label="Descargar reporte CSV",
                    data=df_resultado.to_csv(index=False),
                    file_name="predicciones_co2.csv",
                    mime="text/csv",
                )