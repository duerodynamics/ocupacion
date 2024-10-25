import streamlit as st
import xgboost
import pandas as pd
import numpy as np
import joblib
import requests
import matplotlib.pyplot as plt

# Cargar el modelo preentrenado
daily_model = joblib.load("xgboost_model.pkl")

# Gestionar la navegación entre secciones
if "page" not in st.session_state:
    st.session_state.page = "info"

# Definir una función para cambiar la página
def change_page(page_name):
    st.session_state.page = page_name

# Sidebar para navegación
st.sidebar.title("Navegación")
st.sidebar.button("Información", on_click=change_page, args=("info",))
st.sidebar.button("Manual", on_click=change_page, args=("manual",))
st.sidebar.button("Automático", on_click=change_page, args=("automatic",))

# Mapa de meses y días de la semana para facilitar la selección
months_map = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
    "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
    "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
}
days_of_week_map = {
    "Lunes": 0, "Martes": 1, "Miércoles": 2, "Jueves": 3,
    "Viernes": 4, "Sábado": 5, "Domingo": 6
}

# Página de información (Principal)
if st.session_state.page == "info":
    st.title("Información sobre la Predicción de Camas Ocupadas")
    
    st.markdown("""

        En este proyecto, se ha desarrollado un modelo de inteligencia artificial (IA) para predecir la ocupación diaria de camas hospitalarias en el **Complejo Asistencial de Zamora**. El modelo ha sido entrenado utilizando los datos abiertos proporcionados por la **Junta de Castilla y León**, disponibles a través de su plataforma de datos abiertos.

        Los datos históricos sobre la ocupación de camas se obtienen de la siguiente fuente oficial: 
        [Ocupación de camas en hospitales - Junta de Castilla y León](https://datosabiertos.jcyl.es/web/jcyl/set/es/salud/ocupacion-hospitales/1284947951914). 

        ### Métricas del Modelo
        A continuación, se muestra un gráfico comparativo entre las camas ocupadas reales, las predicciones del modelo, y el número de camas habilitadas durante el año 2024:
    """)

    # Mostrar la imagen del gráfico que el usuario adjuntó
    st.image("Gráfico.png", caption="Comparación de Camas Habilitadas vs. Camas Ocupadas vs. Predictor - 2024", use_column_width=True)

    st.markdown("""
        ### Evaluación del Modelo
        A continuación, se explican dos métricas clave para entender el desempeño del modelo de predicción:

        - **R² (R-squared) para el Predictor en 2024: 0.81**  
        El R-squared, o coeficiente de determinación, nos indica qué tan bien el modelo de predicción se ajusta a los datos reales. En este caso, un valor de **0.81** significa que el **81% de la variabilidad** en la ocupación de camas puede ser explicada por el modelo de predicción.  
        En términos simples, esto quiere decir que el modelo es bastante bueno para seguir las tendencias y patrones de la ocupación de camas, aunque aún hay un **19% de variabilidad** que el modelo no logra capturar, lo cual puede deberse a factores imprevistos o cambios que no se incluyeron en los datos utilizados para entrenar el modelo.

        - **Root Mean Squared Error (RMSE) para el Predictor en 2024: 10.60**  
        El RMSE mide la magnitud promedio del error en las predicciones del modelo, pero de una forma que penaliza más los errores grandes. En este caso, el **RMSE es de 10.60**, lo que significa que, en promedio, la diferencia entre el número real de camas ocupadas y el número que predice el modelo es de aproximadamente **10.6 camas**.  
        Un RMSE más bajo significa que el modelo está más cerca de los valores reales. Sin embargo, incluso con un RMSE de 10.60, el modelo puede ser útil para dar una idea general de la ocupación de camas y para planificar la gestión hospitalaria.

        """)

    st.markdown("""
    ### Explicación de los modos de uso
                
    La herramienta de predicción de la ocupación de camas del **Complejo Asistencial de Zamora** ofrece dos opciones: una predicción **automática** y una **manual**. La intención original del proyecto era que la predicción automática se actualizara diariamente, utilizando los datos abiertos proporcionados por la **Junta de Castilla y León**, ya que en su [portal de datos abiertos](https://datosabiertos.jcyl.es/web/jcyl/set/es/salud/ocupacion-hospitales/1284947951914) se indica que la frecuencia de actualización es **diaria de lunes a viernes**. Sin embargo, al consultar la [sección de análisis](https://analisis.datosabiertos.jcyl.es/explore/dataset/ocupacion-de-camas-en-hospitales/information/?sort=fecha) de estos datos, se observa que la **actualización real es solo dos veces por semana: los martes y los viernes**. 

    A fecha **25 de octubre de 2024**, los datos más recientes disponibles llegan únicamente hasta el **4 de octubre de 2024**, lo que limita la capacidad del modelo para reflejar la ocupación más actualizada. Para complementar esta situación y permitir obtener predicciones más precisas cuando los datos más recientes no están disponibles, se ha desarrollado una **versión manual**. Esta permite introducir los datos de ocupación de los últimos 15 días de forma manual para generar la predicción del día siguiente. 

    Cabe destacar que el modelo fue entrenado con datos históricos hasta el **20 de septiembre de 2024**, por lo que la calidad de las predicciones depende de la actualización de los datos de entrada.
    """)



    # Botones para ir a las secciones de Manual y Automático
    st.button("Ir a Predicción Manual", on_click=change_page, args=("manual",))
    st.button("Ir a Predicción Automática", on_click=change_page, args=("automatic",))

# Sección de predicción manual
elif st.session_state.page == "manual":
    st.title("Predicción Diaria - Introducir Datos Manualmente")
    
    # Formularios para ingresar los datos de los últimos 15 días
    manual_input = []
    for i in range(15):
        manual_input.append(st.number_input(f"Día {i+1} - Camas Ocupadas", min_value=0, max_value=500))
    
    # Texto aclarativo antes de los selectores
    st.markdown("""
        <p style='color: grey; font-size: 0.9em;'>
            Selecciona el mes y el día de la semana correspondientes al 
            <strong>día previo a la fecha de la predicción deseada</strong>.
        </p>
    """, unsafe_allow_html=True)

    
    # Selector de mes y día de la semana como desplegables
    month = st.selectbox("Mes", list(months_map.keys()))
    day_of_week = st.selectbox("Día de la Semana", list(days_of_week_map.keys()))

    if st.button("Predecir"):
        # Convertir los valores seleccionados a numéricos
        month_num = months_map[month]
        day_of_week_num = days_of_week_map[day_of_week]
        
        # Preparar las características para la predicción (15 lags + month + day_of_week)
        features = np.array(manual_input + [month_num, day_of_week_num]).reshape(1, -1)
        prediction = daily_model.predict(features)
        rounded_prediction = np.ceil(prediction[0])
        st.success(f"Predicción para el próximo día: {int(rounded_prediction)} camas ocupadas")


# Sección de predicción automática
elif st.session_state.page == "automatic":
    st.title("Predicción Automática")
    
    # URL de la API
    api_url = "https://analisis.datosabiertos.jcyl.es/api/explore/v2.1/catalog/datasets/ocupacion-de-camas-en-hospitales/records?select=fecha,camas_ocupadas_planta&where=hospital=%27Complejo%20Asistencial%20de%20Zamora%27&order_by=fecha%20desc&limit=15"
    
    # Intentar realizar la petición GET a la API
    try:
        response = requests.get(api_url)

        # Verificar que la petición fue exitosa
        if response.status_code == 200:
            data = response.json()
            records = data['results']
            extracted_data = [record['camas_ocupadas_planta'] for record in records]
            extracted_data.reverse()

            # Asegurarse de que hay 15 registros
            if len(extracted_data) == 15:
                last_date = pd.to_datetime(records[0]['fecha'])
                month_num = last_date.month
                day_of_week_num = last_date.dayofweek
                
                # Preparar las características para la predicción (15 lags + month + day_of_week)
                features = np.array(extracted_data + [month_num, day_of_week_num]).reshape(1, -1)
                prediction = daily_model.predict(features)
                rounded_prediction = np.ceil(prediction[0])
                
                # Mostrar un mensaje aclaratorio sobre la fecha de la predicción
                st.markdown(f"""
                    <p style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>
                        <strong>Predicción realizada para el día siguiente a la última fecha disponible: {last_date.strftime('%d/%m/%Y')}</strong><br>
                        La predicción se basa en los últimos 15 días de ocupación registrados. Los datos se actualizan de acuerdo a la última información disponible en el 
                        <a href='https://datosabiertos.jcyl.es/web/jcyl/set/es/salud/ocupacion-hospitales/1284947951914' target='_blank'>portal de datos abiertos de la Junta de Castilla y León</a>.
                    </p>
                """, unsafe_allow_html=True)
                
                # Mostrar el resultado de la predicción
                st.success(f"Predicción para el próximo día: {int(rounded_prediction)} camas ocupadas")

                # Graficar los datos históricos y la predicción
                st.subheader("Gráfico de Datos Históricos")
                plt.plot(range(1, 16), extracted_data, marker='o', linestyle='-', label='Datos Históricos')
                plt.scatter(16, rounded_prediction, color='red', label='Predicción', zorder=5)
                plt.xlabel("Día")
                plt.ylabel("Camas Ocupadas")
                plt.title("Datos Históricos de Camas Ocupadas")
                plt.legend()
                st.pyplot(plt)

            else:
                st.error("No se encontraron suficientes registros para la predicción.")
        else:
            st.error(f"Error al obtener datos de la API. Código de estado: {response.status_code}")

    
    except requests.exceptions.RequestException as e:
        st.error(f"Error al realizar la solicitud a la API: {e}")
