# Predicción de Aceptación de Préstamos con Machine Learning

Este repositorio contiene un modelo de Machine Learning basado en Random Forest para predecir si un cliente aceptará un préstamo personal.

## 📌 Descripción del Proyecto

El objetivo de este análisis es desarrollar un modelo predictivo que, a partir de características financieras y demográficas de los clientes, determine la probabilidad de aceptación de un préstamo en el cual quise poner en practica mi aprendizaje empirico sobre machine learning y modelos no supervisados.

## Sobre el DataSet
Este es un caso con fines de estudio en el rubro de la banca .la gestión de la compañía quiere explorar formas de llevar a sus clientes con deudas o responsabilidades hacia clientes que tengan créditos personales con el banco ( mientras los retienen como depositantes). Un campaña que banco realizó el último año para clientes mostró un ratio de conversión óptimo de 9% .Esto ha animado al departamento de marketing a idear campañas con mejor análisis de mercado objetivo a fin de incrementar el óptimo ratio usando el mínimo presupuesto

## 🚀 Objetivo
Elegir un modelo óptimo que me indique si el cliente comprará o no el préstamo usando Machine Learning y la data adjunta.

## 🛠 Tecnologías Utilizadas

- Python 🐍
- Pandas
- NumPy
- Scikit-Learn
- Seaborn
- Matplotlib

## 📊 Análisis Realizado

1. **Carga de Datos:** Importación y previsualización del dataset:
- Los Datos que utilizamos fueron tomados de https://www.kaggle.com/datasets/luisenriquesguerrero/creditos-personales-actualizado/data

2. **Exploración de Variables:** Descripción y análisis de distribuciones:

- Analizamos las variables aplicando de manera logica la inferencia del proposito de las mismas y seleccionando a criterio experto las mas relevantes para el entrenamiento el modelo, excluyendo variables que mostraban multicolinealiad en la matriz de correlacion
3. **Limpieza y Transformación:** Normalización de ingresos y valores hipotecarios:

-  Se normalizan algunas variables ya se que se desconoce en que moneda se contruye esta base de datos practica, asi que se busca simular la escala de millones de pesos para hacer mas comprensible el meaning de las mismas,
-  La unica variable categorica que se tiene en consideracion para el modelo es Educacion, esta se convierte a dummie para 
4. **Entrenamiento del Modelo:** Uso de Random Forest para la predicción:

- El Modelo se entrena con 80/20 y se toman por default 100 arboles de decision para su creacion 
5. **Evaluación del Modelo:** Cálculo de métricas como precisión, recall y matriz de confusión:

- Se tienen en cuenta las siguientes consideraciones: 
    Verdaderos negativos (TN): 893 (clientes que no aceptaron el préstamo y fueron clasificados correctamente).
    Falsos positivos (FP): 2 (clientes que no aceptaron el préstamo pero fueron clasificados como que sí lo aceptaron).
    Falsos negativos (FN): 7 (clientes que sí aceptaron el préstamo pero fueron clasificados como que no lo aceptaron).
    Verdaderos positivos (TP): 98 (clientes que sí aceptaron el préstamo y fueron clasificados correctamente).
Sobreconfianza en la exactitud:
    Un accuracy del 99.1% puede llevar a pensar que el modelo es perfecto, pero no refleja el desempeño en la clase minoritaria.

6. **Interpretacion del Modelo**
Estructura del árbol
    Nodo raíz:

        Condición: Income <= 11.35.

        Gini (gm): 0.17 (medida de impureza; cuanto más cercano a 0, más puro es el nodo).

        Muestras (samples): 4000 (número de muestras en este nodo).

        Valor: [3625, 375] (3625 clientes no aceptaron el préstamo, 375 sí lo aceptaron).

        Clase predicha: "No Acepta" (la clase mayoritaria).

    Este nodo divide los datos en dos ramas: una para clientes con ingresos menores o iguales a 11.35 y otra para ingresos mayores.
    Nodos intermedios:
        Cada nodo intermedio aplica una condición adicional para refinar la predicción.
        Por ejemplo:

            CCAvg <= 2.95: Divide los datos según el gasto promedio en tarjetas de crédito.

            Family <= 2.5: Divide los datos según el número de miembros de la familia.

            CD Account <= 0.5: Divide los datos según si el cliente tiene una cuenta de certificado de depósito (CD).

    Nodos hoja:
        Estos son los nodos finales que predicen la clase ("No Acepta" o "Acepta").
        Por ejemplo:

            Un nodo con value = [4, 169] y class = Acepta indica que 169 clientes aceptaron el préstamo y 4 no lo aceptaron.

            Un nodo con value = [2962, 72] y class = No Acepta indica que 2962 clientes no aceptaron el préstamo y 72 sí lo aceptaron.

Interpretación de las reglas de decisión
    Ingreso (Income):
        Es la primera característica que divide los datos, lo que indica que es la más importante para predecir si un cliente aceptará el préstamo.

        Los clientes con ingresos bajos (Income <= 11.35) tienen menos probabilidades de aceptar el préstamo.

    Gasto en tarjetas de crédito (CCAvg):
        Los clientes con un gasto bajo en tarjetas de crédito (CCAvg <= 2.95) tienen menos probabilidades de aceptar el préstamo.

    Familia (Family):
        Los clientes con familias pequeñas (Family <= 2.5) tienen menos probabilidades de aceptar el préstamo.

    Cuenta de certificado de depósito (CD Account):
        Los clientes sin una cuenta de certificado de depósito (CD Account <= 0.5) tienen menos probabilidades de aceptar el préstamo.

    Educación (Education):
        La educación también juega un papel importante. Los clientes con un nivel de educación más alto tienen más probabilidades de aceptar el préstamo.

Perfil del cliente que acepta el préstamo

Basado en las reglas del árbol, el perfil ideal de un cliente que aceptará el préstamo es:

    Ingreso alto: Income > 11.35.

    Gasto alto en tarjetas de crédito: CCAvg > 2.95.

    Familia grande: Family > 2.5.

    Tiene una cuenta de certificado de depósito: CD Account > 0.5.

    Nivel de educación alto: Education > 0.5.

Ejemplo de interpretación de un nodo

    Nodo con value = [4, 169] y class = Acepta:

        Este nodo representa a clientes con:

            Ingresos altos.

            Gasto alto en tarjetas de crédito.

            Familia grande.

            Cuenta de certificado de depósito.

        De estos clientes, 169 aceptaron el préstamo y 4 no lo aceptaron.

        La clase predicha es "Acepta" porque es la mayoría.

7. **Testeo**

Para el testeo del modelo se le pide a chat gpt generar datos aleatorios con la misma estructura de datos que estamos trabajando para simular
prospectos de clientes nuevos a los cuales les intentaremos predecir si van a tomar o no el credito personal que les queremos ofrecer

##  Cómo Usar el Notebook

1. En Use_model.ipynb tiene la estructura para correr el modelo entrenado
2. random_forest_model.pkl contiene el modelo

## 📈 Resultados Clave

El árbol de decisión muestra que las características más importantes para predecir si un cliente aceptará un préstamo son:
Ingreso, Gasto en tarjetas de crédito, Tamaño de la familia, Cuenta de certificado de depósito, Nivel de educación.

El perfil ideal de un cliente que aceptará el préstamo es aquel con ingresos altos, gasto alto en tarjetas de crédito, familia grande, cuenta de certificado de depósito y nivel de educación alto. Este análisis te permite enfocar tus esfuerzos de marketing en clientes con estas características. 🚀

## 📬 Contacto
Jose Ricardo Gonzalez
Analista de Riesgo de Credito, Economista y Desarrollador de Software:
Si tienes preguntas o sugerencias, ¡no dudes en conectarte conmigo en [LinkedIn](https://www.linkedin.com/in/josericardogob/)!

