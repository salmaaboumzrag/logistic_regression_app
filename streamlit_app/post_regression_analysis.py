('''# post_regression_analysis.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

def render(data, model, scaler, imputer, feature_columns):
    if data is not None and model is not None and scaler is not None and imputer is not None and feature_columns is not None:
        st.title("Analyse Post-Régression")

        # Histogramme
        st.subheader("Histogramme")
        hist_column = st.selectbox("Colonne pour l'histogramme", feature_columns)
        if hist_column:
            plt.figure()
            sns.histplot(data[hist_column], kde=True, color='skyblue')
            plt.title(f"Distribution de {hist_column}")
            st.pyplot(plt)

        # Diagramme en boîte
        st.subheader("Diagramme en boîte")
        box_column = st.selectbox("Colonne pour le diagramme en boîte", feature_columns, key='boxplot')
        if box_column:
            plt.figure()
            sns.boxplot(y=data[box_column], color='lightgreen')
            plt.title(f"Boxplot de {box_column}")
            st.pyplot(plt)

        # Scatter plot
        st.subheader("Scatter Plot")
        scatter_x = st.selectbox("Axe X pour le scatter plot", feature_columns, index=0, key='scatter_x')
        scatter_y = st.selectbox("Axe Y pour le scatter plot", feature_columns, index=1, key='scatter_y')
        if scatter_x and scatter_y:
            plt.figure()
            sns.scatterplot(x=data[scatter_x], y=data[scatter_y], hue=data[scatter_y], palette='viridis')
            plt.title(f"{scatter_x} vs {scatter_y}")
            st.pyplot(plt)

        # Graphique de ligne
        st.subheader("Graphique de ligne")
        line_column = st.selectbox("Colonne pour le graphique de ligne", feature_columns, key='lineplot')
        if line_column:
            plt.figure()
            sns.lineplot(data=data[line_column], color='magenta')
            plt.title(f"Graphique de ligne pour {line_column}")
            st.pyplot(plt)

            # Diagramme à barres
    st.subheader("Diagramme à barres")
    bar_column = st.selectbox("Colonne pour le diagramme à barres", feature_columns, key='barplot')
    if bar_column:
        plt.figure()
        sns.barplot(x=data[bar_column].value_counts().index, y=data[bar_column].value_counts(), palette='coolwarm')
        plt.title(f"Diagramme à barres pour {bar_column}")
        plt.xticks(rotation=45)
        st.pyplot(plt)

    # Graphique de densité
    st.subheader("Graphique de densité")
    density_column = st.selectbox("Colonne pour le graphique de densité", feature_columns, key='densityplot')
    if density_column:
        plt.figure()
        sns.kdeplot(data[density_column], shade=True, color='orange')
        plt.title(f"Graphique de densité pour {density_column}")
        st.pyplot(plt)

    # Vous pouvez également inclure ici d'autres analyses comme la matrice de confusion, la courbe ROC, etc.
    # ...

 ''')


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def render(data, feature_columns, target_column):
    if data is not None and feature_columns is not None:
        st.title("Analyse de l'Impact sur la Variable Cible")

        # Séparation des variables quantitatives et qualitatives
        quantitative_variables = data.select_dtypes(include=['number']).columns.tolist()
        qualitative_variables = data.select_dtypes(exclude=['number']).columns.tolist()

        # Sélectionnez le type de variable à analyser (quantitative ou qualitative)
        variable_type = st.radio("Sélectionnez le type de variable à analyser :", ["Quantitative", "Qualitative"])

        if variable_type == "Quantitative":
            selected_feature = st.selectbox("Choisissez une variable quantitative :", quantitative_variables)
        else:
            selected_feature = st.selectbox("Choisissez une variable qualitative :", qualitative_variables)

        # Affichez l'impact de la variable explicative sur la variable cible
        st.subheader(f"Impact de {selected_feature} sur {target_column}")

        if variable_type == "Quantitative":
            # Diagramme en boîte pour la variable explicative en fonction de la variable cible
            plt.figure()
            sns.boxplot(x=target_column, y=selected_feature, data=data, palette='Set2')
            plt.title(f"{selected_feature} en fonction de {target_column}")
            plt.xticks(rotation=45)
            st.pyplot(plt)
        else:
            # Graphique de comptage pour la variable qualitative en fonction de la variable cible
            plt.figure()
            sns.countplot(x=selected_feature, data=data, hue=target_column, palette='Set2')
            plt.title(f"Comptage de {selected_feature} en fonction de {target_column}")
            plt.xticks(rotation=45)
            st.pyplot(plt)

    else:
        st.warning("Veuillez d'abord exécuter la régression logistique pour obtenir les données nécessaires à l'analyse.")
