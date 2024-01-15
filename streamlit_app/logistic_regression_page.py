('''import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def render(data):
    if data is not None:
        st.title("Régression Logistique")

        st.markdown("""
        La régression logistique est utilisée pour prédire l'issue d'une variable dépendante catégorielle en fonction des variables indépendantes. 
        Elle est adaptée pour les problèmes de classification binaire et multinomiale.
        """)
        st.markdown("""
        #### Choix des variables
        - **Variables indépendantes** : Les variables que nous supposons être les prédicteurs de la variable de sortie.
        - **Variable dépendante** : La variable de sortie que nous essayons de prédire. Dans le contexte de la régression logistique, elle doit être catégorielle (souvent binaire).""")
        
        # Sélection des colonnes de caractéristiques et de la colonne cible
        all_columns = data.columns.tolist()
        feature_columns = st.multiselect("Sélectionnez les colonnes de caractéristiques", all_columns, default=all_columns[:-1])
        target_column = st.selectbox("Sélectionnez la colonne cible", all_columns, index=len(all_columns)-1)

        if st.button("Exécuter la régression logistique"):
            # Préparation des données
            X = data[feature_columns]
            y = data[target_column]

            # Gestion des variables catégorielles
            X = pd.get_dummies(X, drop_first=True)

            # Gestion des valeurs manquantes
            imputer = SimpleImputer(strategy='mean')  # Remplacer par 'median' ou une autre stratégie si nécessaire
            X = imputer.fit_transform(X)

            # Séparation en jeux d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # Normalisation des caractéristiques
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Entraînement du modèle
            model = LogisticRegression()
            model.fit(X_train_scaled, y_train)

            # Sauvegarde du modèle et des transformateurs dans session_state
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['imputer'] = imputer
            st.session_state['feature_columns'] = feature_columns
            st.session_state['target_column'] = target_column

            st.success("La régression logistique a été exécutée avec succès.")

            # Prédictions et rapports
            y_pred = model.predict(X_test_scaled)
            report = classification_report(y_test, y_pred, output_dict=True)  # Obtenir le rapport sous forme de dictionnaire
            report_df = pd.DataFrame(report).transpose()  # Convertir le dictionnaire en DataFrame
            report_df = report_df.round(decimals=2)  # Arrondir les valeurs pour une meilleure lisibilité

            st.write("Rapport de Classification :")
            st.dataframe(report_df)

            st.markdown("""
            ### Interprétation du Rapport de Classification
            - **Précision (Precision)**: Proportion des identifications positives qui étaient effectivement correctes.
            - **Rappel (Recall)**: Proportion des résultats positifs réels qui ont été correctement identifiés.
            - **Score F1**: Moyenne harmonique de la précision et du rappel. Plus le score F1 est élevé, mieux c'est.
            - **Support**: Le nombre d'occurrences réelles de la classe dans l'ensemble de données spécifié.

            Lorsque le modèle présente une précision ou un rappel de 0 pour une classe, cela signifie qu'il n'a pas réussi à identifier correctement les instances de cette classe. Dans un tel cas, il est essentiel de revoir le choix des variables, la qualité des données, et potentiellement d'appliquer des méthodes de rééquilibrage des classes ou d'autres techniques de prétraitement.        
            """)

            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred)
            st.write("Matrice de Confusion :")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            ax.set_xlabel('Prédictions')
            ax.set_ylabel('Valeurs Réelles')
            ax.set_title('Matrice de Confusion')
            st.pyplot(fig)
    else:
            st.error("Veuillez charger un dataset avant d'exécuter la régression logistique.")
 ''')
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def render(data):
    if data is not None:
        st.title("Régression Logistique")

        st.markdown("""
        La régression logistique est utilisée pour prédire l'issue d'une variable dépendante catégorielle en fonction des variables indépendantes. Elle est adaptée pour les problèmes de classification binaire et multinomiale.
        """)

        # Initialisation de imputer avec None
        imputer = None

        # Section pour la gestion des valeurs manquantes
        st.markdown("### Gestion des Valeurs Manquantes")
        missing_value_strategy = st.selectbox("Choisissez la stratégie pour les valeurs manquantes", 
                                              ["Supprimer les lignes", "Moyenne", "Médiane", "Constante"])
        if missing_value_strategy == "Supprimer les lignes":
            data.dropna(inplace=True)
        elif missing_value_strategy == "Moyenne":
            imputer = SimpleImputer(strategy='mean')
        elif missing_value_strategy == "Médiane":
            imputer = SimpleImputer(strategy='median')
        else:
            fill_value = st.number_input("Entrez la valeur constante pour remplir", value=0)
            imputer = SimpleImputer(strategy='constant', fill_value=fill_value)

        # Appliquer l'imputation si nécessaire
        if imputer is not None:
            data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

        # Assigner imputer à st.session_state seulement s'il est défini
        if imputer is not None:
            st.session_state['imputer'] = imputer


        # Sélection des colonnes de caractéristiques et de la colonne cible
        st.markdown("""
        #### Sélection des Colonnes pour la Modélisation
        Dans cette section, vous devez choisir les colonnes qui seront utilisées comme caractéristiques (variables indépendantes) et la colonne qui sera utilisée comme cible (variable dépendante) pour la régression logistique.

        - **Colonnes de Caractéristiques**: Ce sont les colonnes qui contiennent les données utilisées pour prédire la variable cible. Sélectionnez une ou plusieurs colonnes qui, selon vous, sont pertinentes pour la prédiction.

        - **Colonne Cible**: C'est la colonne que vous essayez de prédire. Pour la régression logistique, cette colonne doit être catégorielle (par exemple, binaire avec des valeurs comme 0 et 1).

        Veuillez choisir soigneusement les colonnes, car le choix des caractéristiques et de la colonne cible peut avoir un impact significatif sur les performances du modèle.
        """)
        
        all_columns = data.columns.tolist()
        feature_columns = st.multiselect("Sélectionnez les colonnes de caractéristiques", all_columns, default=all_columns[:-1])
        target_column = st.selectbox("Sélectionnez la colonne cible", all_columns, index=len(all_columns)-1)

        if st.button("Exécuter la régression logistique"):
            # Préparation des données
            X = data[feature_columns]
            y = data[target_column]

            # Gestion des variables catégorielles
            X = pd.get_dummies(X, drop_first=True)

            # Gestion des valeurs manquantes
            if missing_value_strategy != "Supprimer les lignes":
                X = pd.DataFrame(imputer.transform(X), columns=X.columns)

            # Séparation en jeux d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # Normalisation des caractéristiques
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Entraînement du modèle
            model = LogisticRegression()
            model.fit(X_train_scaled, y_train)

            # Sauvegarde du modèle et des transformateurs dans session_state
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['imputer'] = imputer
            st.session_state['feature_columns'] = feature_columns
            st.session_state['target_column'] = target_column

            st.success("La régression logistique a été exécutée avec succès.")

            # Prédictions et rapports
            y_pred = model.predict(X_test_scaled)
            report = classification_report(y_test, y_pred, output_dict=True)  # Obtenir le rapport sous forme de dictionnaire
            report_df = pd.DataFrame(report).transpose()  # Convertir le dictionnaire en DataFrame
            report_df = report_df.round(decimals=2)  # Arrondir les valeurs pour une meilleure lisibilité

            st.write("Rapport de Classification :")
            st.dataframe(report_df)

            st.markdown("""
            ### Interprétation du Rapport de Classification
            - **Précision (Precision)**: Proportion des identifications positives qui étaient effectivement correctes.
            - **Rappel (Recall)**: Proportion des résultats positifs réels qui ont été correctement identifiés.
            - **Score F1**: Moyenne harmonique de la précision et du rappel. Plus le score F1 est élevé, mieux c'est.
            - **Support**: Le nombre d'occurrences réelles de la classe dans l'ensemble de données spécifié.

            Lorsque le modèle présente une précision ou un rappel de 0 pour une classe, cela signifie qu'il n'a pas réussi à identifier correctement les instances de cette classe. Dans un tel cas, il est essentiel de revoir le choix des variables, la qualité des données, et potentiellement d'appliquer des méthodes de rééquilibrage des classes ou d'autres techniques de prétraitement.        
            """)

            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred)
            st.write("Matrice de Confusion :")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            ax.set_xlabel('Prédictions')
            ax.set_ylabel('Valeurs Réelles')
            ax.set_title('Matrice de Confusion')
            st.pyplot(fig)
    else:
        st.error("Veuillez charger un dataset avant d'exécuter la régression logistique.")



