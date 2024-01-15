import streamlit as st
import load_data_page
import logistic_regression_page
import post_regression_analysis

# Initialisation des variables de session si elles n'existent pas
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
if 'feature_columns' not in st.session_state:
    st.session_state['feature_columns'] = None
if 'target_column' not in st.session_state:
    st.session_state['target_column'] = None

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'  # Page d'accueil par défaut

def main():
    st.sidebar.title("Navigation")

    # Boutons de navigation dans la barre latérale
    if st.sidebar.button("Chargement du Dataset"):
        st.session_state['page'] = 'load_data'
        st.session_state['current_page'] = 'load_data'
    if st.sidebar.button("Régression Logistique"):
        st.session_state['page'] = 'logistic_regression'
        st.session_state['current_page'] = 'logistic_regression'
    if st.sidebar.button("Analyse Post-Régression"):
        st.session_state['page'] = 'post_regression_analysis'
        st.session_state['current_page'] = 'post_regression_analysis'
    
    st.sidebar.markdown("""
    ---
    **Réalisé par :**  
    ABOUMZRAG Salma  
    KESRI Lotfi
    """, unsafe_allow_html=True)

    # Afficher la documentation uniquement sur la page d'accueil
    if st.session_state['current_page'] == 'home':
        st.title("Bienvenue dans l'Application d'Analyse de Régression Logistique")

        st.markdown("""
        ## Documentation de l'Application

        Cette application permet de réaliser une analyse complète à l'aide de la régression logistique. Elle est structurée en trois parties principales :

        1. **Chargement du Dataset** : Téléchargez vos données au format CSV et obtenez un aperçu du dataset. Assurez-vous que vos données sont correctement formatées et nettoyées avant de procéder à l'analyse.

        2. **Régression Logistique** : Une fois vos données chargées, vous pouvez effectuer une régression logistique. Dans cette section, vous sélectionnerez les caractéristiques (variables indépendantes) et la variable cible (dépendante), gérerez les valeurs manquantes, et entraînerez le modèle. Des métriques de performance telles que le rapport de classification et la matrice de confusion seront fournies pour évaluer le modèle.

        3. **Analyse Post-Régression** : Après avoir entraîné le modèle, cette section permet une analyse plus approfondie des résultats. Explorez les métriques de performance supplémentaires et obtenez des insights basés sur les prédictions du modèle.

        Utilisez la barre latérale pour naviguer entre les différentes sections. Chaque section est conçue pour vous guider à travers les étapes spécifiques du processus d'analyse.
        ### Conseils pour une Meilleure Expérience Utilisateur
        - **Format des Données** : Assurez-vous que votre fichier CSV est correctement formaté. Les en-têtes de colonnes doivent être clairement définis.
        - **Nettoyage des Données** : Avant de charger vos données, il est conseillé de les nettoyer pour supprimer les incohérences ou valeurs aberrantes qui pourraient affecter l'analyse.
        - **Sélection Judicieuse des Caractéristiques** : Le choix des variables indépendantes et de la variable dépendante est crucial. Sélectionnez les variables qui sont les plus pertinentes pour votre question de recherche.
        - **Interprétation des Résultats** : Prenez le temps d'analyser et d'interpréter les résultats fournis par le modèle. Cela inclut non seulement les métriques de performance mais aussi les implications pratiques des prédictions.

        N'hésitez pas à expérimenter avec différentes combinaisons de variables et paramètres pour voir comment ils affectent les résultats du modèle.

        ### Besoin d'Aide ?
        Si vous avez des questions ou besoin d'assistance, consultez la [documentation de Streamlit](https://docs.streamlit.io).

        """)

    if st.session_state.get('page') == 'load_data':
        load_data_page.render()
    elif st.session_state.get('page') == 'logistic_regression':
        logistic_regression_page.render(st.session_state['data'])
    elif st.session_state.get('page') == 'post_regression_analysis':
        if st.session_state['data'] is not None:
            post_regression_analysis.render(
                data=st.session_state['data'],
                feature_columns=st.session_state['feature_columns'],
                target_column=st.session_state['target_column']
            )
        else:
            st.warning("Veuillez d'abord exécuter la régression logistique pour obtenir les données nécessaires à l'analyse.")


    



if __name__ == "__main__":
    main()
