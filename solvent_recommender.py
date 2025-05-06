import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Configuration de l'application
st.set_page_config(page_title="Solvent System Recommender", layout="wide")

# Titre de l'application
st.title("🔬 Système de Recommandation de Solvants")
st.markdown("""
Entrez le SMILES d'une molécule pour obtenir des recommandations de systèmes de solvants avec un log KD entre -1 et 1.
""")

# Données des solvants (à remplacer par vos données complètes)
SOLVENT_SYSTEMS = {
    "CPME Ethanol Water": {
        "composition": [0.893, 0.078, 0.029],
        "molar_mass": [100.158, 46.068, 18.015],
        "density": [0.861, 0.789, 0.997]
    },
    "ButylAcetate Ethanol Water": {
        "composition": [0.916, 0.058, 0.026],
        "molar_mass": [116.16, 46.068, 18.015],
        "density": [0.882, 0.789, 0.997]
    },
    "Arizona": {
        "composition": [0.965, 0, 0.035],
        "molar_mass": [100.202, 88.11, 32.04, 18.0153],
        "density": [0.684, 0.902, 0.792, 0.997]
    }
}

# Fonction pour calculer les descripteurs moléculaires
def calculate_molecular_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        return {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol)
        }
    except:
        return None

# Fonction pour préparer les caractéristiques pour la prédiction
def prepare_features(mol_descriptors, solvent_system):
    features = list(mol_descriptors.values())
    features.extend(solvent_system["composition"])
    features.extend(solvent_system["molar_mass"])
    features.extend(solvent_system["density"])
    return features

# Chargement du modèle (version simplifiée - à remplacer par votre vrai modèle)
def load_model():
    # Création d'un modèle factice pour l'exemple
    # En production, vous devriez sauvegarder et charger un vrai modèle entraîné
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    
    # Création de données factices pour l'entraînement
    # REMPLACER par vos vraies données d'entraînement
    X = [[1.2, 3.4, 2, 4, 50.0, 0.8, 0.2, 100, 46, 18, 0.8, 0.7, 1.0]] * 10
    y = [0.5] * 10
    
    model.fit(X, y)
    return model

# Interface utilisateur
def main():
    # Section d'entrée
    col1, col2 = st.columns([3, 1])
    
    with col1:
        smiles_input = st.text_input(
            "Entrez le SMILES de la molécule:",
            placeholder="Ex: C1=CC(=CC=C1/C=C/C2=CC(=CC(=C2)O)O)O",
            help="Entrez la représentation SMILES de votre molécule"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Trouver des solvants")
    
    # Traitement lorsque le bouton est cliqué
    if predict_btn and smiles_input:
        with st.spinner("Recherche des systèmes de solvants appropriés..."):
            # Calcul des descripteurs moléculaires
            mol_descriptors = calculate_molecular_descriptors(smiles_input)
            
            if mol_descriptors is None:
                st.error("SMILES invalide. Veuillez entrer une représentation SMILES valide.")
                return
            
            # Chargement du modèle
            model = load_model()
            
            # Prédiction pour chaque système de solvant
            results = []
            for system_name, system_data in SOLVENT_SYSTEMS.items():
                try:
                    features = prepare_features(mol_descriptors, system_data)
                    log_kd = model.predict([features])[0]
                    
                    if -1 <= log_kd <= 1:
                        results.append({
                            "Système": system_name,
                            "Composition": " / ".join([f"{x:.3f}" for x in system_data["composition"]]),
                            "Log KD prédit": f"{log_kd:.2f}",
                            "Masses molaires": " / ".join([f"{x:.2f}" for x in system_data["molar_mass"]]),
                            "Densités": " / ".join([f"{x:.3f}" for x in system_data["density"]])
                        })
                except Exception as e:
                    st.warning(f"Erreur avec le système {system_name}: {str(e)}")
            
            # Affichage des résultats
            if results:
                st.success(f"{len(results)} systèmes de solvants trouvés avec log KD ∈ [-1, 1]")
                
                # Tri des résultats par log KD le plus proche de 0
                results_sorted = sorted(results, key=lambda x: abs(float(x["Log KD prédit"])))
                
                # Affichage sous forme de tableau
                df_results = pd.DataFrame(results_sorted)
                st.dataframe(
                    df_results,
                    column_config={
                        "Système": "Système de solvant",
                        "Composition": "Composition (vol.)",
                        "Log KD prédit": "Log KD prédit",
                        "Masses molaires": "Masses molaires",
                        "Densités": "Densités (g/mL)"
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Option pour télécharger les résultats
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Télécharger les résultats",
                    data=csv,
                    file_name="solvent_recommendations.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Aucun système de solvant trouvé avec log KD dans l'intervalle [-1, 1].")
    
    # Section d'information
    st.markdown("---")
    st.markdown("""
    ### Comment utiliser cette application
    1. Entrez la représentation SMILES de votre molécule dans le champ ci-dessus
    2. Cliquez sur "Trouver des solvants"
    3. Consultez les systèmes de solvants recommandés avec leur composition
    
    **Critère de sélection:** Seuls les systèmes avec un log KD prédit entre -1 et 1 sont affichés.
    """)

if __name__ == "__main__":
    main()