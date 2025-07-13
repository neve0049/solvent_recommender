import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Draw
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
st.set_page_config(page_title="🔬 Solvent System Recommender Pro", layout="wide")
st.title("🔬 Solvent System Recommender Pro")

# Constants
MODEL_PATH = "solvent_model.pkl"
KDDB_PATH = "KDDB.xlsx"
DBDQ_PATH = "DBDQ.xlsx"
DBDT_PATH = "DBDT.xlsx"

def load_solvents_from_files():
    """Charge la liste des solvants depuis les fichiers Excel"""
    solvents = set()
    
    try:
        # Charger tous les fichiers
        kddb = pd.read_excel(KDDB_PATH, sheet_name=None)
        dbdq = pd.read_excel(DBDQ_PATH, sheet_name=None)
        dbdt = pd.read_excel(DBDT_PATH, sheet_name=None)
        
        # Extraire les solvants uniques de tous les fichiers
        for data in [kddb, dbdq, dbdt]:
            for sheet_name, sheet_data in data.items():
                if 'System' in sheet_data.columns:
                    solvents.update(sheet_data['System'].unique())
        
        return sorted(list(solvents))
    
    except Exception as e:
        st.error(f"Error loading solvent data: {str(e)}")
        return []

def load_model():
    """Charge le modèle avec vérification des features"""
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            if hasattr(model, 'feature_names_in_'):
                st.session_state['expected_features'] = list(model.feature_names_in_)
            return model
        st.error("Model file not found!")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def calculate_molecular_features(smiles):
    """Calcule les features moléculaires"""
    if not smiles:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        st.error("Invalid SMILES string")
        return None
    
    return {
        'MolWeight': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'RotatableBonds': Lipinski.NumRotatableBonds(mol),
        'AromaticRings': Lipinski.NumAromaticRings(mol),
        'HeavyAtoms': Lipinski.HeavyAtomCount(mol),
        'RingCount': Lipinski.RingCount(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol)
    }

def create_input_vector(mol_features, solvent_data):
    """Crée le vecteur d'entrée pour le modèle"""
    input_features = {}
    
    # Ajoute les features moléculaires
    input_features.update(mol_features)
    
    # Ajoute les compositions de solvants
    perc_type = solvent_data['percentage_type']
    for i, (solvent, percentage) in enumerate(zip(solvent_data['names'], solvent_data['percentages']), 1):
        input_features[f"{perc_type}{i}"] = percentage
    
    # Crée un DataFrame avec les colonnes dans le bon ordre
    if 'expected_features' in st.session_state:
        input_df = pd.DataFrame(columns=st.session_state['expected_features'])
        for feat in st.session_state['expected_features']:
            input_df[feat] = [input_features.get(feat, 0)]
    else:
        input_df = pd.DataFrame([input_features])
    
    return input_df

def main():
    # Charge la liste des solvants depuis les fichiers
    all_solvents = load_solvents_from_files()
    if not all_solvents:
        st.error("No solvents found in data files!")
        return
    
    # Charge le modèle
    model = load_model()
    if not model:
        return

    # Interface utilisateur
    st.header("1. Molecule Input")
    smiles = st.text_input("Enter SMILES:", "CCO")
    
    if not smiles:
        return
    
    mol_features = calculate_molecular_features(smiles)
    if not mol_features:
        return
    
    # Affichage molécule
    st.subheader("Molecular Structure")
    img = Draw.MolToImage(Chem.MolFromSmiles(smiles))
    st.image(img, caption=smiles, width=300)

    # Sélection solvants
    st.header("2. Solvent Composition")
    
    percentage_type = st.radio(
        "Percentage type:",
        ['%Vol', '%Mol', '%Mas'],
        horizontal=True
    )
    
    num_solvents = st.slider("Number of solvents", 1, 4, 1)
    
    solvents = []
    percentages = []
    remaining_perc = 100.0
    
    for i in range(num_solvents):
        cols = st.columns([3, 2])
        
        with cols[0]:
            solvent = st.selectbox(
                f"Solvent {i+1}",
                all_solvents,
                key=f"solv_{i}"
            )
            solvents.append(solvent)
        
        with cols[1]:
            if i == num_solvents - 1:
                perc = st.number_input(
                    f"Percentage {solvent}",
                    min_value=0.0,
                    max_value=remaining_perc,
                    value=remaining_perc,
                    key=f"perc_{i}"
                )
            else:
                max_val = min(100.0, remaining_perc)
                perc = st.number_input(
                    f"Percentage {solvent}",
                    min_value=0.0,
                    max_value=max_val,
                    value=(100.0/num_solvents) if i == 0 else 0.0,
                    key=f"perc_{i}"
                )
            
            percentages.append(perc)
            remaining_perc -= perc
    
    solvent_data = {
        'names': solvents,
        'percentages': percentages,
        'percentage_type': percentage_type
    }
    
    # Prédiction
    st.header("3. Prediction")
    
    if st.button("Predict Log KD"):
        input_df = create_input_vector(mol_features, solvent_data)
        
        # Debug
        if st.session_state.get('debug_mode', False):
            st.write("Input DataFrame:", input_df)
            st.write("Columns:", input_df.columns.tolist())
        
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Log KD: {prediction:.2f}")
            
            # Affiche le système de solvants
            system_desc = " + ".join([
                f"{p}% {n}" for n, p in zip(solvents, percentages)
            ])
            st.write(f"Solvent System: {system_desc} ({percentage_type})")
            
            # Importance des features
            st.subheader("Feature Importance")
            importances = pd.DataFrame({
                'Feature': input_df.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importances, ax=ax)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# Debug mode
if st.sidebar.checkbox("Debug Mode"):
    st.session_state.debug_mode = True

if __name__ == "__main__":
    main()
