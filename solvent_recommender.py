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
DATA_CACHE_PATH = "processed_data_cache.pkl"

# Solvent database
SOLVENT_DB = {
    'Water': {'SMILES': 'O', 'Type': 'Polar'},
    'Methanol': {'SMILES': 'CO', 'Type': 'Polar protic'},
    'Ethanol': {'SMILES': 'CCO', 'Type': 'Polar protic'},
    'Acetonitrile': {'SMILES': 'CC#N', 'Type': 'Polar aprotic'},
    'Chloroform': {'SMILES': 'ClC(Cl)Cl', 'Type': 'Non-polar'},
    'Hexane': {'SMILES': 'CCCCCC', 'Type': 'Non-polar'},
    'Acetone': {'SMILES': 'CC(=O)C', 'Type': 'Polar aprotic'},
    'DMSO': {'SMILES': 'CS(=O)C', 'Type': 'Polar aprotic'}
}

def load_model():
    """Charge le modèle avec vérification des features"""
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            # Vérification des features attendues
            if hasattr(model, 'feature_names_in_'):
                st.session_state['model_features'] = list(model.feature_names_in_)
            return model
        st.error("Model file not found!")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def calculate_molecular_features(smiles):
    """Calcule TOUTES les features nécessaires dans le bon ordre"""
    if not smiles:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        st.error("Invalid SMILES string")
        return None
    
    features = {
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
    
    # Ordonne les features selon le modèle si disponible
    if 'model_features' in st.session_state:
        ordered_features = {}
        for feat in st.session_state['model_features']:
            if feat in features:
                ordered_features[feat] = features[feat]
            elif feat.startswith(('%Vol', '%Mol', '%Mas')):
                ordered_features[feat] = 0  # Initialisé à 0 pour les solvants
            else:
                ordered_features[feat] = 0  # Valeur par défaut si feature manquante
        return ordered_features
    
    return features

def create_input_vector(mol_features, solvent_data):
    """Crée un input parfaitement formaté pour le modèle"""
    # Crée une copie pour ne pas modifier l'original
    input_features = mol_features.copy()
    
    # Ajoute les compositions de solvants
    perc_type = solvent_data['percentage_type']
    for i, percentage in enumerate(solvent_data['percentages'], 1):
        input_features[f"{perc_type}{i}"] = percentage
    
    # Convertit en DataFrame avec les colonnes dans le bon ordre
    if 'model_features' in st.session_state:
        input_df = pd.DataFrame(columns=st.session_state['model_features'])
        for feat in st.session_state['model_features']:
            input_df[feat] = [input_features.get(feat, 0)]
    else:
        input_df = pd.DataFrame([input_features])
    
    return input_df

def main():
    # Charge le modèle
    model = load_model()
    if not model:
        return

    # Input molécule
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
                list(SOLVENT_DB.keys()),
                key=f"solv_{i}"
            )
            solvents.append(solvent)
        
        with cols[1]:
            if i == num_solvents - 1:
                # Dernier solvant prend le reste
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
        
        # Debug: Affiche les features
        if st.session_state.get('debug_mode', False):
            st.write("Input DataFrame:", input_df)
            st.write("Columns:", input_df.columns.tolist())
            st.write("Model features:", st.session_state.get('model_features', []))
        
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
            if 'model_features' in st.session_state:
                st.write("Model expects features in this exact order:")
                st.write(st.session_state['model_features'])

# Debug mode
if st.sidebar.checkbox("Debug Mode"):
    st.session_state.debug_mode = True

if __name__ == "__main__":
    main()
