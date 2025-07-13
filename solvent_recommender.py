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
    try:
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
        st.error("Model file not found!")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def calculate_molecular_features(smiles):
    """Calculate ALL features needed by the model"""
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
    """Create input with EXACTLY the features the model expects"""
    # Template with all expected features initialized to 0
    input_template = {
        # Molecular features
        'MolWeight': 0, 'LogP': 0, 'HBD': 0, 'HBA': 0,
        'TPSA': 0, 'RotatableBonds': 0, 'AromaticRings': 0,
        'HeavyAtoms': 0, 'RingCount': 0, 'FractionCSP3': 0,
        
        # Solvent composition features
        '%Vol1': 0, '%Vol2': 0, '%Vol3': 0, '%Vol4': 0,
        '%Mol1': 0, '%Mol2': 0, '%Mol3': 0, '%Mol4': 0,
        '%Mas1': 0, '%Mas2': 0, '%Mas3': 0, '%Mas4': 0
    }
    
    # Update with actual molecular features
    input_template.update(mol_features)
    
    # Update with solvent percentages
    perc_type = solvent_data['percentage_type']
    for i, percentage in enumerate(solvent_data['percentages'], 1):
        input_template[f"{perc_type}{i}"] = percentage
    
    return pd.DataFrame([input_template])

def main():
    model = load_model()
    if not model:
        return

    # Molecule input
    st.header("1. Molecule Input")
    smiles = st.text_input("Enter SMILES:", "CCO")
    
    if not smiles:
        return
    
    mol_features = calculate_molecular_features(smiles)
    if not mol_features:
        return
    
    # Display molecule
    st.subheader("Molecular Structure")
    img = Draw.MolToImage(Chem.MolFromSmiles(smiles))
    st.image(img, caption=smiles, width=300)

    # Solvent selection
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
                # Last solvent gets remaining percentage
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
    
    # Prediction
    st.header("3. Prediction")
    
    if st.button("Predict Log KD"):
        input_df = create_input_vector(mol_features, solvent_data)
        
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Log KD: {prediction:.2f}")
            
            # Display solvent system
            system_desc = " + ".join([
                f"{p}% {n}" for n, p in zip(solvents, percentages)
            ])
            st.write(f"System: {system_desc} ({percentage_type})")
            
            # Feature importance
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

if __name__ == "__main__":
    main()
