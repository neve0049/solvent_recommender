import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Draw
import numpy as np
from io import BytesIO
from PIL import Image

# Configuration
st.set_page_config(page_title="Solvent System Recommender", layout="wide")
st.title("🔬 Solvent System Recommender")

# Debug flag
debug_mode = st.checkbox("Enable debug mode", value=True)

def load_data():
    try:
        kddb = pd.read_excel("KDDB.xlsx", sheet_name=None)
        dbdq = pd.read_excel("DBDQ.xlsx", sheet_name=None)
        dbdt = pd.read_excel("DBDT.xlsx", sheet_name=None)
        
        # Debug: Check system names match between files
        if debug_mode:
            kddb_systems = set()
            for sheet in kddb.values():
                kddb_systems.update(sheet['System'].unique())
            
            db_systems = set()
            for db in [dbdq, dbdt]:
                for sheet in db.values():
                    db_systems.update(sheet['System'].unique())
            
            st.write(f"Systems in KDDB but not in DBDQ/DBDT: {kddb_systems - db_systems}")
            st.write(f"Systems in DBDQ/DBDT but not in KDDB: {db_systems - kddb_systems}")
            
        return kddb, dbdq, dbdt
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None, None

def prepare_training_data(kddb, dbdq, dbdt):
    data = []
    composition_stats = []
    
    for sheet_name, sheet_data in kddb.items():
        for _, row in sheet_data.iterrows():
            try:
                # Skip if essential data is missing
                if pd.isna(row['SMILES']) or pd.isna(row['Log KD']) or pd.isna(row['System']):
                    continue
                    
                # Get molecular features
                mol = Chem.MolFromSmiles(row['SMILES'])
                if mol is None:
                    continue
                    
                features = {
                    'MolWeight': Descriptors.MolWt(mol),
                    'LogP': Descriptors.MolLogP(mol),
                    'HBD': Lipinski.NumHDonors(mol),
                    'HBA': Lipinski.NumHAcceptors(mol),
                    'TPSA': Descriptors.TPSA(mol),
                    'RotatableBonds': Lipinski.NumRotatableBonds(mol),
                    'AromaticRings': Lipinski.NumAromaticRings(mol),
                    'HeavyAtoms': Lipinski.HeavyAtomCount(mol),
                    'Log_KD': float(row['Log KD'])
                }
                
                # Get solvent composition
                system_name = row['System']
                composition = str(row['Composition']).strip()
                solvent_data = None
                
                # Search in both DBDQ and DBDT
                for db in [dbdq, dbdt]:
                    if system_name in db:
                        system_df = db[system_name]
                        match = system_df[system_df['Composition'].astype(str) == composition]
                        if not match.empty:
                            solvent_data = match.iloc[0].to_dict()
                            break
                
                if not solvent_data:
                    if debug_mode:
                        st.warning(f"No match for {system_name}-{composition}")
                    continue
                
                # Add composition features
                comp_features = {}
                for i in range(1, 5):  # For up to 4 solvents
                    for prefix in ['%Vol', '%Mol', '%Mas']:
                        col = f"{prefix}{i} - UP"
                        if col in solvent_data:
                            try:
                                value = float(solvent_data[col])
                                comp_features[f"{prefix}{i}"] = value
                                composition_stats.append({
                                    'System': system_name,
                                    'Composition': composition,
                                    'Feature': f"{prefix}{i}",
                                    'Value': value
                                })
                            except:
                                comp_features[f"{prefix}{i}"] = 0.0
                
                record = {**features, **comp_features}
                data.append(record)
                
            except Exception as e:
                if debug_mode:
                    st.warning(f"Error processing row: {str(e)}")
                continue
    
    df = pd.DataFrame(data)
    
    if debug_mode:
        # Analyze composition features
        if composition_stats:
            comp_df = pd.DataFrame(composition_stats)
            st.write("Composition Features Analysis:")
            
            # Check for constant features
            st.write("Features with no variation:")
            constant_features = []
            for feat in comp_df['Feature'].unique():
                if comp_df[comp_df['Feature'] == feat]['Value'].nunique() == 1:
                    constant_features.append(feat)
            st.write(constant_features)
            
            # Show value distributions
            st.write("Value distributions per feature:")
            for feat in comp_df['Feature'].unique():
                st.write(f"{feat}:")
                st.write(comp_df[comp_df['Feature'] == feat]['Value'].describe())
        
        st.write("Training data summary:")
        st.write(f"Total samples: {len(df)}")
        st.write("Log KD distribution:", df['Log_KD'].describe())
    
    return df

def main():
    st.markdown("""
    ## Solvent System Recommender
    Enter a SMILES string to get solvent system recommendations.
    """)
    
    # Load data
    with st.spinner("Loading and validating data..."):
        kddb, dbdq, dbdt = load_data()
        if None in (kddb, dbdq, dbdt):
            return
    
    # Prepare training data
    with st.spinner("Preparing training data..."):
        training_data = prepare_training_data(kddb, dbdq, dbdt)
        if training_data.empty:
            st.error("No valid training data could be prepared.")
            return
    
    # Check for varying composition features
    comp_features = [col for col in training_data.columns 
                    if any(col.startswith(p) for p in ['%Vol', '%Mol', '%Mas'])]
    
    if not comp_features:
        st.error("No composition features found in data!")
        return
    
    varying_features = [col for col in comp_features 
                       if training_data[col].nunique() > 1]
    
    if not varying_features:
        st.error("""
        All composition features are constant (no variation)!
        Possible causes:
        1. All solvent systems have identical compositions
        2. Composition data is not properly linked
        3. Columns names mismatch between files
        """)
        
        if debug_mode:
            st.write("Debug - Sample composition features:")
            st.write(training_data[comp_features].head())
        return
    
    # Proceed with training if we have varying features
    st.success(f"Found {len(varying_features)} varying composition features")
    
    # [Rest of your code...]

if __name__ == "__main__":
    main()
