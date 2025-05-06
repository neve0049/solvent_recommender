import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import numpy as np
import joblib
import os

# Configuration
st.set_page_config(page_title="Solvent System Recommender", layout="wide")
st.title("🔬 Solvent System Recommender")

# Load data functions
@st.cache_data
def load_data():
    # Load KDDB data
    kddb = pd.read_excel("KDDB.xlsx", sheet_name=None)
    
    # Load solvent system data
    dbdq = pd.read_excel("DBDQ.xlsx", sheet_name=None)
    dbdt = pd.read_excel("DBDT.xlsx", sheet_name=None)
    
    return kddb, dbdq, dbdt

def extract_features(smiles):
    """Calculate molecular features from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    features = {
        'MolWeight': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'RotatableBonds': Lipinski.NumRotatableBonds(mol),
        'AromaticRings': Lipinski.NumAromaticRings(mol),
        'HeavyAtoms': Lipinski.HeavyAtomCount(mol)
    }
    return features

def prepare_training_data(kddb, dbdq, dbdt):
    """Prepare training data from all Excel files"""
    data = []
    
    for sheet_name, sheet_data in kddb.items():
        for _, row in sheet_data.iterrows():
            if pd.isna(row['SMILES']) or pd.isna(row['Log KD']):
                continue
                
            # Get molecular features
            features = extract_features(row['SMILES'])
            if features is None:
                continue
                
            # Get solvent system data
            system_name = row['System']
            number = str(row['Number'])
            
            solvent_data = None
            if system_name in dbdq:
                solvent_sheet = dbdq[system_name]
                solvent_row = solvent_sheet[solvent_sheet['Number'] == number]
            elif system_name in dbdt:
                solvent_sheet = dbdt[system_name]
                solvent_row = solvent_sheet[solvent_sheet['Number'] == number]
            else:
                continue
                
            if solvent_row.empty:
                continue
                
            # Combine all features
            record = {
                **features,
                'Log_KD': row['Log KD'],
                'System': system_name,
                'Composition_Number': number
            }
            
            # Add solvent composition features
            for col in solvent_row.columns:
                if col.startswith('%Vol') or col.startswith('%Mol') or col.startswith('%Mas'):
                    record[col] = solvent_row[col].values[0]
            
            data.append(record)
    
    return pd.DataFrame(data)

# Train model function
def train_model(df):
    """Train a RandomForest model on the prepared data"""
    if df.empty:
        return None
        
    # Select features and target
    feature_cols = [
        'MolWeight', 'LogP', 'HBD', 'HBA', 'TPSA', 
        'RotatableBonds', 'AromaticRings', 'HeavyAtoms'
    ]
    
    # Add solvent composition features
    solvent_features = [col for col in df.columns if col.startswith('%Vol') or 
                       col.startswith('%Mol') or col.startswith('%Mas')]
    feature_cols.extend(solvent_features)
    
    X = df[feature_cols]
    y = df['Log_KD']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, feature_cols

# Main app
def main():
    st.markdown("""
    Enter a SMILES string to get solvent system recommendations with predicted log KD between -1 and 1.
    """)
    
    # Load data
    kddb, dbdq, dbdt = load_data()
    
    # Prepare training data
    training_data = prepare_training_data(kddb, dbdq, dbdt)
    
    if training_data.empty:
        st.error("No valid training data could be prepared from the provided files.")
        return
    
    # Train model
    model, feature_cols = train_model(training_data)
    
    if model is None:
        st.error("Failed to train model.")
        return
    
    # Get user input
    st.subheader("Molecule Input")
    smiles = st.text_input("Enter SMILES string", value="C1=CC(=CC=C1/C=C/C2=CC(=CC(=C2)O)O)O")
    
    if st.button("Find Suitable Solvent Systems"):
        # Calculate molecular features
        features = extract_features(smiles)
        if features is None:
            st.error("Invalid SMILES string. Please enter a valid SMILES.")
            return
            
        results = []
        
        # Predict for all solvent systems
        for system_name, system_data in {**dbdq, **dbdt}.items():
            for _, solvent_row in system_data.iterrows():
                # Prepare input features
                input_features = {
                    **features,
                    'System': system_name,
                    'Composition_Number': solvent_row['Number']
                }
                
                # Add solvent composition features
                for col in solvent_row.index:
                    if col.startswith('%Vol') or col.startswith('%Mol') or col.startswith('%Mas'):
                        input_features[col] = solvent_row[col]
                
                # Create DataFrame row
                input_df = pd.DataFrame([input_features])
                
                # Ensure all feature columns are present
                for col in feature_cols:
                    if col not in input_df:
                        input_df[col] = 0  # Fill missing with 0 (shouldn't happen with proper data)
                
                # Predict
                log_kd = model.predict(input_df[feature_cols])[0]
                
                if -1 <= log_kd <= 1:
                    # Get composition details
                    composition = []
                    for col in ['%Vol1 - UP', '%Vol2 - UP', '%Vol3 - UP', '%Vol4 - UP']:
                        if col in solvent_row:
                            val = solvent_row[col]
                            if not pd.isna(val):
                                composition.append(f"{val:.3f}")
                    
                    results.append({
                        "System": system_name,
                        "Composition": " / ".join(composition),
                        "Number": solvent_row['Number'],
                        "Predicted Log KD": f"{log_kd:.2f}"
                    })
        
        if results:
            st.success(f"Found {len(results)} suitable solvent systems")
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, hide_index=True)
            
            # Download button
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results",
                data=csv,
                file_name="solvent_recommendations.csv",
                mime="text/csv"
            )
        else:
            st.warning("No solvent systems found with log KD between -1 and 1")

if __name__ == "__main__":
    main()
