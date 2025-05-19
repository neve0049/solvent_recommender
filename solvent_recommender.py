import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Draw
import numpy as np
from io import BytesIO
from PIL import Image

# Configuration
st.set_page_config(page_title="Solvent System Recommender", layout="wide")
st.title("🔬 Solvent System Recommender")

# Debug flag
debug_mode = st.checkbox("Enable debug mode")

def show_debug_data(data, name):
    if debug_mode:
        st.write(f"Debug - {name} columns:", data.columns.tolist())
        st.write(f"Debug - {name} sample data:", data.head(3))

@st.cache_data
def load_data():
    try:
        kddb = pd.read_excel("KDDB.xlsx", sheet_name=None)
        dbdq = pd.read_excel("DBDQ.xlsx", sheet_name=None)
        dbdt = pd.read_excel("DBDT.xlsx", sheet_name=None)
        
        if debug_mode:
            show_debug_data(pd.concat(kddb.values()), "KDDB")
            show_debug_data(pd.concat(dbdq.values()), "DBDQ") 
            show_debug_data(pd.concat(dbdt.values()), "DBDT")
            
        return kddb, dbdq, dbdt
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None, None

def extract_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        return {
            'MolWeight': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBD': Lipinski.NumHDonors(mol),
            'HBA': Lipinski.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'RotatableBonds': Lipinski.NumRotatableBonds(mol),
            'AromaticRings': Lipinski.NumAromaticRings(mol),
            'HeavyAtoms': Lipinski.HeavyAtomCount(mol)
        }
    except Exception as e:
        if debug_mode:
            st.warning(f"Error extracting features from SMILES: {str(e)}")
        return None

def prepare_training_data(kddb, dbdq, dbdt):
    data = []
    
    for sheet_name, sheet_data in kddb.items():
        for _, row in sheet_data.iterrows():
            try:
                # Skip if essential data is missing
                if pd.isna(row['SMILES']) or pd.isna(row['Log KD']) or pd.isna(row['System']):
                    continue
                    
                features = extract_features(row['SMILES'])
                if features is None:
                    continue
                    
                system_name = row['System']
                composition = str(row['Composition']).strip()
                
                # Find matching solvent composition
                solvent_data = None
                for db in [dbdq, dbdt]:
                    if system_name in db:
                        system_df = db[system_name]
                        match = system_df[system_df['Composition'].astype(str) == composition]
                        if not match.empty:
                            solvent_data = match.iloc[0].to_dict()
                            break
                
                if not solvent_data:
                    if debug_mode:
                        st.warning(f"No solvent data found for {system_name}-{composition}")
                    continue
                    
                # Create base record with molecular features
                record = {
                    **features,
                    'Log_KD': float(row['Log KD']),
                    'System': system_name,
                    'Composition_ID': composition
                }
                
                # Add ALL solvent composition features (critical part)
                for col, val in solvent_data.items():
                    if any(col.startswith(p) for p in ['%Vol', '%Mol', '%Mas']):
                        try:
                            # Normalize column names
                            norm_col = col.replace(" ", "_")
                            record[norm_col] = float(val) if not pd.isna(val) else 0.0
                        except:
                            record[norm_col] = 0.0
                
                data.append(record)
                
            except Exception as e:
                if debug_mode:
                    st.warning(f"Error processing row: {str(e)}")
                continue
    
    df = pd.DataFrame(data)
    
    if debug_mode and not df.empty:
        st.write("Debug - Training data summary:")
        st.write(f"Total records: {len(df)}")
        st.write("Log KD distribution:", df['Log_KD'].describe())
        
        # Show composition features
        comp_features = [c for c in df.columns if any(c.startswith(p) for p in ['%Vol', '%Mol', '%Mas'])]
        if comp_features:
            st.write("Sample composition features:", df[comp_features[:3] + ['Log_KD']].head())
    
    return df

def train_model(df):
    if df.empty:
        st.error("No training data available!")
        return None, None, None
        
    # Molecular descriptors
    base_features = [
        'MolWeight', 'LogP', 'HBD', 'HBA', 'TPSA',
        'RotatableBonds', 'AromaticRings', 'HeavyAtoms'
    ]
    
    # Automatically detect ALL solvent composition features
    solvent_features = [col for col in df.columns 
                       if any(col.startswith(p) for p in ['%Vol_', '%Mol_', '%Mas_']]
    
    # Combine all features
    feature_cols = base_features + solvent_features
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    if debug_mode:
        st.write("DEBUG: Using features:", feature_cols)
        st.write(f"Total features: {len(feature_cols)}")
        st.write(f"Molecular features: {len(base_features)}")
        st.write(f"Solvent features: {len(solvent_features)}")
    
    X = df[feature_cols]
    y = df['Log_KD']
    
    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Enhanced Random Forest model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=3,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    
    if debug_mode:
        st.write("Model performance:")
        st.write(f"Train R²: {train_score:.3f}, Test R²: {test_score:.3f}, RMSE: {rmse:.3f}")
        
        # Show feature importances
        importances = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.write("Top 10 features:", importances.head(10))
    
    return model, feature_cols, rmse

def predict_for_systems(model, features, feature_cols, dbdq, dbdt):
    results = []
    
    for system_name, system_data in {**dbdq, **dbdt}.items():
        for _, solvent_row in system_data.iterrows():
            try:
                input_features = features.copy()
                composition = str(solvent_row['Composition'])
                
                # Add system info
                input_features['System'] = system_name
                input_features['Composition_ID'] = composition
                
                # Add ALL solvent composition features
                for col in solvent_row.index:
                    if any(col.startswith(p) for p in ['%Vol', '%Mol', '%Mas']):
                        norm_col = col.replace(" ", "_")
                        try:
                            input_features[norm_col] = float(solvent_row[col]) if not pd.isna(solvent_row[col]) else 0.0
                        except:
                            input_features[norm_col] = 0.0
                
                # Create input dataframe
                input_df = pd.DataFrame([input_features])
                
                # Ensure all features are present
                missing_cols = set(feature_cols) - set(input_df.columns)
                for col in missing_cols:
                    input_df[col] = 0.0
                
                input_df = input_df[feature_cols]
                
                # Predict
                log_kd = model.predict(input_df)[0]
                
                if debug_mode and len(results) < 3:
                    st.write(f"Debug prediction sample for {system_name}-{composition}:")
                    st.write("Input features:", input_df.iloc[0][feature_cols[:5]].to_dict())
                    st.write(f"Predicted Log KD: {log_kd:.3f}")
                
                if -1 <= log_kd <= 1:
                    # Get composition details
                    composition_parts = []
                    solvent_names = []
                    for i in range(1, 5):
                        vol_col = f'%Vol{i}_-_UP'
                        name_col = f'Solvent {i}'
                        
                        if vol_col in solvent_row:
                            composition_parts.append(f"{solvent_row[vol_col]:.1f}%")
                        if name_col in solvent_row:
                            solvent_names.append(str(solvent_row[name_col]))
                    
                    results.append({
                        "System": system_name,
                        "Solvents": " / ".join(solvent_names),
                        "Composition": " / ".join(composition_parts),
                        "Composition_ID": composition,
                        "Predicted_Log_KD": log_kd
                    })
                    
            except Exception as e:
                if debug_mode:
                    st.warning(f"Skipping {system_name}-{composition}: {str(e)}")
                continue
    
    return results

def main():
    st.markdown("""
    ## Solvent System Recommender
    Enter a SMILES string to get solvent system recommendations with predicted log KD between -1 and 1.
    """)
    
    # Load data
    with st.spinner("Loading data..."):
        kddb, dbdq, dbdt = load_data()
    
    if kddb is None or dbdq is None or dbdt is None:
        st.error("Failed to load data files. Please check the Excel files.")
        return
    
    # Prepare training data
    with st.spinner("Preparing training data..."):
        training_data = prepare_training_data(kddb, dbdq, dbdt)
    
    if training_data.empty:
        st.error("No valid training data could be prepared. Possible issues:")
        st.write("- Invalid SMILES in KDDB")
        st.write("- Missing System/Composition matches between files")
        st.write("- Invalid Log KD values")
        return
    
    # Train model
    with st.spinner("Training model (this may take a few minutes)..."):
        model, feature_cols, rmse = train_model(training_data)
    
    if model is None:
        st.error("Model training failed.")
        return
    
    # User input
    st.subheader("Molecule Input")
    smiles = st.text_input("Enter SMILES string", value="C1=CC(=CC=C1/C=C/C2=CC(=CC(=C2)O)O)O")
    
    if st.button("Find Suitable Solvent Systems"):
        if not smiles:
            st.warning("Please enter a SMILES string")
            return
            
        with st.spinner("Calculating molecular features..."):
            features = extract_features(smiles)
        
        if features is None:
            st.error("Invalid SMILES string")
            return

        # Display molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.subheader("Molecular Structure")
            img = Draw.MolToImage(mol, size=(400, 400))
            buf = BytesIO()
            img.save(buf, format="PNG")
            st.image(buf.getvalue(), width=300)

        # Make predictions
        with st.spinner("Analyzing solvent systems..."):
            results = predict_for_systems(model, features, feature_cols, dbdq, dbdt)

        if results:
            st.subheader("🔍 Recommended Solvent Systems")
            df_results = pd.DataFrame(results)
            
            # Convert KD to numeric and sort by proximity to 0
            df_results['KD_abs'] = df_results['Predicted_Log_KD'].abs()
            df_results = df_results.sort_values('KD_abs')
            
            # Format output
            st.dataframe(
                df_results[['System', 'Solvents', 'Composition', 'Predicted_Log_KD']],
                column_config={
                    "Predicted_Log_KD": st.column_config.NumberColumn(
                        format="%.2f",
                        help="Predicted log KD value (ideal between -1 and 1)"
                    )
                },
                use_container_width=True,
                hide_index=True
            )
            
            if debug_mode:
                st.write("Debug - Full prediction results:", df_results)
        else:
            st.warning("No solvent systems found with -1 ≤ predicted log KD ≤ 1")
            if debug_mode:
                st.write("Debug - Input features:", features)
                st.write("Debug - Feature columns:", feature_cols)

if __name__ == "__main__":
    main()
