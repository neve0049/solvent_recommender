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
        st.write(f"Debug - {name} columns: {data.columns.tolist()}")
        st.write(f"Debug - {name} first rows:", data.head())

@st.cache_data
def load_data():
    try:
        kddb = pd.read_excel("KDDB.xlsx", sheet_name=None)
        dbdq = pd.read_excel("DBDQ.xlsx", sheet_name=None)
        dbdt = pd.read_excel("DBDT.xlsx", sheet_name=None)
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
    except:
        return None

def prepare_training_data(kddb, dbdq, dbdt):
    data = []
    
    for sheet_name, sheet_data in kddb.items():
        for _, row in sheet_data.iterrows():
            try:
                if pd.isna(row['SMILES']) or pd.isna(row['Log KD']) or pd.isna(row['System']):
                    continue
                    
                features = extract_features(row['SMILES'])
                if features is None:
                    continue
                    
                system_name = row['System']
                composition = str(row['Composition']).strip()
                
                solvent_data = None
                for db in [dbdq, dbdt]:
                    if system_name in db:
                        solvent_sheet = db[system_name]
                        solvent_row = solvent_sheet[solvent_sheet['Composition'].astype(str) == composition]
                        if not solvent_row.empty:
                            solvent_data = solvent_row.iloc[0].to_dict()
                            break
                
                if not solvent_data:
                    continue
                    
                record = {
                    **features,
                    'Log_KD': float(row['Log KD']),
                    'System': system_name,
                    'Composition_Number': composition
                }
                
                # Add all composition features
                for col in solvent_data:
                    if any(col.startswith(prefix) for prefix in ['%Vol', '%Mol', '%Mas']):
                        try:
                            record[col] = float(solvent_data[col]) if not pd.isna(solvent_data[col]) else 0.0
                        except:
                            record[col] = 0.0
                
                data.append(record)
            except Exception as e:
                if debug_mode:
                    st.warning(f"Skipping row: {str(e)}")
                continue
    
    return pd.DataFrame(data)

def train_model(df):
    if df.empty:
        st.error("Training data is empty!")
        return None, None, None
        
    # Feature selection
    feature_cols = [
        'MolWeight', 'LogP', 'HBD', 'HBA', 'TPSA', 
        'RotatableBonds', 'AromaticRings', 'HeavyAtoms'
    ]
    
    # Add all solvent composition features
    solvent_features = [col for col in df.columns 
                       if any(col.startswith(prefix) for prefix in ['%Vol', '%Mol', '%Mas'])]
    feature_cols.extend(solvent_features)
    
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols]
    y = df['Log_KD']
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Enhanced model with regularization
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    if debug_mode:
        st.write("Debug - Model Evaluation:")
        st.write(f"Train R²: {train_score:.3f}, Test R²: {test_score:.3f}, RMSE: {rmse:.3f}")
        st.write("Feature importances:", sorted(zip(feature_cols, model.feature_importances_), 
                key=lambda x: x[1], reverse=True))
        st.write("Predicted vs Actual sample:")
        eval_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        st.write(eval_df.head(10))
    
    return model, feature_cols, rmse

def predict_for_systems(model, features, feature_cols, dbdq, dbdt):
    results = []
    
    for system_name, system_data in {**dbdq, **dbdt}.items():
        for _, solvent_row in system_data.iterrows():
            try:
                input_features = features.copy()
                composition = str(solvent_row['Composition'])
                
                # Add system and composition info
                input_features['System'] = system_name
                input_features['Composition_Number'] = composition
                
                # Add all solvent features
                for col in solvent_row.index:
                    if any(col.startswith(prefix) for prefix in ['%Vol', '%Mol', '%Mas']):
                        try:
                            input_features[col] = float(solvent_row[col]) if not pd.isna(solvent_row[col]) else 0.0
                        except:
                            input_features[col] = 0.0
                
                # Create input dataframe
                input_df = pd.DataFrame([input_features])
                
                # Ensure correct feature order and columns
                missing_cols = set(feature_cols) - set(input_df.columns)
                for col in missing_cols:
                    input_df[col] = 0.0
                input_df = input_df[feature_cols]
                
                # Predict
                log_kd = model.predict(input_df)[0]
                
                if debug_mode and len(results) < 5:  # Show first few predictions
                    st.write(f"Debug - Sample prediction for {system_name}-{composition}:")
                    st.write(input_df.iloc[0])
                    st.write(f"Predicted Log KD: {log_kd:.3f}")
                
                if -1 <= log_kd <= 1:
                    # Get composition details
                    composition_parts = []
                    solvent_names = []
                    for i in range(1, 5):
                        vol_col = f'%Vol{i} - UP'
                        name_col = f'Solvent {i}'
                        if vol_col in solvent_row and not pd.isna(solvent_row[vol_col]):
                            composition_parts.append(f"{solvent_row[vol_col]:.1f}%")
                        if name_col in solvent_row and not pd.isna(solvent_row[name_col]):
                            solvent_names.append(str(solvent_row[name_col]))
                    
                    results.append({
                        "System": system_name,
                        "Solvents": " / ".join(solvent_names),
                        "Composition": " / ".join(composition_parts),
                        "Composition ID": composition,
                        "Predicted Log KD": log_kd
                    })
            except Exception as e:
                if debug_mode:
                    st.warning(f"Skipping {system_name}-{solvent_row['Composition']}: {str(e)}")
                continue
    
    return results

def main():
    st.markdown("""
    Entrez une chaîne SMILES pour obtenir des recommandations de systèmes de solvants avec un log KD prédit entre -1 et 1.
    """)
    
    # Load data
    with st.spinner("Chargement des données..."):
        kddb, dbdq, dbdt = load_data()
    
    if kddb is None or dbdq is None or dbdt is None:
        return
    
    # Prepare training data
    with st.spinner("Préparation des données d'entraînement..."):
        training_data = prepare_training_data(kddb, dbdq, dbdt)
        if debug_mode:
            show_debug_data(training_data, "Training Data")
            if not training_data.empty:
                st.write("Debug - Log KD statistics:")
                st.write(training_data['Log_KD'].describe())
                st.write("Debug - Sample training instances:")
                st.write(training_data[['System', 'Composition_Number', 'Log_KD']].head(10))
    
    if training_data.empty:
        st.error("Aucune donnée d'entraînement valide.")
        return
    
    # Train model
    with st.spinner("Entraînement du modèle (cela peut prendre quelques minutes)..."):
        model, feature_cols, rmse = train_model(training_data)
    
    if model is None:
        st.error("Échec de l'entraînement du modèle.")
        return
    
    # User input
    st.subheader("Entrée de la molécule")
    smiles = st.text_input("Entrez une chaîne SMILES", value="C1=CC(=CC=C1/C=C/C2=CC(=CC(=C2)O)O)O")
    
    if st.button("Trouver des systèmes de solvants appropriés"):
        if not smiles:
            st.warning("Veuillez entrer une chaîne SMILES")
            return
            
        with st.spinner("Calcul des caractéristiques moléculaires..."):
            features = extract_features(smiles)
        
        if features is None:
            st.error("SMILES invalide.")
            return

        # Display molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.subheader("Structure moléculaire")
            img = Draw.MolToImage(mol, size=(400, 400))
            buf = BytesIO()
            img.save(buf, format="PNG")
            st.image(buf.getvalue(), width=200)

        # Make predictions
        with st.spinner("Analyse des systèmes de solvants..."):
            results = predict_for_systems(model, features, feature_cols, dbdq, dbdt)

        if results:
            st.subheader("🔍 Meilleurs systèmes prédits")
            df_results = pd.DataFrame(results)
            
            # Convert KD to numeric and sort
            df_results['KD_value'] = df_results['Predicted Log KD'].astype(float)
            df_results['Abs_KD'] = df_results['KD_value'].abs()
            df_results = df_results.sort_values('Abs_KD')
            
            # Format output
            st.dataframe(
                df_results[['System', 'Solvents', 'Composition', 'Predicted Log KD']],
                column_config={
                    "Predicted Log KD": st.column_config.NumberColumn(
                        format="%.2f",
                        help="Valeur prédite du log KD (idéalement proche de 0)"
                    )
                },
                use_container_width=True,
                hide_index=True
            )
            
            if debug_mode:
                st.write("Debug - All predictions:")
                st.write(df_results)
        else:
            st.warning("Aucun système trouvé avec -1 ≤ log KD ≤ 1")
            if debug_mode:
                st.write("Debug - Input features:", features)
                st.write("Debug - Feature columns:", feature_cols)

if __name__ == "__main__":
    main()
