import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Draw
import numpy as np
from io import BytesIO

# Configuration
st.set_page_config(page_title="Solvent System Recommender", layout="wide")
st.title("🔬 Solvent System Recommender")

# Debug flag (moved outside cached functions)
debug_mode = st.checkbox("Enable debug mode")

# Debug function (modified to not use widgets)
def show_debug_data(data, name):
    if debug_mode:
        st.write(f"Debug - {name} columns: {data.columns.tolist()}")
        st.write(f"Debug - {name} first rows:", data.head())

# Load data functions
@st.cache_data
def load_data():
    try:
        # Load KDDB data
        kddb = pd.read_excel("KDDB.xlsx", sheet_name=None)
        if debug_mode:
            show_debug_data(pd.concat(kddb.values()), "KDDB")

        # Load solvent system data
        dbdq = pd.read_excel("DBDQ.xlsx", sheet_name=None)
        if debug_mode:
            show_debug_data(pd.concat(dbdq.values()), "DBDQ")

        dbdt = pd.read_excel("DBDT.xlsx", sheet_name=None)
        if debug_mode:
            show_debug_data(pd.concat(dbdt.values()), "DBDT")

        return kddb, dbdq, dbdt
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None, None

def extract_features(smiles):
    """Calculate molecular features from SMILES"""
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
    """Prepare training data from all Excel files"""
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
                number = row['Number']

                if isinstance(number, str):
                    number = number.strip()
                else:
                    number = str(int(number)) if not pd.isna(number) else ""

                solvent_data = None

                for db in [dbdq, dbdt]:
                    if system_name in db:
                        solvent_sheet = db[system_name]
                        solvent_row = solvent_sheet[solvent_sheet['Number'].astype(str) == str(number)]
                        if not solvent_row.empty:
                            solvent_data = solvent_row.iloc[0].to_dict()
                            break

                if not solvent_data:
                    continue

                record = {
                    **features,
                    'Log_KD': float(row['Log KD']),
                    'System': system_name,
                    'Composition_Number': number
                }

                for col, val in solvent_data.items():
                    if any(col.startswith(prefix) for prefix in ['%Vol', '%Mol', '%Mas']):
                        try:
                            record[col] = float(val) if not pd.isna(val) else 0.0
                        except:
                            record[col] = 0.0

                data.append(record)
            except Exception as e:
                if debug_mode:
                    st.warning(f"Skipping row due to error: {str(e)}")
                continue

    return pd.DataFrame(data)

def train_model(df):
    if df.empty:
        st.error("Training data is empty!")
        return None, None

    feature_cols = [
        'MolWeight', 'LogP', 'HBD', 'HBA', 'TPSA', 
        'RotatableBonds', 'AromaticRings', 'HeavyAtoms'
    ]

    solvent_features = [col for col in df.s 
                        if any(col.startswith(prefix) for prefix in ['%Vol', '%Mol', '%Mas'])]
    feature_cols.extend(solvent_features)
    feature_cols = [col for col in feature_cols if col in df.s]

    X = df[feature_cols]
    y = df['Log_KD']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, feature_cols

def predict_for_systems(model, features, feature_cols, dbdq, dbdt):
    results = []

    for system_name, system_data in {**dbdq, **dbdt}.items():
        for _, solvent_row in system_data.iterrows():
            try:
                input_features = features.copy()
                input_features['System'] = system_name
                input_features['Composition_Number'] = str(solvent_row['Number'])

                for col in solvent_row.index:
                    if any(col.startswith(prefix) for prefix in ['%Vol', '%Mol', '%Mas']):
                        try:
                            input_features[col] = float(solvent_row[col]) if not pd.isna(solvent_row[col]) else 0.0
                        except:
                            input_features[col] = 0.0

                input_df = pd.DataFrame([input_features])

                for col in feature_cols:
                    if col not in input_df:
                        input_df[col] = 0.0

                log_kd = model.predict(input_df[feature_cols])[0]

                if -1 <= log_kd <= 1:
                    composition = []
                    for i in range(1, 5):
                        col = f'%Vol{i} - UP'
                        if col in solvent_row and not pd.isna(solvent_row[col]):
                            composition.append(f"{float(solvent_row[col]):.3f}")

                    results.append({
                        "System": system_name,
                        "Composition": " / ".join(composition),
                        "Number": str(solvent_row['Number']),
                        "Predicted Log KD": f"{log_kd:.2f}"
                    })
            except Exception as e:
                if debug_mode:
                    st.warning(f"Skipping prediction for {system_name}-{solvent_row['Number']}: {str(e)}")
                continue

    return results

# Main app
def main():
    st.markdown("""
    Enter a SMILES string to get solvent system recommendations with predicted log KD between -1 and 1.
    """)

    with st.spinner("Loading data..."):
        kddb, dbdq, dbdt = load_data()

    if kddb is None or dbdq is None or dbdt is None:
        st.error("Failed to load data files. Please check the files exist and are valid Excel files.")
        return

    with st.spinner("Preparing training data..."):
        training_data = prepare_training_data(kddb, dbdq, dbdt)
        if debug_mode:
            show_debug_data(training_data, "Training Data")

    if training_data.empty:
        st.error("No valid training data could be prepared.")
        return

    with st.spinner("Training model..."):
        model, feature_cols = train_model(training_data)

    if model is None:
        st.error("Failed to train model.")
        return

    st.subheader("Molecule Input")
    smiles = st.text_input("Enter SMILES string", value="C1=CC(=CC=C1/C=C/C2=CC(=CC(=C2)O)O)O")

    if st.button("Find Suitable Solvent Systems"):
        if not smiles:
            st.warning("Please enter a SMILES string")
            return

        with st.spinner("Calculating molecular features..."):
            features = extract_features(smiles)

        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.subheader("Molecular Structure")
            img = Draw.MolToImage(mol, size=(300, 300))
            st.image(img, caption="Structure of the molecule", use_container_width=False)
        else:
            st.warning("Unable to parse the SMILES into a valid molecule.")

        if features is None:
            st.error("Invalid SMILES string. Please enter a valid SMILES.")
            return

        with st.spinner("Searching for suitable solvent systems..."):
            results = predict_for_systems(model, features, feature_cols, dbdq, dbdt)

        if results:
            st.success(f"Found {len(results)} suitable solvent systems")
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, hide_index=True)

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
