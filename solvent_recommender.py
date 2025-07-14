import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import BytesIO
import webbrowser
import base64
import PyPDF2
import os

# Configuration de la page
st.set_page_config(
    page_title="Quaterco", 
    layout="wide", 
    page_icon=":chart_with_upwards_trend:",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        height: 3em;
        width: 100%;
        border-radius: 5px;
        font-size: 16px;
    }
    .stFileUploader>div>div>button {
        background-color: #4CAF50;
        color: white;
    }
    .css-1aumxhk {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #e8f5e9;
    }
    h1, h2, h3 {
        color: #2e7d32;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre de l'application
st.title("📊 Quaterco - Tool for HSP and COSMO-RS data plotting")

# Navigation entre les modules
modules = {
    "Hansen Solubility Parameters": "hansen",
    "Ternary Plot Diagram": "ternary",
    "Quaternary Plot Diagram": "quaternary"
}

selected_module = st.sidebar.radio(
    "NAVIGATION", 
    list(modules.keys()),
    index=0
)

# Fonction pour télécharger un fichier
def get_file_download_link(file_data, filename, text):
    b64 = base64.b64encode(file_data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href

# ==============================================
# Module Paramètres de Solubilité de Hansen
# ==============================================
if modules[selected_module] == "hansen":
    st.header("🧪 Hansen Solubility Parameters")
    
    with st.expander("ℹ️ Instructions"):
        st.write("""
        Upload the excel file containing your data. The file must contain the following columns:
        δD, δP, δH, Compounds, Type (by default the color are 0 = Red = petro-sourced or unsafe compound, 1 = Green = bio-sourced compound, 2 = Blue = simulated HSP from COSMOQuick or other prediction software), CAS, R0.
        """)
    
uploaded_file = st.file_uploader("Upload your Excel file containing your data", type=["xlsx"])
    
if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            required_columns = ['δD', 'δP', 'δH', 'Type', 'Compounds', 'CAS', 'R0']
            
            if not all(col in data.columns for col in required_columns):
                st.error(f"Colonnes requises manquantes: {', '.join(required_columns)}")
            else:
                # Préparation des données
                x = data['δD'].astype(float)
                y = data['δP'].astype(float)
                z = data['δH'].astype(float)
                names = data['Compounds']
                types = data['Type']
                CAS = data['CAS']
                radii = data['R0'].astype(float)
                
                # Couleurs et légendes
                colors = ['red', 'green', 'blue']
                type_labels = ['Non-Green', 'Green', 'Simulated']
                
                # Bouton ON/OFF pour les noms des composés
                show_names = st.toggle("Display compound names", value=False)
                
                # Création du graphique 3D
                fig = go.Figure()
                
                # Ajout des points avec des couleurs selon le type
                for i in range(len(x)):
                    fig.add_trace(go.Scatter3d(
                        x=[x[i]], y=[y[i]], z=[z[i]],
                        mode='markers' + ('+text' if show_names else ''),
                        marker=dict(
                            size=6,
                            color=colors[types[i]],
                            opacity=0.8
                        ),
                        text=names[i] if show_names else None,
                        textposition="top center",
                        name=names[i],
                        hoverinfo='text',
                        hovertext=f"""
                        <b>{names[i]}</b><br>
                        CAS: {CAS[i]}<br>
                        δD: {x[i]:.2f}<br>
                        δP: {y[i]:.2f}<br>
                        δH: {z[i]:.2f}<br>
                        R0: {radii[i]:.2f}
                        """,
                        showlegend=False
                    ))
                
                # Mise en forme du graphique
                fig.update_layout(
                    scene=dict(
                        xaxis_title='δD',
                        yaxis_title='δP',
                        zaxis_title='δH',
                        xaxis=dict(gridcolor='lightgray', backgroundcolor='rgba(0,0,0,0)'),
                        yaxis=dict(gridcolor='lightgray', backgroundcolor='rgba(0,0,0,0)'),
                        zaxis=dict(gridcolor='lightgray', backgroundcolor='rgba(0,0,0,0)'),
                    ),
                    margin=dict(l=0, r=0, b=0, t=30),
                    height=700,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Ajout de la légende
                for i, label in enumerate(type_labels):
                    fig.add_trace(go.Scatter3d(
                        x=[None], y=[None], z=[None],
                        mode='markers',
                        marker=dict(size=10, color=colors[i]),
                        name=label,
                        showlegend=True
                    ))
                
                # Affichage du graphique
                st.plotly_chart(fig, use_container_width=True)
                
                # Section d'analyse interactive
                st.subheader("🔍 Interactive Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Recherche de composé
                    compound_name = st.selectbox(
                        "Search compound", 
                        [""] + sorted(names.unique()),
                        index=0
                    )
                    
                    if compound_name:
                        idx = names[names == compound_name].index[0]
                        st.success(f"Selected compound: {compound_name}")
                        
                        # Affichage des informations
                        st.write(f"**CAS:** {CAS[idx]}")
                        st.write(f"**Coordinates:** δD={x[idx]:.2f}, δP={y[idx]:.2f}, δH={z[idx]:.2f}")
                        st.write(f"**Hansen Sphere Radius (R0):** {radii[idx]:.2f}")
                        
                        # Bouton pour ouvrir dans PubChem
                        if st.button(f"🔎 Search {compound_name} on PubChem"):
                            url = f"https://pubchem.ncbi.nlm.nih.gov/#query={CAS[idx]}"
                            webbrowser.open_new_tab(url)
                
                with col2:
                    # Visualisation de la sphère d'interaction
                    show_sphere = st.checkbox("Display Hansen Sphere")
                    
                    if show_sphere and compound_name:
                        idx = names[names == compound_name].index[0]
                        center = [x[idx], y[idx], z[idx]]
                        radius = radii[idx]
                        
                        # Création de la sphère
                        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                        sphere_x = center[0] + radius * np.cos(u) * np.sin(v)
                        sphere_y = center[1] + radius * np.sin(u) * np.sin(v)
                        sphere_z = center[2] + radius * np.cos(v)
                        
                        # Ajout de la sphère au graphique
                        fig.add_trace(go.Surface(
                            x=sphere_x,
                            y=sphere_y,
                            z=sphere_z,
                            colorscale=[[0, 'rgba(255, 255, 0, 0.1)'], [1, 'rgba(255, 255, 0, 0.1)']],
                            showscale=False,
                            name=f"Hansen Sphere: {compound_name}"
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Options d'export
                st.subheader("📤 Export data")
                export_format = st.radio("Export format", ["PNG", "HTML", "CSV"])
                
                if st.button("Export the results"):
                    if export_format == "PNG":
                        img_bytes = fig.to_image(format="png")
                        st.download_button(
                            label="Download PNG",
                            data=img_bytes,
                            file_name="hansen_parameters.png",
                            mime="image/png"
                        )
                    elif export_format == "HTML":
                        html = fig.to_html()
                        st.download_button(
                            label="Download HTML",
                            data=html,
                            file_name="hansen_parameters.html",
                            mime="text/html"
                        )
                    elif export_format == "CSV":
                        csv = data.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="hansen_data.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# ==============================================
# Module Diagramme de Phase Ternaire
# ==============================================
elif modules[selected_module] == "ternary":
    st.header("📐 Ternary Phase Diagram")

    with st.expander("ℹ️ Instructions"):
        st.write("""
        Upload the excel file containing your data. The file must contain the following columns:
        V1, V2, V1', V2', names of solvents must be in cells: H2, I2 and J2
        """)
    
    uploaded_file = st.file_uploader("Upload your excel file", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            required_columns = ['V1', 'V2', "V1'", "V2'"]
            
            if not all(col in data.columns for col in required_columns):
                st.error(f"Missing columns: {', '.join(required_columns)}")
            else:
                # Préparation des données
                v1 = data['V1']
                v2 = data['V2']
                v1_prime = data["V1'"]
                v2_prime = data["V2'"]
                
                # Création de la figure matplotlib
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # Dessin du triangle rectangle
                ax.plot([0, 1, 0, 0], [0, 0, 1, 0], 'k-', linewidth=2)
                
                # Ajout des lignes et points
                for i in range(len(v1)):
                    color = np.random.rand(3,)
                    ax.plot([v1[i], v1_prime[i]], [v2[i], v2_prime[i]], 
                            color=color, marker='o', markersize=6, linewidth=2)
                
                # Configuration des axes
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect('equal', 'box')
                
                # Labels des axes
                ax.set_xlabel(data.iloc[0, 7] if len(data.columns) > 7 else 'Solvant 1', fontsize=12)
                ax.set_ylabel(data.iloc[0, 8] if len(data.columns) > 8 else 'Solvant 2', fontsize=12)
                
                # Ajout de la grille
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_xticks(np.arange(0, 1.1, 0.1))
                ax.set_yticks(np.arange(0, 1.1, 0.1))
                
                # Titre
                title = f"{data.iloc[0, 7] if len(data.columns) > 7 else 'Solvant 1'} / {data.iloc[0, 8] if len(data.columns) > 8 else 'Solvant 2'} / {data.iloc[0, 9] if len(data.columns) > 9 else 'Solvant 3'}"
                ax.set_title(title, fontsize=14, pad=20)
                
                # Affichage dans Streamlit
                st.pyplot(fig)
                
                # Options d'export
                st.subheader("📤 Export Options")
                export_format = st.selectbox("Export format", ["PNG", "PDF", "SVG"])
                
                if st.button("Export the diagram"):
                    buf = BytesIO()
                    if export_format == "PNG":
                        fig.savefig(buf, format="png", dpi=300)
                        st.download_button(
                            label="Download PNG",
                            data=buf.getvalue(),
                            file_name="ternary_diagram.png",
                            mime="image/png"
                        )
                    elif export_format == "PDF":
                        fig.savefig(buf, format="pdf")
                        st.download_button(
                            label="Download PDF",
                            data=buf.getvalue(),
                            file_name="ternary_diagram.pdf",
                            mime="application/pdf"
                        )
                    elif export_format == "SVG":
                        fig.savefig(buf, format="svg")
                        st.download_button(
                            label="Download SVG",
                            data=buf.getvalue(),
                            file_name="ternary_diagram.svg",
                            mime="image/svg+xml"
                        )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
# ==============================================
# Module Diagramme de Phase Quaternaire
# ==============================================
elif modules[selected_module] == "quaternary":
    st.header("🧊 Quaternary Phase Diagram")
    
    with st.expander("ℹ️ Instructions"):
        st.write("""
        1. Téléversez un fichier Excel avec les colonnes: V1, V2, V3, V1', V2', V3'
        2. Le diagramme quaternaire 3D sera généré automatiquement
        3. Utilisez les outils pour explorer la visualisation
        """)
    
    uploaded_file = st.file_uploader("Téléversez votre fichier Excel", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            required_columns = ['V1', 'V2', 'V3', "V1'", "V2'", "V3'"]
            
            if not all(col in data.columns for col in required_columns):
                st.error(f"Colonnes requises manquantes: {', '.join(required_columns)}")
            else:
                # Préparation des données
                x = data['V1']
                y = data['V2']
                z = data['V3']
                w = 1 - x - y - z
                x_prime = data["V1'"]
                y_prime = data["V2'"]
                z_prime = data["V3'"]
                w_prime = 1 - x_prime - y_prime - z_prime
                
                # Création du graphique 3D
                fig = go.Figure()
                
                # Ajout des lignes
                for i in range(len(x)):
                    color = f'rgb({np.random.randint(50,200)},{np.random.randint(50,200)},{np.random.randint(50,200)})'
                    fig.add_trace(go.Scatter3d(
                        x=[x[i], x_prime[i]],
                        y=[y[i], y_prime[i]],
                        z=[z[i], z_prime[i]],
                        w=[w[i], w_prime[i]],
                        mode='lines+markers',
                        line=dict(width=4, color=color),
                        marker=dict(size=5, color=color),
                        name=f"Ligne {i+1}",
                        hoverinfo='text',
                        text=f"Ligne {i+1}: ({x[i]:.2f}, {y[i]:.2f}, {z[i]:.2f}, {w[i]:.2f}) → ({x_prime[i]:.2f}, {y_prime[i]:.2f}, {z_prime[i]:.2f}, {w_prime[i]:.2f})"
                    ))
                
                # Ajout de la pyramide
                pyramid_vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
                pyramid_edges = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
                
                for edge in pyramid_edges:
                    fig.add_trace(go.Scatter3d(
                        x=[pyramid_vertices[edge[0]][0], pyramid_vertices[edge[1]][0]],
                        y=[pyramid_vertices[edge[0]][1], pyramid_vertices[edge[1]][1]],
                        z=[pyramid_vertices[edge[0]][2], pyramid_vertices[edge[1]][2]],
                        mode='lines',
                        line=dict(color='black', width=3),
                        showlegend=False,
                        hoverinfo='none'
                    ))
                
                # Mise en forme
                fig.update_layout(
                    scene=dict(
                        xaxis_title=data.iloc[0, 9] if len(data.columns) > 9 else 'V1',
                        yaxis_title=data.iloc[0, 10] if len(data.columns) > 10 else 'V2',
                        zaxis_title=data.iloc[0, 11] if len(data.columns) > 11 else 'V3',
                        xaxis=dict(gridcolor='lightgray', backgroundcolor='rgba(0,0,0,0)'),
                        yaxis=dict(gridcolor='lightgray', backgroundcolor='rgba(0,0,0,0)'),
                        zaxis=dict(gridcolor='lightgray', backgroundcolor='rgba(0,0,0,0)'),
                    ),
                    margin=dict(l=0, r=0, b=0, t=30),
                    height=700,
                    title={
                        'text': f"{data.iloc[0, 9] if len(data.columns) > 9 else 'V1'} / {data.iloc[0, 10] if len(data.columns) > 10 else 'V2'} / {data.iloc[0, 11] if len(data.columns) > 11 else 'V3'} / {data.iloc[0, 12] if len(data.columns) > 12 else 'V4'}",
                        'x': 0.5,
                        'font': {'size': 16}
                    }
                )
                
                # Affichage du graphique
                st.plotly_chart(fig, use_container_width=True)
                
                # Options avancées
                st.subheader("🎚 Contrôles 3D")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Animation de rotation
                    if st.button("🔄 Lancer l'animation de rotation"):
                        fig.update_layout(
                            scene=dict(
                                camera=dict(
                                    up=dict(x=0, y=0, z=1),
                                    center=dict(x=0, y=0, z=0),
                                    eye=dict(x=1.25, y=1.25, z=1.25)
                                )
                            ),
                            updatemenus=[dict(
                                type="buttons",
                                buttons=[dict(
                                    label="▶️",
                                    method="animate",
                                    args=[None, {"frame": {"duration": 50, "redraw": True}}]
                                )]
                        )])
                        
                        frames = []
                        for angle in range(0, 360, 5):
                            frames.append(go.Frame(
                                layout=dict(
                                    scene_camera=dict(
                                        eye=dict(
                                            x=1.25 * np.cos(np.radians(angle)),
                                            y=1.25 * np.sin(np.radians(angle)),
                                            z=1.25
                                        )
                                    )
                                )
                            ))
                        
                        fig.frames = frames
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Export des données
                    st.write("**Exporter le diagramme:**")
                    export_format = st.radio("Format", ["HTML", "PNG"], key="quat_export", horizontal=True)
                    
                    if st.button("Générer l'export", key="quat_export_btn"):
                        if export_format == "HTML":
                            html = fig.to_html()
                            st.download_button(
                                label="Télécharger HTML",
                                data=html,
                                file_name="quaternary_diagram.html",
                                mime="text/html"
                            )
                        elif export_format == "PNG":
                            img_bytes = fig.to_image(format="png")
                            st.download_button(
                                label="Télécharger PNG",
                                data=img_bytes,
                                file_name="quaternary_diagram.png",
                                mime="image/png"
                            )
        
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier: {str(e)}")

# Pied de page
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.9em; padding: 20px;">
    <p>Quaterco - Plateforme d'Analyze Scientifique | Développé avec Streamlit, Plotly et Pandas</p>
    <p>© 2023 Tous droits réservés</p>
    </div>
    """, unsafe_allow_html=True)
