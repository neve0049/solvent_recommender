import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
st.title("📊 Quaterco - Plateforme d'Analyze Scientifique")

# Navigation entre les modules
modules = {
    "Paramètres de Solubilité de Hansen": "hansen",
    "Diagramme de Phase Ternaire": "ternary",
    "Diagramme de Phase Quaternaire": "quaternary"
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
    st.header("🧪 Paramètres de Solubilité de Hansen 3D")
    
    with st.expander("ℹ️ Instructions"):
        st.write("""
        1. Téléversez un fichier Excel avec les colonnes: δD, δP, δH, Type, Compounds, CAS, R0
        2. Explorez la visualisation 3D interactive
        3. Utilisez les outils pour analyser les composés
        """)
    
    uploaded_file = st.file_uploader("Téléversez votre fichier Excel", type=["xlsx"])
    
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
                
                # Création du graphique 3D
                fig = go.Figure()
                
                # Ajout des points avec des couleurs selon le type
                for i in range(len(x)):
                    fig.add_trace(go.Scatter3d(
                        x=[x[i]], y=[y[i]], z=[z[i]],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=colors[types[i]],
                            opacity=0.8
                        ),
                        name=names[i],
                        text=f"""
                        <b>{names[i]}</b><br>
                        CAS: {CAS[i]}<br>
                        δD: {x[i]:.2f}<br>
                        δP: {y[i]:.2f}<br>
                        δH: {z[i]:.2f}<br>
                        R0: {radii[i]:.2f}
                        """,
                        hoverinfo='text',
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
                st.subheader("🔍 Analyse Interactive")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Recherche de composé
                    compound_name = st.selectbox(
                        "Rechercher un composé", 
                        [""] + sorted(names.unique()),
                        index=0
                    )
                    
                    if compound_name:
                        idx = names[names == compound_name].index[0]
                        st.success(f"Composé sélectionné: {compound_name}")
                        
                        # Affichage des informations
                        st.write(f"**CAS:** {CAS[idx]}")
                        st.write(f"**Coordonnées:** δD={x[idx]:.2f}, δP={y[idx]:.2f}, δH={z[idx]:.2f}")
                        st.write(f"**Rayon d'interaction (R0):** {radii[idx]:.2f}")
                        
                        # Bouton pour ouvrir dans PubChem
                        if st.button(f"🔎 Rechercher {compound_name} sur PubChem"):
                            url = f"https://pubchem.ncbi.nlm.nih.gov/#query={CAS[idx]}"
                            webbrowser.open_new_tab(url)
                
                with col2:
                    # Visualisation de la sphère d'interaction
                    show_sphere = st.checkbox("Afficher la sphère d'interaction")
                    
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
                            name=f"Sphère d'interaction: {compound_name}"
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Options d'export
                st.subheader("📤 Export des Données")
                export_format = st.radio("Format d'export", ["PNG", "HTML", "CSV"])
                
                if st.button("Exporter les résultats"):
                    if export_format == "PNG":
                        img_bytes = fig.to_image(format="png")
                        st.download_button(
                            label="Télécharger PNG",
                            data=img_bytes,
                            file_name="hansen_parameters.png",
                            mime="image/png"
                        )
                    elif export_format == "HTML":
                        html = fig.to_html()
                        st.download_button(
                            label="Télécharger HTML",
                            data=html,
                            file_name="hansen_parameters.html",
                            mime="text/html"
                        )
                    elif export_format == "CSV":
                        csv = data.to_csv(index=False)
                        st.download_button(
                            label="Télécharger CSV",
                            data=csv,
                            file_name="hansen_data.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier: {str(e)}")

# ==============================================
# Module Diagramme de Phase Ternaire
# ==============================================
elif modules[selected_module] == "ternary":
    st.header("📐 Diagramme de Phase Ternaire Interactif")
    
    with st.expander("ℹ️ Instructions"):
        st.write("""
        1. Téléversez un fichier Excel avec les colonnes: V1, V2, V1', V2'
        2. Le diagramme ternaire interactif sera généré automatiquement
        3. Utilisez les outils pour explorer les données
        """)
    
    uploaded_file = st.file_uploader("Téléversez votre fichier Excel", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            required_columns = ['V1', 'V2', "V1'", "V2'"]
            
            if not all(col in data.columns for col in required_columns):
                st.error(f"Colonnes requises manquantes: {', '.join(required_columns)}")
            else:
                # Préparation des données
                x = data['V1']
                y = data['V2']
                x_prime = data["V1'"]
                y_prime = data["V2'"]
                
                # Conversion en coordonnées ternaires
                def to_ternary(x, y):
                    a = x
                    b = y
                    c = 1 - x - y
                    return a, b, c
                
                # Création du diagramme ternaire
                fig = go.Figure()
                
                # Ajout du triangle de base
                fig.add_trace(go.Scatterternary({
                    'mode': 'lines',
                    'a': [1, 0, 0, 1],
                    'b': [0, 1, 0, 0],
                    'c': [0, 0, 1, 0],
                    'line': {'color': 'black', 'width': 2},
                    'hoverinfo': 'none',
                    'showlegend': False
                }))
                
                # Ajout des lignes et points
                for i in range(len(x)):
                    a1, b1, c1 = to_ternary(x[i], y[i])
                    a2, b2, c2 = to_ternary(x_prime[i], y_prime[i])
                    
                    color = f'rgb({np.random.randint(50,200)},{np.random.randint(50,200)},{np.random.randint(50,200)})'
                    
                    fig.add_trace(go.Scatterternary({
                        'mode': 'lines+markers',
                        'a': [a1, a2],
                        'b': [b1, b2],
                        'c': [c1, c2],
                        'line': {'width': 2, 'color': color},
                        'marker': {'size': 8, 'color': color},
                        'hoverinfo': 'text',
                        'text': f"Ligne {i+1}: ({x[i]:.2f}, {y[i]:.2f}) → ({x_prime[i]:.2f}, {y_prime[i]:.2f})",
                        'showlegend': False
                    }))
                
                # Mise en forme
                axis_title_size = 14
                fig.update_layout({
                    'ternary': {
                        'sum': 1,
                        'aaxis': {
                            'title': {'text': data.iloc[0, 7] if len(data.columns) > 7 else 'A', 'font': {'size': axis_title_size}},
                            'min': 0.01, 'linewidth': 2, 'ticks': 'outside',
                            'tickvals': np.arange(0, 1.1, 0.1), 'tickformat': '.0%',
                            'gridcolor': 'lightgray', 'minorgridcount': 4
                        },
                        'baxis': {
                            'title': {'text': data.iloc[0, 8] if len(data.columns) > 8 else 'B', 'font': {'size': axis_title_size}},
                            'min': 0.01, 'linewidth': 2, 'ticks': 'outside',
                            'tickvals': np.arange(0, 1.1, 0.1), 'tickformat': '.0%',
                            'gridcolor': 'lightgray', 'minorgridcount': 4
                        },
                        'caxis': {
                            'title': {'text': data.iloc[0, 9] if len(data.columns) > 9 else 'C', 'font': {'size': axis_title_size}},
                            'min': 0.01, 'linewidth': 2, 'ticks': 'outside',
                            'tickvals': np.arange(0, 1.1, 0.1), 'tickformat': '.0%',
                            'gridcolor': 'lightgray', 'minorgridcount': 4
                        }
                    },
                    'showlegend': False,
                    'height': 700,
                    'title': {
                        'text': f"{data.iloc[0, 7] if len(data.columns) > 7 else 'A'} / {data.iloc[0, 8] if len(data.columns) > 8 else 'B'} / {data.iloc[0, 9] if len(data.columns) > 9 else 'C'}",
                        'x': 0.5,
                        'font': {'size': 16}
                    },
                    'hovermode': 'closest',
                    'margin': {'t': 60}
                })
                
                # Affichage du graphique
                st.plotly_chart(fig, use_container_width=True)
                
                # Options interactives
                st.subheader("🛠 Options Avancées")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Affichage des labels
                    show_labels = st.checkbox("Afficher les étiquettes des points", value=False)
                    if show_labels:
                        for i in range(len(x)):
                            a, b, c = to_ternary(x_prime[i], y_prime[i])
                            fig.add_annotation(
                                x=a, y=b, text=f"P{i+1}",
                                showarrow=True, arrowhead=1, ax=0, ay=-20
                            )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Grille fine
                    grid_option = st.selectbox("Style de grille", ["Normale", "Fine", "Aucune"])
                    if grid_option == "Fine":
                        fig.update_layout({
                            'ternary': {
                                'aaxis': {'minorgridcount': 9},
                                'baxis': {'minorgridcount': 9},
                                'caxis': {'minorgridcount': 9}
                            }
                        })
                    elif grid_option == "Aucune":
                        fig.update_layout({
                            'ternary': {
                                'aaxis': {'gridcolor': 'rgba(0,0,0,0)', 'minorgridcolor': 'rgba(0,0,0,0)'},
                                'baxis': {'gridcolor': 'rgba(0,0,0,0)', 'minorgridcolor': 'rgba(0,0,0,0)'},
                                'caxis': {'gridcolor': 'rgba(0,0,0,0)', 'minorgridcolor': 'rgba(0,0,0,0)'}
                            }
                        })
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Export des données
                    st.write("**Exporter le diagramme:**")
                    export_format = st.radio("Format", ["HTML", "PNG", "SVG"], horizontal=True)
                    
                    if st.button("Générer l'export"):
                        if export_format == "HTML":
                            html = fig.to_html()
                            st.download_button(
                                label="Télécharger HTML",
                                data=html,
                                file_name="ternary_diagram.html",
                                mime="text/html"
                            )
                        elif export_format == "PNG":
                            img_bytes = fig.to_image(format="png")
                            st.download_button(
                                label="Télécharger PNG",
                                data=img_bytes,
                                file_name="ternary_diagram.png",
                                mime="image/png"
                            )
                        elif export_format == "SVG":
                            img_bytes = fig.to_image(format="svg")
                            st.download_button(
                                label="Télécharger SVG",
                                data=img_bytes,
                                file_name="ternary_diagram.svg",
                                mime="image/svg+xml"
                            )
        
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier: {str(e)}")

# ==============================================
# Module Diagramme de Phase Quaternaire
# ==============================================
elif modules[selected_module] == "quaternary":
    st.header("🧊 Diagramme de Phase Quaternaire 3D")
    
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
                x_prime = data["V1'"]
                y_prime = data["V2'"]
                z_prime = data["V3'"]
                
                # Création du graphique 3D
                fig = go.Figure()
                
                # Ajout des lignes
                for i in range(len(x)):
                    color = f'rgb({np.random.randint(50,200)},{np.random.randint(50,200)},{np.random.randint(50,200)})'
                    fig.add_trace(go.Scatter3d(
                        x=[x[i], x_prime[i]],
                        y=[y[i], y_prime[i]],
                        z=[z[i], z_prime[i]],
                        mode='lines+markers',
                        line=dict(width=4, color=color),
                        marker=dict(size=5, color=color),
                        name=f"Ligne {i+1}",
                        hoverinfo='text',
                        text=f"Ligne {i+1}: ({x[i]:.2f}, {y[i]:.2f}, {z[i]:.2f}) → ({x_prime[i]:.2f}, {y_prime[i]:.2f}, {z_prime[i]:.2f})"
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
                        )
                        
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
