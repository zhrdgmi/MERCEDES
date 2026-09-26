# -*- coding: utf-8 -*-
"""
Prestige Motors — Dashboard Concession & Après-Vente (style Mercedes-Benz)
============================================================================
Application Streamlit avec authentification, nettoyage de données,
KPIs, filtres dynamiques et visualisations (Matplotlib / Seaborn).

Auteur : généré avec Claude

Remarque : le logo utilisé est un emblème générique "Prestige Motors"
créé pour cette démo (et non le logo officiel déposé Mercedes-Benz).
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ----------------------------------------------------------------------------
# CONFIGURATION GÉNÉRALE DE LA PAGE
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Prestige Motors | Dashboard Concession",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(APP_DIR, "dataset.csv")
LOGO_PATH = os.path.join(APP_DIR, "logo.png")

# Identifiants de connexion (démo)
VALID_USERNAME = "zhrdgmi"
VALID_PASSWORD = "1234"

sns.set_theme(style="whitegrid")
MB_BLACK = "#121212"
MB_SILVER = "#C4C8CC"
MB_DARK_SILVER = "#8C9196"
MB_ACCENT = "#0B7FAB"   # bleu-argent discret pour distinguer certaines séries
PALETTE = [MB_BLACK, MB_SILVER, MB_ACCENT, "#4A4E52", "#D9DCDE", "#3C6E8F", "#6E7276"]
sns.set_palette(sns.color_palette(PALETTE))

# ----------------------------------------------------------------------------
# STYLE CSS PERSONNALISÉ — noir / argent / gris (style automobile premium)
# ----------------------------------------------------------------------------
CUSTOM_CSS = f"""
<style>
    .main {{
        background-color: #F4F4F5;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {MB_BLACK};
    }}
    section[data-testid="stSidebar"] * {{
        color: #EDEDED !important;
    }}
    div[data-testid="stMetric"] {{
        background-color: #FFFFFF;
        border: 1px solid #DADDE0;
        border-left: 6px solid {MB_BLACK};
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    div[data-testid="stMetricLabel"] {{
        color: {MB_BLACK};
        font-weight: 600;
    }}
    h1, h2, h3 {{
        color: {MB_BLACK};
    }}
    .prestige-header {{
        display: flex;
        align-items: center;
        gap: 18px;
        padding: 6px 0 18px 0;
    }}
    .prestige-badge {{
        background-color: {MB_BLACK};
        color: {MB_SILVER};
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}
    .login-card {{
        max-width: 420px;
        margin: 60px auto 0 auto;
        padding: 36px 34px 28px 34px;
        background: #FFFFFF;
        border-radius: 16px;
        border: 1px solid #DADDE0;
        box-shadow: 0 4px 18px rgba(0,0,0,0.10);
    }}
    .stButton>button {{
        background-color: {MB_BLACK};
        color: {MB_SILVER};
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 0.5em 1.2em;
    }}
    .stButton>button:hover {{
        background-color: #2A2A2A;
        color: white;
    }}
    div[data-testid="stTabs"] button {{
        font-weight: 600;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# AUTHENTIFICATION
# ----------------------------------------------------------------------------
def login_page():
    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        if os.path.exists(LOGO_PATH):
            lc1, lc2, lc3 = st.columns([1, 1, 1])
            with lc2:
                st.image(LOGO_PATH, width=110)
        st.markdown(
            "<h2 style='text-align:center;margin-top:0;'>PRESTIGE MOTORS</h2>"
            "<p style='text-align:center;color:#666;margin-top:-10px;'>"
            "Concession &amp; Après-Vente — Style Mercedes-Benz</p>",
            unsafe_allow_html=True,
        )
        st.write("")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Identifiant")
            password = st.text_input("Mot de passe", type="password")
            submitted = st.form_submit_button("Se connecter", use_container_width=True)

        if submitted:
            if username == VALID_USERNAME and password == VALID_PASSWORD:
                st.session_state["authenticated"] = True
                st.session_state["user"] = username
                st.rerun()
            else:
                st.error("Identifiants incorrects. Veuillez réessayer.")

        st.markdown(
            "<p style='text-align:center;color:#999;font-size:0.8rem;margin-top:18px;'>"
            "Accès réservé au personnel de la concession</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


def logout_button():
    with st.sidebar:
        st.write("---")
        if st.button("🚪 Se déconnecter", use_container_width=True):
            st.session_state["authenticated"] = False
            st.rerun()


# ----------------------------------------------------------------------------
# CHARGEMENT & NETTOYAGE DES DONNÉES
# ----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 1. Suppression des doublons
    df = df.drop_duplicates(subset=["transaction_id"]).reset_index(drop=True)

    # 2. Conversion des types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "prix_mad", "duree_reparation_h", "cout_reparation_mad",
        "satisfaction_client", "stock_disponible", "jours_en_stock",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3. Traitement des valeurs manquantes
    #    - satisfaction manquante -> imputée par la médiane du type de transaction
    df["satisfaction_client"] = df.groupby("type_transaction")["satisfaction_client"].transform(
        lambda s: s.fillna(s.median())
    )

    # 4. Suppression des lignes sans date / ville / type de transaction
    df = df.dropna(subset=["date", "ville", "type_transaction"]).reset_index(drop=True)

    # 5. Colonnes dérivées utiles au dashboard
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["weekday"] = df["date"].dt.day_name()
    df["is_vente"] = df["type_transaction"].eq("Vente")
    df["is_entretien"] = df["type_transaction"].eq("Entretien")

    return df


# ----------------------------------------------------------------------------
# EN-TÊTE & FILTRES
# ----------------------------------------------------------------------------
def render_header():
    c1, c2 = st.columns([0.12, 0.88])
    with c1:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=80)
    with c2:
        st.markdown(
            "<div class='prestige-header'>"
            "<div><h1 style='margin-bottom:0;'>Prestige Motors — Dashboard Concession &amp; Après-Vente</h1>"
            "<span class='prestige-badge'>RÉSEAU MULTI-VILLES — STYLE MERCEDES-BENZ</span></div>"
            "</div>",
            unsafe_allow_html=True,
        )


def sidebar_filters(df: pd.DataFrame):
    st.sidebar.markdown("## 🔎 Filtres")

    min_date, max_date = df["date"].min().date(), df["date"].max().date()
    date_range = st.sidebar.date_input(
        "Période",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    villes = sorted(df["ville"].dropna().unique().tolist())
    sel_villes = st.sidebar.multiselect("Ville / Showroom", villes, default=villes)

    modeles = sorted(df["modele"].dropna().unique().tolist())
    sel_modeles = st.sidebar.multiselect("Modèle", modeles, default=modeles)

    types_trans = sorted(df["type_transaction"].dropna().unique().tolist())
    sel_types = st.sidebar.multiselect("Type de service", types_trans, default=types_trans)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Connecté en tant que **{st.session_state.get('user','')}**")

    mask = (
        (df["date"].dt.date >= start_date)
        & (df["date"].dt.date <= end_date)
        & (df["ville"].isin(sel_villes))
        & (df["modele"].isin(sel_modeles))
        & (df["type_transaction"].isin(sel_types))
    )
    return df.loc[mask].copy()


# ----------------------------------------------------------------------------
# KPIs
# ----------------------------------------------------------------------------
def render_kpis(df: pd.DataFrame):
    st.subheader("📊 Indicateurs clés (KPIs)")
    if df.empty:
        st.warning("Aucune donnée pour les filtres sélectionnés.")
        return

    ventes = df[df["is_vente"]]
    entretien = df[df["is_entretien"]]

    chiffre_affaires = ventes["prix_mad"].sum()
    nb_ventes = len(ventes)
    satisfaction_moy = df["satisfaction_client"].mean()
    delai_moy_reparation = entretien["duree_reparation_h"].mean()
    cout_moy_reparation = entretien["cout_reparation_mad"].mean()

    # taux de rotation du stock (approximation) : 365 / jours moyens en stock
    jours_stock_moy = ventes["jours_en_stock"].mean()
    taux_rotation = 365 / jours_stock_moy if jours_stock_moy and jours_stock_moy > 0 else np.nan

    row1 = st.columns(4)
    row1[0].metric("Chiffre d'affaires", f"{chiffre_affaires:,.0f} MAD".replace(",", " "))
    row1[1].metric("Nombre de ventes", f"{nb_ventes:,}".replace(",", " "))
    row1[2].metric("Satisfaction moyenne", f"{satisfaction_moy:.2f} / 5")
    row1[3].metric("Interventions", f"{len(entretien):,}".replace(",", " "))

    row2 = st.columns(4)
    row2[0].metric("Délai moyen réparation", f"{delai_moy_reparation:.1f} h" if pd.notna(delai_moy_reparation) else "—")
    row2[1].metric("Coût moyen réparation", f"{cout_moy_reparation:,.0f} MAD".replace(",", " ") if pd.notna(cout_moy_reparation) else "—")
    row2[2].metric("Jours moyens en stock", f"{jours_stock_moy:.0f} j" if pd.notna(jours_stock_moy) else "—")
    row2[3].metric("Taux de rotation stock", f"{taux_rotation:.1f}x / an" if pd.notna(taux_rotation) else "—")


# ----------------------------------------------------------------------------
# GRAPHIQUES
# ----------------------------------------------------------------------------
def fig_monthly_revenue(ventes):
    monthly = ventes.groupby("month")["prix_mad"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(monthly["month"], monthly["prix_mad"], color=MB_BLACK, linewidth=2, marker="o", markersize=4)
    ax.fill_between(range(len(monthly)), monthly["prix_mad"], color=MB_SILVER, alpha=0.4)
    ax.set_title("Chiffre d'affaires mensuel (Ventes)")
    ax.set_xlabel("Mois")
    ax.set_ylabel("MAD")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def fig_sales_by_model(ventes):
    counts = ventes["modele"].value_counts().reset_index()
    counts.columns = ["modele", "count"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=counts, x="count", y="modele", ax=ax, color=MB_BLACK)
    ax.set_title("Nombre de ventes par modèle")
    ax.set_xlabel("Nombre de ventes")
    ax.set_ylabel("Modèle")
    fig.tight_layout()
    return fig


def fig_revenue_by_ville(ventes):
    rev = ventes.groupby("ville")["prix_mad"].sum().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(rev["ville"], rev["prix_mad"], color=MB_BLACK)
    ax.set_title("Chiffre d'affaires par ville / showroom")
    ax.set_ylabel("MAD")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def fig_sales_by_vendeur(ventes):
    top = ventes.groupby("vendeur")["prix_mad"].sum().sort_values(ascending=False).head(10).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=top, x="prix_mad", y="vendeur", ax=ax, color=MB_SILVER, edgecolor=MB_BLACK)
    ax.set_title("Top 10 vendeurs par chiffre d'affaires")
    ax.set_xlabel("MAD")
    ax.set_ylabel("Vendeur")
    fig.tight_layout()
    return fig


def fig_price_distribution(ventes):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.histplot(ventes["prix_mad"].dropna(), bins=30, ax=ax, color=MB_BLACK)
    ax.set_title("Distribution des prix de vente")
    ax.set_xlabel("Prix (MAD)")
    ax.set_ylabel("Nombre de ventes")
    fig.tight_layout()
    return fig


def fig_neuf_vs_occasion(ventes):
    counts = ventes["categorie_vehicule"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90,
           colors=[MB_BLACK, MB_SILVER])
    ax.set_title("Répartition des ventes Neuf / Occasion")
    ax.axis("equal")
    fig.tight_layout()
    return fig


def fig_panne_frequency(entretien):
    counts = entretien["type_panne"].value_counts().reset_index()
    counts.columns = ["type_panne", "count"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=counts, x="count", y="type_panne", ax=ax, color=MB_ACCENT)
    ax.set_title("Fréquence des types de panne / entretien")
    ax.set_xlabel("Nombre d'interventions")
    ax.set_ylabel("Type de panne")
    fig.tight_layout()
    return fig


def fig_repair_duration_by_panne(entretien):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.boxplot(data=entretien, x="duree_reparation_h", y="type_panne", ax=ax, color=MB_SILVER)
    ax.set_title("Durée de réparation par type de panne")
    ax.set_xlabel("Durée (heures)")
    ax.set_ylabel("Type de panne")
    fig.tight_layout()
    return fig


def fig_repair_cost_by_model(entretien):
    cost = entretien.groupby("modele")["cout_reparation_mad"].mean().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(cost["modele"], cost["cout_reparation_mad"], color=MB_ACCENT)
    ax.set_title("Coût moyen de réparation par modèle")
    ax.set_ylabel("MAD")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def fig_satisfaction_by_type(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="type_transaction", y="satisfaction_client", ax=ax,
                palette=[MB_BLACK, MB_SILVER])
    ax.set_title("Satisfaction client par type de service")
    ax.set_xlabel("")
    ax.set_ylabel("Satisfaction (1-5)")
    fig.tight_layout()
    return fig


def fig_satisfaction_by_ville(df):
    sat = df.groupby("ville")["satisfaction_client"].mean().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(sat["ville"], sat["satisfaction_client"], color=MB_BLACK)
    ax.set_title("Satisfaction moyenne par ville")
    ax.set_ylabel("Satisfaction (1-5)")
    ax.set_ylim(0, 5)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def fig_technician_performance(entretien):
    perf = entretien.groupby("technicien").agg(
        nb_interventions=("transaction_id", "count"),
        duree_moyenne=("duree_reparation_h", "mean"),
    ).sort_values("nb_interventions", ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=perf, x="nb_interventions", y="technicien", ax=ax, color=MB_ACCENT)
    ax.set_title("Nombre d'interventions par technicien")
    ax.set_xlabel("Nombre d'interventions")
    ax.set_ylabel("Technicien")
    fig.tight_layout()
    return fig


def fig_stock_by_model(ventes):
    stock = ventes.groupby("modele")["stock_disponible"].mean().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(stock["modele"], stock["stock_disponible"], color=MB_SILVER, edgecolor=MB_BLACK)
    ax.set_title("Stock disponible moyen par modèle")
    ax.set_ylabel("Unités en stock")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def fig_corr_heatmap(df):
    numeric_cols = ["prix_mad", "duree_reparation_h", "cout_reparation_mad",
                     "satisfaction_client", "stock_disponible", "jours_en_stock"]
    corr = df[numeric_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Greys", ax=ax)
    ax.set_title("Matrice de corrélation")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# DASHBOARD
# ----------------------------------------------------------------------------
def render_dashboard(df_full: pd.DataFrame):
    render_header()
    df = sidebar_filters(df_full)
    logout_button()

    render_kpis(df)
    st.write("")

    ventes = df[df["is_vente"]]
    entretien = df[df["is_entretien"]]

    tabs = st.tabs([
        "💰 Ventes", "🔧 Entretien & Réparations", "⭐ Satisfaction",
        "📦 Stock", "🧾 Données"
    ])

    with tabs[0]:
        if not ventes.empty:
            st.pyplot(fig_monthly_revenue(ventes))
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(fig_sales_by_model(ventes))
            with c2:
                st.pyplot(fig_revenue_by_ville(ventes))
            c3, c4 = st.columns(2)
            with c3:
                st.pyplot(fig_sales_by_vendeur(ventes))
            with c4:
                st.pyplot(fig_neuf_vs_occasion(ventes))
            st.pyplot(fig_price_distribution(ventes))
        else:
            st.info("Aucune vente pour les filtres sélectionnés.")

    with tabs[1]:
        if not entretien.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(fig_panne_frequency(entretien))
            with c2:
                st.pyplot(fig_technician_performance(entretien))
            st.pyplot(fig_repair_duration_by_panne(entretien))
            st.pyplot(fig_repair_cost_by_model(entretien))
        else:
            st.info("Aucune intervention pour les filtres sélectionnés.")

    with tabs[2]:
        if not df.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(fig_satisfaction_by_type(df))
            with c2:
                st.pyplot(fig_satisfaction_by_ville(df))
            st.pyplot(fig_corr_heatmap(df))

    with tabs[3]:
        if not ventes.empty:
            st.pyplot(fig_stock_by_model(ventes))
            st.caption(
                "Le stock et les jours en stock sont estimés au moment de chaque vente "
                "(snapshot par modèle / showroom)."
            )
        else:
            st.info("Pas de données de stock pour les filtres sélectionnés (filtrez sur 'Vente').")

    with tabs[4]:
        st.markdown("#### Aperçu des données filtrées")
        st.dataframe(df.head(200), use_container_width=True)
        st.download_button(
            "⬇️ Télécharger les données filtrées (CSV)",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="prestige_motors_donnees_filtrees.csv",
            mime="text/csv",
        )


# ----------------------------------------------------------------------------
# POINT D'ENTRÉE
# ----------------------------------------------------------------------------
def main():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        login_page()
        return

    if not os.path.exists(DATA_PATH):
        st.error(f"Fichier de données introuvable : {DATA_PATH}")
        return

    df_full = load_and_clean_data(DATA_PATH)
    render_dashboard(df_full)


if __name__ == "__main__":
    main()
