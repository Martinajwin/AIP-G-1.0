# ==========================================================
# 🏷️ AIP-G 1.0
# ==========================================================

# ======================================
# 🔍 Streamlit App: Two-Stage ML Prediction Pipeline with Mordred Descriptors & AD
# ======================================

import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
import joblib
from graphviz import Digraph

# -----------------------------
# ⚙️ Mahalanobis Distance (dimension-safe)
# -----------------------------
def mahalanobis_distance(X, mean_vec, cov_inv):
    n_features = X.shape[1]
    m_features = mean_vec.shape[0]
    if n_features != m_features:
        mean_vec = mean_vec[:n_features]
        cov_inv = cov_inv[:n_features, :n_features]
    diffs = X - mean_vec
    return np.sqrt(np.sum(diffs @ cov_inv * diffs, axis=1))

# -----------------------------
# ✅ Label Normalization
# -----------------------------
def normalize_label(lbl):
    lbl = str(lbl).strip().lower()
    if "high" in lbl:
        return "HighlyActive"
    elif "active" in lbl and "inactive" not in lbl:
        return "Active"
    elif "inactive" in lbl:
        return "Inactive"
    else:
        return lbl.capitalize()

# -----------------------------
# ⚙️ Load models & AD params
# -----------------------------
def load_ad_params(model_name):
    mean_vec = np.load(f"models/mean_{model_name}.npy")
    cov_inv = np.load(f"models/covinv_{model_name}.npy")
    ad_cutoff = np.load(f"models/adcutoff_{model_name}.npy")
    return mean_vec, cov_inv, ad_cutoff

# Load classifiers
rf1 = joblib.load("models/RF_ActvInact.pkl")
et1 = joblib.load("models/ET_ActvInact.pkl")
rf2 = joblib.load("models/RF_HactAct.pkl")
et2 = joblib.load("models/ET_HactAct.pkl")

# -----------------------------
# 🧩 Model Feature Lists
# -----------------------------
stage1_rf_features = ['Xch-6d','nAromAtom','SMR_VSA9','PEOE_VSA2','Xch-5d','nHRing','NssNH',
                      'EState_VSA8','FilterItLogS','NaaNH','SlogP_VSA1','C3SP3','C2SP2',
                      'ATSC5dv','PEOE_VSA12','n9FRing','ECIndex','SlogP_VSA3','AATS0v','PEOE_VSA3']
stage1_et_features = ['nHRing','Xch-6d','SMR_VSA9','nHBDon','nAromAtom','NaaNH','n9FRing',
                      'NaaaC','PEOE_VSA8','NsssCH','PEOE_VSA12','SlogP_VSA1','C2SP2',
                      'NaasN','SlogP_VSA10','nBondsD','SlogP_VSA3','VSA_EState3','nHetero','PEOE_VSA3']
stage2_rf_features = ['PEOE_VSA8','nAromAtom','PEOE_VSA13','ATSC5dv','Xch-6d','Xch-5d','NaasC',
                      'SlogP_VSA10','nBondsD','ATSC5v','Diameter','nN','ATSC6Z','PEOE_VSA7',
                      'C3SP2','ATSC2v','ATSC7c','ATSC8i','FCSP3','ATSC3Z']
stage2_et_features = ['PEOE_VSA12','C1SP2','PEOE_VSA13','n9FRing','SMR_VSA4','NaaaC','SlogP_VSA3',
                      'SlogP_VSA4','C2SP2','Xc-5d','Xch-5d','Xch-6d','PEOE_VSA7','nHBDon',
                      'ATSC5dv','AATSC0p','SMR_VSA9','ATSC8v','PEOE_VSA4']
# -----------------------------
# ⚗️ Streamlit UI
# -----------------------------
st.set_page_config(page_title="AIP-G 1.0", layout="wide")

st.title("Alzheimer’s disease, Inhibitor Predictor for GSK-3β (1.0)")  
st.subheader("(AIP-G 1.0)")
tabs = st.tabs(["1️⃣ Molecule Screening", "2️⃣ Methodology", "3️⃣ Model Performance", "4️⃣ References and Citation"])
tab1, tab2, tab3, tab4 = tabs

# ==========================================================
# 1️⃣ SCREENING TAB
# ==========================================================
with tab1:
    st.title("Predict GSK-3β inhibitors for Alzheimer’s Disease")

    input_option = st.radio("Input Type:", ["Enter SMILES manually", "Upload CSV"])
    smiles_list = []

    if input_option == "Upload CSV":
        st.write("CSV must contain column heading 'SMILES' in the first column.")
        uploaded_file = st.file_uploader("Upload CSV with SMILES", type=["csv"])
        if uploaded_file is not None:
            df_input = pd.read_csv(uploaded_file)
            if "SMILES" not in df_input.columns:
                st.error("CSV must contain a 'SMILES' column.")
                st.stop()
            smiles_list = [s for s in df_input["SMILES"] if isinstance(s, str)]
    else:
        st.write("A minimum of 10 SMILES is recomended.")
        user_smiles = st.text_area("Enter SMILES (one per line)")
        smiles_list = [s.strip() for s in user_smiles.split("\n") if s.strip()]

    if st.button("🚀 Predict") and smiles_list:
        st.info("Computing Descriptors... please wait ⏳")

        canonical_smiles, mols = [], []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
                mols.append(mol)
            else:
                st.warning(f"Invalid SMILES skipped: {smi}")

        if len(mols) == 0:
            st.error("No valid SMILES found. Please check your input.")
            st.stop()

        # 🧮 Compute Mordred descriptors
        calc = Calculator(descriptors, ignore_3D=True)
        df_desc = calc.pandas(mols)

        # 🧼 Clean descriptors
        df_desc = df_desc.replace([np.inf, -np.inf, "Error", "error"], np.nan)
        df_desc = df_desc.apply(pd.to_numeric, errors="coerce")
        df_desc = df_desc.fillna(df_desc.mean(numeric_only=True))

        # ✅ Safe feature alignment
        def prepare_features_safe(df, model):
            """
            Align descriptor DataFrame to match features used during model training.
            Any missing features are added with zeros, and unexpected ones are dropped.
            """
            # Get the exact list of features seen by the model
            model_features = getattr(model, "feature_names_in_", None)

            if model_features is None:
                raise ValueError(
                    "Model does not have 'feature_names_in_' attribute. "
                    "Please retrain with scikit-learn >=1.0."
                )

            # Ensure all expected columns exist
            for f in model_features:
                if f not in df.columns:
                    df[f] = 0.0

            # Drop unexpected columns
            df = df.loc[:, model_features]

            # Convert all to numeric and handle missing
            df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

            return df


        # ==========================================================
        # 🔹 Stage 1 Prediction
        # ==========================================================
        X1_rf = prepare_features_safe(df_desc, rf1)
        X1_et = prepare_features_safe(df_desc, et1)


        mean_rf1, covinv_rf1, adcut_rf1 = load_ad_params("RF_ActvInact")
        mean_et1, covinv_et1, adcut_et1 = load_ad_params("ET_ActvInact")

        md_rf1 = mahalanobis_distance(X1_rf, mean_rf1, covinv_rf1)
        md_et1 = mahalanobis_distance(X1_et, mean_et1, covinv_et1)

        ad_rf1 = ["Within" if d <= adcut_rf1 else "Outside" for d in md_rf1]
        ad_et1 = ["Within" if d <= adcut_et1 else "Outside" for d in md_et1]

        pred_rf1 = [normalize_label(p) for p in rf1.predict(X1_rf)]
        pred_et1 = [normalize_label(p) for p in et1.predict(X1_et)]
        proba_rf1 = rf1.predict_proba(X1_rf)[:, 1]
        proba_et1 = et1.predict_proba(X1_et)[:, 1]

        final_stage1 = []
        for i in range(len(canonical_smiles)):
            prf, pet = pred_rf1[i], pred_et1[i]
            arf, aet = ad_rf1[i], ad_et1[i]
            prb_rf, prb_et = proba_rf1[i], proba_et1[i]
            if prf == pet:
                final = prf
            elif arf == "Within" and aet == "Outside":
                final = prf
            elif aet == "Within" and arf == "Outside":
                final = pet
            else:
                final = prf if prb_rf >= prb_et else pet
            final_stage1.append(final)

        stage1_df = pd.DataFrame({
            "SMILES": canonical_smiles,
            "Stage1_RF": pred_rf1,
            "Stage1_ET": pred_et1,
            "Stage1_Consensus": final_stage1
        })

        # ==========================================================
        # 🔹 Stage 2 Prediction (Highly Active vs Active)
        # ==========================================================
        active_mask = np.array(final_stage1) == "Active"

        if active_mask.any():
            active_df = df_desc.loc[active_mask].reset_index(drop=True)
            active_smiles = [s for i, s in enumerate(canonical_smiles) if active_mask[i]]

            X2_rf = prepare_features_safe(active_df, rf2)
            X2_et = prepare_features_safe(active_df, et2)

            mean_rf2, covinv_rf2, adcut_rf2 = load_ad_params("RF_HactAct")
            mean_et2, covinv_et2, adcut_et2 = load_ad_params("ET_HactAct")

            md_rf2 = mahalanobis_distance(X2_rf, mean_rf2, covinv_rf2)
            md_et2 = mahalanobis_distance(X2_et, mean_et2, covinv_et2)

            ad_rf2 = ["Within" if d <= adcut_rf2 else "Outside" for d in md_rf2]
            ad_et2 = ["Within" if d <= adcut_et2 else "Outside" for d in md_et2]

            pred_rf2 = [normalize_label(p) for p in rf2.predict(X2_rf)]
            pred_et2 = [normalize_label(p) for p in et2.predict(X2_et)]
            proba_rf2 = rf2.predict_proba(X2_rf)[:, 1]
            proba_et2 = et2.predict_proba(X2_et)[:, 1]

            final_stage2 = []
            for i in range(len(active_smiles)):
                prf, pet = pred_rf2[i], pred_et2[i]
                arf, aet = ad_rf2[i], ad_et2[i]
                prb_rf, prb_et = proba_rf2[i], proba_et2[i]
                if prf == pet:
                    final = prf
                elif arf == "Within" and aet == "Outside":
                    final = prf
                elif aet == "Within" and arf == "Outside":
                    final = pet
                else:
                    final = prf if prb_rf >= prb_et else pet
                final_stage2.append(final)
        else:
            active_smiles, final_stage2 = [], []

        stage2_df = pd.DataFrame({
            "SMILES": active_smiles,
            "Stage2_RF": pred_rf2 if active_smiles else [],
            "Stage2_ET": pred_et2 if active_smiles else [],
            "Stage2_Consensus": final_stage2
        })

        # ==========================================================
        # 🧩 Combine Results
        # ==========================================================
        results = pd.merge(stage1_df, stage2_df, on="SMILES", how="left")
        results["Final_Prediction"] = results["Stage2_Consensus"].fillna(results["Stage1_Consensus"])

        st.dataframe(results)
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
        st.success("✅ Prediction complete!")
# ==========================================================
# 2️⃣ PROCEDURE & FLOWCHART (Boxes in front, arrows behind)
# ==========================================================
with tab2:
    st.header("Methodology and Working of AIP-G 1.0")

    st.markdown(
        """
**Two-Stage ML Workflow for Compound Activity Prediction**

The AIP-G 1.0 pipeline implements a two-stage machine learning framework for predicting GSK-3β inhibitors for Alzheimer’s disease. Here the SMILES input form manual entry or CSV is validated, canonicalized, and is used for computing predefined 1D and 2D descriptors using Mordred. These features are cleaned and standardized with standard scalar. There are 2 stages of classification, Stage 1, where Random Forest (RF) and Extra Trees (ET) models predict Active and Inactive molecules, supported by Applicability Domain (AD) checks and a consensus rule. Molecules predicted as Active move to Stage 2, where another set of RF and ET models classify Highly Active from the Active molecules, using the same AD-driven consensus logic. Also, the prediction probability in each stage is also computed. The Stage 1 and Stage 2 outputs are merged and the final prediction is then displayed, with all results available for export.
"""
    )

    # ⭐ NEW HEADING BEFORE FLOWCHART (only change requested)
    st.subheader("AIP-G 1.0 Flowchart Overview")
    st.markdown("<br>", unsafe_allow_html=True)

    # Flowchart Heading (your original line restored inside tab2)

    # ================= FLOWCHART BELOW (UNCHANGED) =================
    dot = Digraph("TwoStageFlow", engine="dot")
    dot.attr(rankdir="TB", splines="ortho", nodesep="0.6", ranksep="0.8")

    # Nodes style
    dot.attr("node", shape="box", style="rounded,filled,solid", fontsize="10", margin="0.15,0.1")
    dot.attr("edge", style="solid", arrowhead="normal", constraint="true")

    # Main nodes
    dot.node("Start", "START", fillcolor="lightblue")
    dot.node("Load", "LOAD LIBRARIES & MODELS\n• Load RDKit, Mordred, scikit-learn\n• Load RF & ET models (Stage1 & 2)\n• Load AD parameters", fillcolor="lightcyan")
    dot.node("Input", "USER INPUT\n• Enter SMILES manually\n• Upload CSV (SMILES column)", fillcolor="lightyellow")
    dot.node("Validate", "VALIDATE SMILES\n• RDKit Mol conversion\n• Remove invalid entries", fillcolor="gold")
    dot.node("Desc", "COMPUTE MORDRED DESCRIPTORS\n• ignore 3D Descriptors\n• Clean descriptor table", fillcolor="lightcoral")

    # Stage 1
    dot.node("Stage1", "STAGE 1:\nActive vs Inactive", fillcolor="orange")
    dot.node("RF1", "RF MODEL (Stage 1)\n• Predict: Active / Inactive\n• Probability\n• AD status", fillcolor="lightskyblue")
    dot.node("ET1", "ET MODEL (Stage 1)\n• Predict: Active / Inactive\n• Probability\n• AD status", fillcolor="lightgreen")
    cons1_label = (
        "CONSENSUS LOGIC (Stage 1)\n"
        "• If, RF1_prediction = ET1_prediction → Final = RF1/ET1\n"
        "• If, RF1_AD = Within and ET1_AD = Outside → Final = RF1\n"
        "• If, ET1_AD = Within and RF1_AD = Outside → Final = ET1\n"
        "• If, ET1_AD = Outside and RF1_AD = Outside → Higher Probability Prediction\n"
        "Output: Active / Inactive"
    )
    dot.node("Cons1", cons1_label, fillcolor="navajowhite", width="3.5")

    dot.node("Inactive", "INACTIVE", fillcolor="lightgray")
    dot.node("Active", "ACTIVE (to Stage 2)", fillcolor="lightgoldenrod")

    # Stage 2
    dot.node("Stage2", "STAGE 2:\nHighly Active vs Active", fillcolor="lightgreen")
    dot.node("RF2", "RF MODEL (Stage 2)\n• Predict: HighlyActive / Active\n• Probability\n• AD status", fillcolor="lightskyblue")
    dot.node("ET2", "ET MODEL (Stage 2)\n• Predict: HighlyActive / Active\n• Probability\n• AD status", fillcolor="lightgreen")
    cons2_label = (
        "CONSENSUS LOGIC (Stage 2)\n"
        "• If RF2_prediction = ET2_prediction → Final = RF2/ET2\n"
        "• If, RF2_AD = Within and ET2_AD = Outside → Final = RF2\n"
        "• If, ET2_AD = Within and RF2_AD = Outside → Final = ET2\n"
        "• If, ET2_AD = Outside and RF2_AD = Outside → Higher Probability Prediction\n"
        "Output: Highly Active / Active"
    )
    dot.node("Cons2", cons2_label, fillcolor="navajowhite", width="3.5")

    # Stage 2 outputs aligned horizontally
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node("Active2", "ACTIVE", fillcolor="lightyellow")
        s.node("HActive", "HIGHLY ACTIVE", fillcolor="palegreen")

    # Place Merge
    with dot.subgraph() as s_merge:
        s_merge.attr(rank='min')
        s_merge.node("Merge", "FINAL MERGE & OUTPUT\n• Merge Stage1 + Stage2\n• Use Stage2 for Actives only", fillcolor="lightskyblue")

    dot.node("Display", "DISPLAY & EXPORT\n• Show table\n• Download CSV", fillcolor="lightblue")
    dot.node("End", "END", fillcolor="lightblue")

    # Connections
    dot.edge("Start", "Load")
    dot.edge("Load", "Input")
    dot.edge("Input", "Validate")
    dot.edge("Validate", "Desc")
    dot.edge("Desc", "Stage1")
    dot.edge("Stage1", "RF1")
    dot.edge("Stage1", "ET1")
    dot.edge("RF1", "Cons1")
    dot.edge("ET1", "Cons1")
    dot.edge("Cons1", "Inactive", minlen="0.4")
    dot.edge("Cons1", "Active", minlen="0.4")
    dot.edge("Active", "Stage2", minlen="0.4")
    dot.edge("Stage2", "ET2")
    dot.edge("Stage2", "RF2")
    dot.edge("RF2", "Cons2")
    dot.edge("ET2", "Cons2")
    dot.edge("Cons2", "Active2")
    dot.edge("Cons2", "HActive")
    dot.edge("Inactive", "Merge")
    dot.edge("Active2", "Merge")
    dot.edge("HActive", "Merge")
    dot.edge("Merge", "Display")
    dot.edge("Display", "End")

    st.graphviz_chart(dot, use_container_width=True)


# ==========================================================
# 3️⃣ MODEL PERFORMANCE TAB
# ==========================================================
with tab3:
    st.header("Model Performance Evaluation")

    st.write("The webtool was validated for its performance with a test set of 2,662 molecules, another validation set with decoys of 1,745 molecules and a set of 320 molecules labelled as PAINS. The webtool performance during these validation steps is provided below.")

    st.subheader("1.Validation with Test Set")
    test_data = {
        "Model": ["Stage 1 - RF","Stage 1 - ET","Stage 1 - Consensus",
                  "Stage 2 - RF","Stage 2 - ET","Stage 2 - Consensus",
                  "Overall Final Prediction (2-Stage)"],
        "Accuracy": [0.4408,0.4085,0.4382,0.3701,0.3188,0.2943,0.3987],
        "Precision": [0.3491,0.3232,0.3389,0.2402,0.2303,0.2285,0.3876],
        "Recall": [0.4408,0.4085,0.4382,0.3701,0.3188,0.2943,0.3987],
        "F1 Score": [0.3876,0.3599,0.3748,0.2902,0.2570,0.2217,0.3435],
        "Sensitivity Active": [0.6042,0.5297,0.3716,0.7073,0.3848,0.2331,0.0866],
        "Sensitivity Highly Active": [0.0000,0.0000,0.0000,0.3047,0.6180,0.7639,0.3090],
        "Sensitivity Inactive": [0.5247,0.5137,0.7299,0.0000,0.0000,0.0000,0.7299],
        "ROC AUC": [0.5421,0.5151,0.5336,0.5067,0.4967,0.4972,0.5319],
        "Balanced Accuracy": [0.3763,0.3478,0.3672,0.3373,0.3343,0.3323,0.3752]
    }
    st.dataframe(pd.DataFrame(test_data))

    st.subheader("2.Validation with Decoys Dataset")
    decoy_data = {
        "Model": ["Stage 1 - RF","Stage 1 - ET","Stage 1 - Consensus",
                  "Stage 2 - RF","Stage 2 - ET","Stage 2 - Consensus",
                  "Overall Final (2-Stage)","Decoys Only","Ligands Only"],
        "Accuracy": [0.4516,0.4350,0.6281,0.0063,0.0016,0.0031,0.6275,0.6336,0.1000],
        "Precision": [0.9749,0.9718,0.9743,0.0000,0.0000,0.0000,0.9743,1.0000,0.6667],
        "Recall": [0.4516,0.4350,0.6281,0.0063,0.0016,0.0031,0.6275,0.6336,0.1000],
        "F1 Score": [0.6151,0.5983,0.7626,0.0001,0.0000,0.0001,0.7626,0.7757,0.1678],
        "Sensitivity Active": [0.4000,0.5000,0.3000,1.0000,0.3333,0.3333,0.1000,"-","-"],
        "Sensitivity Highly Active": [0.0000,0.0000,0.0000,1.0000,0.0000,1.0000,0.1000,"-","-"],
        "Sensitivity Inactive": [0.4545,0.4371,0.6336,0.0000,0.0000,0.0000,0.6336,0.6336,"-"],
        "ROC AUC": [0.4524,0.3946,0.4176,0.5014,0.4994,0.5004,0.4173,"-", "-"],
        "Balanced Accuracy": [0.2848,0.3124,0.3112,0.6667,0.1111,0.4444,0.2779,0.6336,0.1000]
    }
    st.dataframe(pd.DataFrame(decoy_data))

    st.subheader("3.Validation with PAINS Dataset")
    pains_data = {
        "Input": ["PAINS Molecules","PAINS Molecules","PAINS Molecules"],
        "Predicted Class": ["Inactive","Highly Active","Active"],
        "Count": [224,57,39],
        "Percent": ["70.00%","17.81%","12.19%"]
    }
    st.dataframe(pd.DataFrame(pains_data))

# ==========================================================
# 4️⃣ REFERENCES TAB
# ==========================================================
with tab4:
    st.header("References and Citation")

    st.markdown("""
Below is the complete list of scientific literature, software tools, and computational packages used
in the development, validation, and deployment of **AIP-G 1.0**.

---

#### 1. Machine Learning & Data Processing
1. Breiman, L. *Random Forests*. Machine Learning, 45, 5–32 (2001).  
2. Geurts, P., Ernst, D., Wehenkel, L. *Extremely Randomized Trees*. Machine Learning, 63, 3–42 (2006).  
3. Pedregosa et al. *Scikit-Learn: Machine Learning in Python*. JMLR 12, 2825–2830 (2011).  
4. Chicco, D., Jurman, G. *The advantages of the Matthews correlation coefficient (MCC)*. BMC Genomics 21, 6 (2020).

---

#### 2. Descriptor Generation & Cheminformatics
1. Moriwaki et al. *Mordred: A Comprehensive Descriptor Library for Molecular Descriptors*. J. Cheminf. 10, 4 (2018).  
2. RDKit: Open-source cheminformatics. *http://www.rdkit.org*.  
3. Todeschini, R.; Consonni, V. *Handbook of Molecular Descriptors*. Wiley-VCH (2000).

---

#### 3. Model Interpretation & Performance Evaluation
1. Powers, D. *Evaluation: Precision, Recall, F-measure, ROC, Informedness, Markedness*. JMLT 2, 37–63 (2011).  
2. Hand, D.J., Till, R.J. *A Simple Generalisation of the AUC for Multiclass Problems*. ML 45, 171–186 (2001).  
3. Trenton, M. *Balanced Accuracy and Its Advantages in Imbalanced Data*. Pattern Recogn. Lett., 120 (2019).

---

#### 4. Applicability Domain (AD)
1. Sahigara, F. et al. *Comparison of Different Approaches to Define the Applicability Domain*. J. Chemometrics 26, 269–276 (2012).  
2. Jaworska, J., Nikolova-Jeliazkova, N. *AD in QSAR Models*. Mutation Research 575, 1–2 (2005).

---

#### 5. Datasets & Decoys
1. Mysinger et al. *Directory of Useful Decoys, Enhanced (DUD-E)*. J. Med. Chem. 55, 14 (2012).  
2. GSK-3β Bioassay Data retrieved from peer-reviewed literature (details in Supplementary Material).

---

#### 6. Software, Platforms & Versions (Used in AIP-G 1.0)

| Software / Package | Version | Purpose |
|--------------------|---------|---------|
| Python | 3.10 | Development |
| Streamlit | 1.50 | Web interface |
| RDKit | 2025.03.6 | SMILES handling |
| Mordred | 1.2.0 | Descriptor generation |
| scikit-learn | 1.4.2 | ML modelling |
| NumPy | 1.25.2 | Numerical computing |
| Pandas | 2.3.2 | Data processing |
| Graphviz | latest | Flowchart rendering |
| Matplotlib | 3.10.6 | Internal plotting |

---

## How to Cite AIP-G 1.0 (Webtool Citation)
If you use the AIP-G 1.0 webtool in research or publications, please cite:

> **AIP-G 1.0 Webtool**  
> *Ajwin Joseph Martin, Dr. Dileep Kumar*  
> Version 1.0 (2025).  
> Webtool URL: *[insert URL]*  

---

## How to Cite the Associated Research Article (Pre-publication)
This tool accompanies an unpublished research manuscript.  
Until acceptance, please cite the webtool:

> **AIP-G 1.0: Machine Learning Based Virtual Screening and Molecular Dynamics Simulations for GSK3β Inhibitors in Alzheimer’s disease**  
> *Ajwin Joseph Martin, Dileep Kumar.*  
> Manuscript in preparation (2025).  
> Final journal citation will be updated once published.
> (A DOI will be added once archived.)

""")
