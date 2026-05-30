# Alzheimer’s Disease Inhibitor Predictor for GSK-3β (AIP-G 1.0)

### Overview
AIP-G 1.0 is a Streamlit web application implementing a highly stringent, two-stage machine learning pipeline to predict GSK-3β inhibitors for Alzheimer’s disease. It employs predictive topological Mordred molecular descriptors alongside Random Forest (RF) and Extremely Randomized Trees (ET) models. The pipeline is rigorously validated on external test sets, decoy datasets, and PAINS datasets to enforce ultra-high precision and minimize false positives during virtual screening.

### Features
* **Flexible Input:** Input SMILES strings manually or via CSV upload.
* **Automated Feature Extraction:** Automatically computes 1D and 2D predictive Mordred molecular descriptors.
* **Two-Stage Prediction:** Evaluates molecules sequentially: Stage 1 classifies molecules as Active vs. Inactive, while Stage 2 further classifies Stage 1 hits as Highly Active vs. Active.
* **Strict Consensus Logic:** Implements independent Applicability Domain (AD) constraints and a hierarchical consensus voting rule to actively prevent structural decoys and reactive artifacts (PAINS) from being misclassified.
* **Exportable Data:** Download standard predictions and detailed tabular data for further analysis.

---

### Access the Web Tool
You can access and use the AIP-G 1.0 virtual screening pipeline directly through your web browser without any installation required:

🔗 **[Launch AIP-G 1.0 Web Tool Here](https://aip-g-1-two-stage-screening.streamlit.app/)**

---

### Citation
If you utilize the AIP-G 1.0 webtool or concepts in your research, please cite:

> **AIP-G 1.0 Webtool** | Dileep Kumar et al. | Version 1.0 (2025).  
> **Webtool URL:** *(https://aip-g-1-two-stage-screening.streamlit.app/)*

> **AIP-G 1.0: Machine Learning Based Virtual Screening and Molecular Dynamics Simulations for GSK3β Inhibitors in Alzheimer’s disease** | A. J. Martin, D. Kumar. | *Manuscript in preparation* (2025).

*(Final journal citation and DOI will be updated here once published and archived.)*

---

### Copyright & Intellectual Property

**© 2025 Ajwin Joseph Martin and Dr. Dileep Kumar. All rights reserved.**

The source code, algorithms, consensus logic, and trained models associated with AIP-G 1.0 are the exclusive intellectual property of the authors. This repository is made public for the sole purpose of deploying the Streamlit web application and facilitating transparency for academic peer review.

**Permissions:**
* You are permitted to view the source code for educational and peer-review purposes.
* You are permitted to use the deployed web tool via the provided Streamlit URL for your own virtual screening tasks, provided proper citation is given.

**Restrictions:**
* You may **NOT** copy, reproduce, distribute, modify, or create derivative works from this codebase.
* You may **NOT** use the code or models for any commercial or private non-commercial deployment without explicit written permission from the authors.

For licensing inquiries or permission requests, please contact the authors directly.
