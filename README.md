# Alzheimer’s Disease Inhibitor Predictor for GSK-3β (AIP-G 1.0)

## Overview
AIP-G 1.0 is a Streamlit web app implementing a two-stage machine learning pipeline to predict GSK-3β inhibitors for Alzheimer’s disease. It employs Mordred molecular descriptors, Random Forest, and Extremely Randomized Trees models, validated on test, decoy, and PAINS datasets.

## Features
- Input SMILES strings manually or via CSV upload
- Computes Mordred molecular descriptors automatically
- Performs two-stage prediction: Active vs Inactive, then Highly Active vs Active
- Applicability Domain checks included
- Downloadable prediction results

## Installation

### Prerequisites
- Python 3.10
- Pip or Conda
- Git (optional for cloning)

### Install dependencies
