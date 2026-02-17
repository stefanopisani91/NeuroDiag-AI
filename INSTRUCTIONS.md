# 🚀 Deployment Instructions: GitHub & Streamlit Community Cloud

Now that your project structure is professional, follow these steps to deploy it.

## Phase 1: Local Git Initialization

1.  **Initialize Git**:
    Open your terminal in the project folder and run:
    ```bash
    git init
    ```

2.  **Add Files**:
    ```bash
    git add .
    ```

3.  **Commit**:
    ```bash
    git commit -m "Initial commit: NeuroDiag-AI v1.0"
    ```

## Phase 2: GitHub Repository

1.  Go to [GitHub.com](https://github.com/new).
2.  Create a **New Repository**.
    - **Name**: `NeuroDiag-AI` (or `Psycho-Tensor`)
    - **Description**: Clinical Decision Support System simulation (Portfolio Project)
    - **Public/Private**: Public (for portfolio visibility).
    - **Do NOT** initialize with README, .gitignore, or license (we already have them local).
3.  Click **Create repository**.

4.  **Connect Local to Remote**:
    Copy the commands from the section "…or push an existing repository from the command line":
    ```bash
    git remote add origin https://github.com/YOUR_USERNAME/NeuroDiag-AI.git
    git branch -M main
    git push -u origin main
    ```

## Phase 3: Streamlit Community Cloud

1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Click **New app**.
3.  Select your GitHub repository (`NeuroDiag-AI`).
4.  **Main file path**:
    - Enter `app/app.py` (Critical! Since we moved the file).
5.  Click **Deploy!** 🎈

## ✅ Verification
- The app should spin up.
- It will automatically install dependencies from `requirements.txt`.
- It will find the model/data because we updated `app.py` to use relative paths (`../models/`, etc.).

## 📝 Troubleshooting
- If you see a "File not found" error, ensure `app/app.py` has the correct `ROOT_DIR` calculation (lines 10-15).
- If database errors occur, the app will auto-create a fresh DB. Privacy safe!
