# 🧪 Temporary Test Execution Guide

This document provides step-by-step instructions and commands to test all project components locally and via Docker.

---

## ⚡ Method 1: Single-Command Automated Local Test (Fastest)

Run the full pipeline (Data Partitioning → Federated Learning → ZK Proof Generation → Blockchain Ledger) in one command:

*   **Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence`
*   **Command:**
    ```bash
    node orchestrate.js
    ```

---

## 🌐 Method 2: Test the Web & Regulator Dashboards

Start the Express backend and open the dashboards in your web browser:

*   **Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence\dashboard`
*   **Command:**
    ```bash
    npm start
    ```
*   **Access URLs:**
    - 🏦 **Bank Admin Dashboard:** `http://localhost:3000`
    - 📜 **Regulator Audit Dashboard:** `http://localhost:3000/regulator.html`

---

## 🐳 Method 3: Test Docker Environment (Optimized CPU PyTorch)

We updated `Dockerfile.fl_server` and `Dockerfile.bank_node` to use PyTorch CPU wheels (`~150MB` instead of `1.6GB CUDA`), resolving pip download timeouts:

*   **Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence`
*   **Command:**
    ```bash
    docker compose up --build
    ```

---

## 🛠️ Method 4: Manual Step-by-Step Component Verification

### Step 1: Data Partitioning
*   **Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence`
*   **Command:**
    ```bash
    python fl_implementation/split_into_banks.py
    ```

### Step 2: Start FL Server
*   **Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence`
*   **Command (WSL / Bash):**
    ```bash
    PYTHONIOENCODING=utf-8 python fl_implementation/server.py
    ```

### Step 3: Start 4 Bank Clients (In 4 separate terminals)
*   **Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence`
*   **Terminal 1:** `DATA_FILE=fl_implementation/data/bank_1.csv python fl_implementation/client.py`
*   **Terminal 2:** `DATA_FILE=fl_implementation/data/bank_2.csv python fl_implementation/client.py`
*   **Terminal 3:** `DATA_FILE=fl_implementation/data/bank_3.csv python fl_implementation/client.py`
*   **Terminal 4:** `DATA_FILE=fl_implementation/data/bank_4.csv python fl_implementation/client.py`

### Step 4: Generate Zero-Knowledge Proof
*   **Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence\zk_proof`
*   **Command:**
    ```bash
    node generate_proof.js
    ```

### Step 5: Submit Proof to Polygon Ledger
*   **Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence\contracts_project`
*   **Command:**
    ```bash
    node scripts/deploy_ledger.js
    ```

---

## ⚡ Method 5: Run Node Failure Stress Test
*   **Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence`
*   **Command:**
    ```bash
    python evaluation/stress_test.py
    ```
