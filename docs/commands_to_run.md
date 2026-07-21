# 📜 Commands Runner Guide

This document lists all the exact commands required to run the Verifiable Federated Intelligence project, organized by execution mode, directory, and sequence.

---

## 🚀 Option A: Automated Single-Command Execution (Recommended)

To run the entire pipeline (Data partitioning → FL Training → ZK Proof Generation → Blockchain Submission) automatically:

*   **Working Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence`
*   **Command:**
    ```bash
    node orchestrate.js
    ```

---

## 🛠️ Option B: Step-by-Step Manual Execution

If you prefer to run each subsystem manually in sequence:

### Step 1: Data Partitioning
Split dataset into 4 bank node shards (80/20 train/test split).
*   **Working Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence`
*   **Command:**
    ```bash
    python fl_implementation/split_into_banks.py
    ```

### Step 2: Launch FL Server
Start the central Flower aggregation server.
*   **Working Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence`
*   **PowerShell:**
    ```powershell
    $env:PYTHONIOENCODING="utf-8"
    python fl_implementation\server.py
    ```
*   **CMD:**
    ```cmd
    set PYTHONIOENCODING=utf-8
    python fl_implementation\server.py
    ```

### Step 3: Launch 4 Bank Clients
Open 4 separate terminal windows and run each command:
*   **Working Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence`

*   **Terminal 1 (Bank 1):**
    ```powershell
    $env:DATA_FILE="fl_implementation\data\bank_1.csv"; python fl_implementation\client.py
    ```
*   **Terminal 2 (Bank 2):**
    ```powershell
    $env:DATA_FILE="fl_implementation\data\bank_2.csv"; python fl_implementation\client.py
    ```
*   **Terminal 3 (Bank 3):**
    ```powershell
    $env:DATA_FILE="fl_implementation\data\bank_3.csv"; python fl_implementation\client.py
    ```
*   **Terminal 4 (Bank 4):**
    ```powershell
    $env:DATA_FILE="fl_implementation\data\bank_4.csv"; python fl_implementation\client.py
    ```

### Step 4: Zero-Knowledge Proof Generation
Generate the zk-SNARK proof from the exported model weights.
*   **Working Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence\zk_proof`
*   **Command:**
    ```bash
    node generate_proof.js
    ```

### Step 5: Blockchain Ledger Submission
Record the verified proof and metrics on the Polygon Amoy testnet contract.
*   **Working Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence\contracts_project`
*   **Command:**
    ```bash
    node scripts/deploy_ledger.js
    ```

---

## 🌐 Option C: Launch Web Dashboard

To view the live Bank Admin & Regulator UI:

*   **Working Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence\dashboard`
*   **Command:**
    ```bash
    npm start
    ```
*   **Access:** Open `http://localhost:3000` in your web browser.

---

## ⚡ Option D: Stress Testing & Benchmarks

*   **Node Failure Stress Test:**
    ```bash
    python evaluation/stress_test.py
    ```
*   **Metrics Benchmark Collection:**
    ```bash
    python evaluation/benchmark_metrics.py
    ```
