# 🛠️ Verifiable Federated Intelligence: Commands Guide

This document lists all the commands needed to execute each stage of the project, including the directory from which they must be run.

---

## 📂 Phase 1: Data Partitioning
Split the source CSV into 4 separate bank node shards (80/20 train/test split).

*   **Working Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence`
*   **Command (PowerShell/CMD):**
    ```powershell
    python fl_implementation\split_into_banks.py
    ```

---

## 🤖 Phase 2: Federated Learning Execution
Run the federated learning server first, then spin up the 4 bank node clients to train the `GraphAwareMLP`.

### 1. FL Server (Central Aggregator)
Start this server first. It will wait for clients to connect.
*   **Working Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence`
*   **Command (PowerShell):**
    ```powershell
    $env:PYTHONIOENCODING="utf-8"
    python fl_implementation\server.py
    ```

### 2. FL Bank Clients (Nodes 1 - 4)
Open 4 separate terminal windows/tabs and run each of the following.
*   **Working Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence`

*   **Client 1:**
    ```powershell
    $env:DATA_FILE="fl_implementation\data\bank_1.csv"; python fl_implementation\client.py
    ```
*   **Client 2:**
    ```powershell
    $env:DATA_FILE="fl_implementation\data\bank_2.csv"; python fl_implementation\client.py
    ```
*   **Client 3:**
    ```powershell
    $env:DATA_FILE="fl_implementation\data\bank_3.csv"; python fl_implementation\client.py
    ```
*   **Client 4:**
    ```powershell
    $env:DATA_FILE="fl_implementation\data\bank_4.csv"; python fl_implementation\client.py
    ```

*Once all 4 clients finish 10 rounds of training, the server will export the final model weights to `schemas/global_model_round_10.json` (or `global_model_round_N.json`).*

---

## 🔒 Phase 3: Zero-Knowledge Proof (ZKP) Generation
Extract model weights from the FL outputs and generate the zk-SNARK proof.

*   **Working Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence\zk_proof`
*   **Command (PowerShell/CMD):**
    ```powershell
    node generate_proof.js
    ```
    *(This script automatically processes the latest round JSON, formats inputs, simulates/generates the proof.json, and verifies it locally.)*

---

## 📜 Phase 4: Global AML Ledger Smart Contract
Simulate deploying the Ledger smart contract and submitting the ZK proof status and model hash to the Polygon Amoy testnet.

*   **Working Directory:** `c:\Users\keeru\Capstone work\project\Verifiable-Federated-Intelligence\contracts_project`
*   **Command (PowerShell/CMD):**
    ```powershell
    node scripts\deploy_ledger.js
    ```
