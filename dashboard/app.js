/**
 * Dashboard App — Connects to blockchain backend API with ethers.js fallback.
 * Implements live polling every 15 seconds with auto-refresh toggle.
 * (Day 2 — Likith: Blockchain Backend Integration)
 */
document.addEventListener("DOMContentLoaded", () => {
    const API_BASE = "http://localhost:3001/api";
    const RPC_URL = "http://127.0.0.1:8545";
    const POLL_INTERVAL_MS = 15000;

    let isAutoPolling = true;
    let pollingInterval = null;
    let useApiServer = true; // Try API server first, fallback to ethers

    // DOM elements
    const refreshBtn = document.getElementById("refresh-ledger-btn");
    const contractAddressEl = document.getElementById("contract-address");
    const contractDeployerEl = document.getElementById("contract-deployer");
    const contractEpochsEl = document.getElementById("contract-epochs-count");
    const ledgerTableBody = document.querySelector("#ledger-table tbody");

    // ── Inject auto-polling toggle into the dashboard header ──
    const dashboardHeader = document.querySelector(".dashboard-header");
    if (dashboardHeader) {
        const existingStatusDiv = dashboardHeader.querySelector("div:last-child");
        if (existingStatusDiv) {
            const pollToggleHTML = `
                <div style="display: flex; align-items: center; gap: 10px; margin-top: 8px;">
                    <label style="display: flex; align-items: center; gap: 6px; cursor: pointer; font-size: 13px; color: var(--text-muted, #888);">
                        <input type="checkbox" id="auto-poll-toggle" checked
                               style="accent-color: var(--accent-cyan, #00d4ff); width: 16px; height: 16px;">
                        Auto-refresh (15s)
                    </label>
                    <div id="poll-indicator" style="width: 8px; height: 8px; border-radius: 50%; background-color: #28a745; box-shadow: 0 0 8px #28a745; transition: all 0.3s;"></div>
                </div>
            `;
            existingStatusDiv.insertAdjacentHTML("beforeend", pollToggleHTML);
        }
    }

    const pollToggle = document.getElementById("auto-poll-toggle");
    const pollIndicator = document.getElementById("poll-indicator");

    if (pollToggle) {
        pollToggle.addEventListener("change", (e) => {
            isAutoPolling = e.target.checked;
            if (pollIndicator) {
                pollIndicator.style.backgroundColor = isAutoPolling ? "#28a745" : "#dc3545";
                pollIndicator.style.boxShadow = isAutoPolling ? "0 0 8px #28a745" : "0 0 8px #dc3545";
            }
            if (isAutoPolling) {
                startPolling();
            } else {
                stopPolling();
            }
        });
    }

    // ── Sync Ledger Button ──
    if (refreshBtn) {
        refreshBtn.addEventListener("click", async () => {
            refreshBtn.innerText = "Syncing...";
            refreshBtn.disabled = true;
            await syncAll();
            refreshBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"></path></svg> Sync Ledger`;
            refreshBtn.disabled = false;
        });
    }

    // ══════════════════════════════════════════════════════════
    // API Server Data Source (primary)
    // ══════════════════════════════════════════════════════════

    async function fetchFromAPI(endpoint) {
        const response = await fetch(`${API_BASE}${endpoint}`);
        if (!response.ok) throw new Error(`API error: ${response.status}`);
        return response.json();
    }

    async function syncViaAPI() {
        // Fetch status
        const statusData = await fetchFromAPI("/status");
        if (contractAddressEl) contractAddressEl.innerText = statusData.address;
        if (contractDeployerEl) contractDeployerEl.innerText = statusData.owner;
        if (contractEpochsEl) contractEpochsEl.innerText = statusData.roundsCount;

        // Fetch events (includes tx hashes)
        const eventsData = await fetchFromAPI("/events");
        const txHashMap = {};
        if (eventsData.events) {
            eventsData.events.forEach(e => {
                txHashMap[e.args.roundNumber] = e.transactionHash;
            });
        }

        // Fetch full ledger
        const ledgerData = await fetchFromAPI("/ledger");
        renderLedgerTable(ledgerData.ledger, txHashMap);
    }

    // ══════════════════════════════════════════════════════════
    // Ethers.js Direct Contract Calls (fallback)
    // ══════════════════════════════════════════════════════════

    async function syncViaEthers() {
        if (typeof ethers === 'undefined') {
            throw new Error("ethers.js not loaded");
        }

        const provider = new ethers.JsonRpcProvider(RPC_URL);

        // Read deployed addresses
        let deployedAddresses;
        try {
            const response = await fetch("../blockchain/deployed_addresses.json");
            deployedAddresses = await response.json();
        } catch {
            throw new Error("Cannot load deployed_addresses.json");
        }

        const contractAddress = deployedAddresses.AMLVerifier;
        if (contractAddressEl) contractAddressEl.innerText = contractAddress;

        // Minimal ABI
        const abi = [
            "function getRoundNumbers() view returns (uint256[])",
            "function getRoundsCount() view returns (uint256)",
            "function rounds(uint256) view returns (uint256 roundNumber, string modelWeightsHash, uint256 accuracy, uint256 precision, uint256 recall, uint256 f1, uint256 clientCount, uint256 timestamp, string circuitHash, address submitter)",
            "function owner() view returns (address)",
            "event RoundVerified(uint256 indexed roundNumber, string modelWeightsHash, uint256 accuracy, uint256 precision, uint256 recall, uint256 f1, uint256 clientCount, uint256 timestamp, string circuitHash, address indexed submitter)"
        ];

        const contract = new ethers.Contract(contractAddress, abi, provider);

        const owner = await contract.owner().catch(() => "Unknown");
        if (contractDeployerEl) contractDeployerEl.innerText = owner;

        const roundsCount = await contract.getRoundsCount().catch(() => 0);
        if (contractEpochsEl) contractEpochsEl.innerText = roundsCount.toString();

        // Query events for tx hashes
        const filter = contract.filters.RoundVerified();
        const events = await contract.queryFilter(filter, 0, "latest").catch(() => []);

        const txHashMap = {};
        const ledger = [];

        for (let i = events.length - 1; i >= 0; i--) {
            const event = events[i];
            const roundNumber = event.args.roundNumber.toString();
            txHashMap[roundNumber] = event.transactionHash;

            const roundData = await contract.rounds(event.args.roundNumber);
            ledger.push({
                roundNumber: roundData.roundNumber.toString(),
                modelWeightsHash: roundData.modelWeightsHash,
                accuracy: roundData.accuracy.toString(),
                precision: roundData.precision.toString(),
                recall: roundData.recall.toString(),
                f1: roundData.f1.toString(),
                clientCount: roundData.clientCount.toString(),
                timestamp: roundData.timestamp.toString(),
                circuitHash: roundData.circuitHash,
                submitter: roundData.submitter
            });
        }

        renderLedgerTable(ledger, txHashMap);
    }

    // ══════════════════════════════════════════════════════════
    // Render the ledger table
    // ══════════════════════════════════════════════════════════

    function renderLedgerTable(ledger, txHashMap) {
        if (!ledgerTableBody) return;

        if (!ledger || ledger.length === 0) {
            ledgerTableBody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 60px;">No audit logs loaded. Sync ledger or run the pipeline training first.</td></tr>`;
            return;
        }

        // Sort by round number descending (newest first)
        ledger.sort((a, b) => parseInt(b.roundNumber) - parseInt(a.roundNumber));

        ledgerTableBody.innerHTML = "";

        ledger.forEach(round => {
            const tr = document.createElement("tr");

            const verifiedF1 = (Number(round.f1) / 100).toFixed(2);
            const verifiedAcc = (Number(round.accuracy) / 100).toFixed(2);
            const timestamp = new Date(Number(round.timestamp) * 1000).toLocaleString();
            const txHash = txHashMap[round.roundNumber] || "N/A";

            const txLink = txHash !== "N/A"
                ? `<a href="https://amoy.polygonscan.com/tx/${txHash}" target="_blank" style="color: var(--accent-cyan, #00d4ff); text-decoration: none;" title="${txHash}">${txHash.substring(0, 10)}...</a>`
                : '<span style="color: var(--text-muted, #888);">N/A</span>';

            const circuitDisplay = round.circuitHash
                ? `${round.circuitHash.substring(0, 15)}...`
                : "N/A";

            tr.innerHTML = `
                <td><strong>${round.roundNumber}</strong></td>
                <td class="mono" style="font-size: 11px;" title="${round.modelWeightsHash}">${round.modelWeightsHash.substring(0, 16)}...</td>
                <td><span style="color: var(--status-success, #28a745); font-weight: bold;">${verifiedF1}%</span></td>
                <td><span style="color: var(--status-success, #28a745); font-weight: bold;">${verifiedAcc}%</span></td>
                <td>${round.clientCount}</td>
                <td>${timestamp}</td>
                <td class="mono" style="font-size: 11px;" title="${round.circuitHash}">${circuitDisplay}</td>
                <td class="mono">${txLink}</td>
            `;
            ledgerTableBody.appendChild(tr);
        });
    }

    // ══════════════════════════════════════════════════════════
    // Sync orchestrator: try API first, fallback to ethers
    // ══════════════════════════════════════════════════════════

    async function syncAll() {
        // Flash poll indicator
        if (pollIndicator) {
            pollIndicator.style.opacity = "0.4";
            setTimeout(() => { pollIndicator.style.opacity = "1"; }, 300);
        }

        if (useApiServer) {
            try {
                await syncViaAPI();
                return;
            } catch (err) {
                console.warn("API server unavailable, falling back to ethers.js:", err.message);
                useApiServer = false; // Don't keep trying API if it's down
            }
        }

        // Ethers.js fallback
        try {
            // Ensure ethers is loaded
            if (typeof ethers === 'undefined') {
                const script = document.createElement('script');
                script.src = "https://cdnjs.cloudflare.com/ajax/libs/ethers/6.7.0/ethers.umd.min.js";
                document.head.appendChild(script);
                await new Promise((resolve, reject) => {
                    script.onload = resolve;
                    script.onerror = reject;
                });
            }
            await syncViaEthers();
        } catch (err) {
            console.error("Both API and ethers fallback failed:", err);
            if (ledgerTableBody) {
                ledgerTableBody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--status-error, #e74c3c); padding: 60px;">
                    ⚠️ Connection failed. Ensure the blockchain node or API server is running.
                </td></tr>`;
            }
        }
    }

    // ══════════════════════════════════════════════════════════
    // Polling control
    // ══════════════════════════════════════════════════════════

    function startPolling() {
        stopPolling();
        syncAll(); // immediate sync
        pollingInterval = setInterval(() => {
            if (isAutoPolling) syncAll();
        }, POLL_INTERVAL_MS);
    }

    function stopPolling() {
        if (pollingInterval) {
            clearInterval(pollingInterval);
            pollingInterval = null;
        }
    }

    // ── Initial load ──
    startPolling();
});
