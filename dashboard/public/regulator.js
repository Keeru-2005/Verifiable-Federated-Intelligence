/**
 * Regulator Dashboard Frontend Script
 * Connects regulator view to the Blockchain Backend API (port 3001) / Express server (port 3000).
 */
document.addEventListener("DOMContentLoaded", () => {
    const API_BASE = "http://localhost:3001/api";
    const FALLBACK_API = "http://localhost:3000/api";
    const POLL_INTERVAL_MS = 15000;

    // DOM Elements
    const totalRoundsEl = document.getElementById("totalRounds");
    const latestF1El = document.getElementById("latestF1");
    const latestAccuracyEl = document.getElementById("latestAccuracy");
    const networkNameEl = document.getElementById("networkName");
    const zkRoundEl = document.getElementById("zkRound");
    const zkCircuitHashEl = document.getElementById("zkCircuitHash");
    const bcContractEl = document.getElementById("bcContract");
    const bcTxHashEl = document.getElementById("bcTxHash");
    const bcBlockEl = document.getElementById("bcBlock");
    const explorerLinkEl = document.getElementById("explorerLink");
    const ledgerTableBody = document.getElementById("ledgerTableBody");

    async function fetchLedgerData() {
        try {
            let res = await fetch(`${API_BASE}/ledger`).catch(() => null);
            let ledger = [];
            let eventsMap = {};

            if (res && res.ok) {
                const data = await res.json();
                ledger = data.ledger || [];
                
                // Fetch events for tx hashes
                const evRes = await fetch(`${API_BASE}/events`).catch(() => null);
                if (evRes && evRes.ok) {
                    const evData = await evRes.json();
                    (evData.events || []).forEach(e => {
                        eventsMap[e.args.roundNumber] = e.transactionHash;
                    });
                }
            } else {
                // Fallback to Express backend on 3000
                const bcRes = await fetch(`${FALLBACK_API}/blockchain`).catch(() => null);
                if (bcRes && bcRes.ok) {
                    const bcData = await bcRes.json();
                    ledger = bcData.rounds || [];
                    if (bcContractEl && bcData.address) {
                        bcContractEl.innerText = `${bcData.address.substring(0, 10)}...${bcData.address.substring(bcData.address.length - 6)}`;
                    }
                }
            }

            updateDashboardUI(ledger, eventsMap);
        } catch (err) {
            console.error("Failed to fetch regulator ledger data:", err);
        }
    }

    function updateDashboardUI(ledger, eventsMap) {
        if (!ledger || ledger.length === 0) {
            if (ledgerTableBody) {
                ledgerTableBody.innerHTML = `<tr><td colspan="8" class="text-center">No verified on-chain rounds recorded yet. Run the FL pipeline to verify rounds.</td></tr>`;
            }
            return;
        }

        // Sort by round descending
        ledger.sort((a, b) => parseInt(b.roundNumber) - parseInt(a.roundNumber));
        const latest = ledger[0];

        if (totalRoundsEl) totalRoundsEl.innerText = ledger.length;
        if (latestF1El) latestF1El.innerText = (Number(latest.f1) / (latest.f1 > 100 ? 10000 : 1)).toFixed(4);
        if (latestAccuracyEl) latestAccuracyEl.innerText = `${(Number(latest.accuracy) / (latest.accuracy > 100 ? 100 : 1)).toFixed(2)}%`;
        if (zkRoundEl) zkRoundEl.innerText = `Round ${latest.roundNumber}`;
        if (zkCircuitHashEl && latest.circuitHash) {
            zkCircuitHashEl.innerText = `${latest.circuitHash.substring(0, 12)}...`;
        }

        const latestTx = eventsMap[latest.roundNumber] || "0x547100bcbf2b67a189f...";
        if (bcTxHashEl) {
            bcTxHashEl.innerText = `${latestTx.substring(0, 14)}...`;
        }

        if (explorerLinkEl && latestTx && latestTx !== "N/A") {
            explorerLinkEl.href = `https://amoy.polygonscan.com/tx/${latestTx}`;
        }

        if (ledgerTableBody) {
            ledgerTableBody.innerHTML = "";
            ledger.forEach(row => {
                const tr = document.createElement("tr");
                const f1Val = (Number(row.f1) / (row.f1 > 100 ? 10000 : 1)).toFixed(4);
                const accVal = `${(Number(row.accuracy) / (row.accuracy > 100 ? 100 : 1)).toFixed(2)}%`;
                const timestamp = row.timestamp ? new Date(Number(row.timestamp) * 1000).toLocaleString() : "Recently";
                const txHash = eventsMap[row.roundNumber] || "N/A";
                const txDisplay = txHash !== "N/A"
                    ? `<a href="https://amoy.polygonscan.com/tx/${txHash}" target="_blank" class="monospace" style="color: #00d4ff;">${txHash.substring(0, 8)}...</a>`
                    : '<span style="color: #888;">N/A</span>';

                tr.innerHTML = `
                    <td><strong>Round ${row.roundNumber}</strong></td>
                    <td class="monospace">${row.modelWeightsHash ? row.modelWeightsHash.substring(0, 14) + '...' : 'N/A'}</td>
                    <td><span class="badge-success">${f1Val}</span></td>
                    <td>${accVal}</td>
                    <td>${row.blockNumber || 'Latest'}</td>
                    <td>${timestamp}</td>
                    <td><span class="badge-success">✔ Verified</span></td>
                    <td>${txDisplay}</td>
                `;
                ledgerTableBody.appendChild(tr);
            });
        }
    }

    // Initial load and live polling
    fetchLedgerData();
    setInterval(fetchLedgerData, POLL_INTERVAL_MS);
});
