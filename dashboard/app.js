// Wait for the DOM to load
document.addEventListener("DOMContentLoaded", () => {
    initRegulatorDashboard();
});

async function initRegulatorDashboard() {
    const refreshBtn = document.getElementById("refresh-ledger-btn");
    
    // Setup ethers provider
    // Hardhat local node usually runs on http://127.0.0.1:8545
    // Ensure ethers.js is available via CDN in the HTML
    if (typeof ethers === 'undefined') {
        // Inject ethers dynamically if missing
        const script = document.createElement('script');
        script.src = "https://cdnjs.cloudflare.com/ajax/libs/ethers/6.7.0/ethers.umd.min.js";
        script.onload = () => initSync(refreshBtn);
        document.head.appendChild(script);
    } else {
        initSync(refreshBtn);
    }
}

function initSync(refreshBtn) {
    const provider = new ethers.JsonRpcProvider("http://127.0.0.1:8545");

    refreshBtn.addEventListener("click", async () => {
        await syncLedger(provider);
    });

    // Initial sync
    syncLedger(provider);
}

async function syncLedger(provider) {
    try {
        const refreshBtn = document.getElementById("refresh-ledger-btn");
        refreshBtn.innerText = "Syncing...";
        
        // Fetch deployed addresses
        const response = await fetch("../blockchain/deployed_addresses.json");
        const deployedAddresses = await response.json();
        
        const contractAddress = deployedAddresses.AMLVerifier;
        document.getElementById("contract-address").innerText = contractAddress;
        
        // Setup contract
        // Minimal ABI
        const abi = [
            "function getEpochsCount() view returns (uint256)",
            "function rounds(uint256) view returns (uint256 roundNumber, bytes32 modelWeightsHash, uint256 accuracy, uint256 precision, uint256 recall, uint256 f1, uint256 clientCount, string circuitHash, uint256 timestamp, address submitter)",
            "function owner() view returns (address)",
            "event RoundVerified(uint256 indexed roundNumber, bytes32 modelWeightsHash, uint256 timestamp, address indexed submitter)"
        ];
        
        const contract = new ethers.Contract(contractAddress, abi, provider);
        
        const owner = await contract.owner().catch(() => "Unknown (node down)");
        document.getElementById("contract-deployer").innerText = owner;
        
        const epochsCount = await contract.getEpochsCount().catch(() => 0);
        document.getElementById("contract-epochs-count").innerText = epochsCount.toString();
        
        const tbody = document.querySelector("#ledger-table tbody");
        
        if (epochsCount == 0) {
            tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 60px;">No audit logs loaded. Sync ledger or run the pipeline training first.</td></tr>`;
        }
        
        const filter = contract.filters.RoundVerified();
        const events = await contract.queryFilter(filter, 0, "latest");
        
        if (events.length > 0) {
            tbody.innerHTML = ""; // Clear table
            // Iterate events backwards (newest first)
            for (let i = events.length - 1; i >= 0; i--) {
                const event = events[i];
                const roundNumber = event.args.roundNumber;
                
                // Fetch full round data
                const roundData = await contract.rounds(roundNumber);
                
                const tr = document.createElement("tr");
                
                const verifiedF1 = (Number(roundData.f1) / 100).toFixed(2);
                const verifiedAcc = (Number(roundData.accuracy) / 100).toFixed(2);
                const timestamp = new Date(Number(roundData.timestamp) * 1000).toLocaleString();
                
                // Mock block explorer URL for transaction hash
                const txLink = `<a href="https://amoy.polygonscan.com/tx/${event.transactionHash}" target="_blank" style="color: var(--accent-cyan); text-decoration: none;" title="${event.transactionHash}">${event.transactionHash.substring(0, 10)}...</a>`;
                
                tr.innerHTML = `
                    <td><strong>${roundData.roundNumber.toString()}</strong></td>
                    <td class="mono" style="font-size: 11px;">${roundData.modelWeightsHash}</td>
                    <td><span style="color: var(--status-success); font-weight: bold;">${verifiedF1}%</span></td>
                    <td><span style="color: var(--status-success); font-weight: bold;">${verifiedAcc}%</span></td>
                    <td>${roundData.clientCount.toString()}</td>
                    <td>${timestamp}</td>
                    <td class="mono" style="font-size: 11px;">${roundData.circuitHash.substring(0, 15)}...</td>
                    <td class="mono">${txLink}</td>
                `;
                tbody.appendChild(tr);
            }
        }
        
        refreshBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"></path></svg> Sync Ledger`;
        
    } catch (err) {
        console.error("Error syncing ledger:", err);
        const refreshBtn = document.getElementById("refresh-ledger-btn");
        refreshBtn.innerText = "Sync Failed";
        setTimeout(() => {
            refreshBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"></path></svg> Sync Ledger`;
        }, 2000);
    }
}
