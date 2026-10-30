const { exec, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const ROUND_TO_PROVE = 1;

async function runCommand(command, cwd) {
    return new Promise((resolve, reject) => {
        console.log(`\n> ${command}`);
        const proc = exec(command, { cwd }, (error, stdout, stderr) => {
            if (error) {
                console.error(`Error: ${error.message}`);
                return reject(error);
            }
            resolve(stdout);
        });
        proc.stdout.pipe(process.stdout);
        proc.stderr.pipe(process.stderr);
    });
}

async function main() {
    console.log("=== VFI Week 3 Automation Pipeline ===");

    const schemasDir = path.join(__dirname, 'schemas');
    if (!fs.existsSync(schemasDir)) {
        fs.mkdirSync(schemasDir);
    }
    
    const modelPath = path.join(schemasDir, `global_model_round_${ROUND_TO_PROVE}.json`);
    
    // Check if FL round is complete
    if (!fs.existsSync(modelPath)) {
        console.log("Starting FL training via docker-compose...");
        const flProcess = spawn('docker-compose', ['up', '--build'], { stdio: 'inherit', cwd: __dirname });
        
        await new Promise(resolve => {
            const interval = setInterval(() => {
                if (fs.existsSync(modelPath)) {
                    clearInterval(interval);
                    console.log(`\n[SUCCESS] FL Round ${ROUND_TO_PROVE} complete!`);
                    flProcess.kill();
                    resolve();
                }
            }, 5000);
        });
    } else {
        console.log(`\n[SKIP] FL Round ${ROUND_TO_PROVE} already exists.`);
    }

    console.log("\n[STEP 1] Generating Circuit Input...");
    await runCommand(`node zkp/circuit/generate_circuit_input.js clients.json ${ROUND_TO_PROVE}`, __dirname);
    
    console.log("\n[STEP 2] Generating genuine zk-SNARK Proof via SnarkJS...");
    const { generateProof } = require(path.join(__dirname, 'zkp', 'generate_proof.js'));
    const inputPath = path.join(__dirname, 'input.json');
    await generateProof(inputPath, ROUND_TO_PROVE);

    console.log("\n[STEP 4] Submitting Proof on-chain...");
    let networkFlag = "";
    try {
        const http = require('http');
        await new Promise((resolve, reject) => {
            const req = http.get('http://127.0.0.1:8545', () => { networkFlag = " --network localhost"; resolve(); });
            req.on('error', () => resolve()); // Node not running, use default hardhat in-memory network
            req.setTimeout(1000, () => { req.destroy(); resolve(); });
        });
    } catch (e) {}

    await runCommand(`npx hardhat run scripts/submit_proof.js${networkFlag}`, path.join(__dirname, 'blockchain'));
    
    console.log("\n=== Pipeline Complete ===");
}

main().catch(console.error);
