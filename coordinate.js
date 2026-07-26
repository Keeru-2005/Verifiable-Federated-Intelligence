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
    
    console.log("\n[STEP 2] Generating zk-SNARK Proof...");
    await runCommand(`npx snarkjs groth16 fullprove input.json zkp/circuit/aml_verify_js/aml_verify.wasm zkp/circuit/aml_verify_final.zkey schemas/raw_proof.json schemas/public.json`, __dirname);
    
    // Wrap proof according to frozen schema
    console.log("\n[STEP 3] Formatting proof.json...");
    const rawProof = JSON.parse(fs.readFileSync(path.join(schemasDir, 'raw_proof.json')));
    const publicSignals = JSON.parse(fs.readFileSync(path.join(schemasDir, 'public.json')));
    const circuitFile = fs.readFileSync(path.join(__dirname, 'zkp/circuit/aml_verify.circom'));
    const circuitHash = crypto.createHash('sha256').update(circuitFile).digest('hex');

    const formattedProof = {
        proof: rawProof,
        publicSignals: publicSignals,
        round_reference: ROUND_TO_PROVE,
        circuit_hash: circuitHash
    };

    fs.writeFileSync(path.join(schemasDir, 'proof.json'), JSON.stringify(formattedProof, null, 2));
    console.log("proof.json created successfully matching the frozen schema.");

    console.log("\n[STEP 4] Submitting Proof on-chain...");
    await runCommand(`npx hardhat run scripts/submit_proof.js --network localhost`, path.join(__dirname, 'blockchain'));
    
    console.log("\n=== Pipeline Complete ===");
}

main().catch(console.error);
