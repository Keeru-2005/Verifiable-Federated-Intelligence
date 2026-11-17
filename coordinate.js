const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

function getSelectedRound() {
    const schemasDir = path.join(__dirname, 'schemas');
    const finalPath = path.join(schemasDir, 'final_evaluation.json');
    if (!fs.existsSync(finalPath)) {
        throw new Error('No final_evaluation.json. Complete FL training and validation selection before proving.');
    }
    const finalResult = JSON.parse(fs.readFileSync(finalPath, 'utf8'));
    const manifestPath = path.join(__dirname, 'data', 'preprocess_manifest.json');
    if (!fs.existsSync(manifestPath)) throw new Error('Current preprocessing manifest is missing.');
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    if (manifest.pipeline_version !== 'causal-v3' || manifest.status !== 'complete' || finalResult.run_id !== manifest.run_id) {
        throw new Error('Final evaluation belongs to stale or incompatible preprocessing artifacts.');
    }
    const selectedRound = Number(finalResult.selected_round);
    if (!Number.isInteger(selectedRound) || selectedRound < 1) {
        throw new Error('final_evaluation.json has no valid selected_round.');
    }
    return selectedRound;
}

const ROUND_TO_PROVE = getSelectedRound();

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
    const updatePath = path.join(schemasDir, 'round_updates', `round_${ROUND_TO_PROVE}.json`);
    if (!fs.existsSync(modelPath) || !fs.existsSync(updatePath)) {
        throw new Error(`Selected round ${ROUND_TO_PROVE} is missing its model or exact Flower update record.`);
    }
    const modelRecord = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
    const updateRecord = JSON.parse(fs.readFileSync(updatePath, 'utf8'));
    const currentRunId = JSON.parse(fs.readFileSync(path.join(__dirname, 'data', 'preprocess_manifest.json'), 'utf8')).run_id;
    if (modelRecord.run_id !== currentRunId || updateRecord.run_id !== currentRunId) {
        throw new Error('Selected model or client update record belongs to a different preprocessing run.');
    }

    console.log("\n[STEP 1] Preparing exact Flower client updates and sample counts...");
    await runCommand(`python fl_implementation/extract_client_weights.py ${ROUND_TO_PROVE}`, __dirname);

    const proofInputPath = path.join(__dirname, 'zkp', 'circuit', 'private_inputs', `round_${ROUND_TO_PROVE}`, 'input.json');
    const proofSourcePath = path.join(schemasDir, 'round_updates', `circuit_round_${ROUND_TO_PROVE}.json`);
    console.log("\n[STEP 2] Generating private circuit input...");
    await runCommand(`node zkp/circuit/generate_circuit_input.js ${proofSourcePath} ${ROUND_TO_PROVE} ${proofInputPath}`, __dirname);
    
    console.log("\n[STEP 3] Generating and locally verifying the Groth16 proof...");
    const { generateProof } = require(path.join(__dirname, 'zkp', 'generate_proof.js'));
    await generateProof(proofInputPath, ROUND_TO_PROVE);

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
