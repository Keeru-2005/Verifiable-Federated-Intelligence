/**
 * Orchestrator Entry Point
 * Runs the end-to-end automation pipeline.
 */
const path = require('path');
const { spawn } = require('child_process');

const scriptToRun = path.join(__dirname, 'coordinate.js');

console.log("[START] Launching Verifiable Federated Intelligence Orchestrator...");

const proc = spawn('node', [scriptToRun], { stdio: 'inherit', cwd: __dirname });

proc.on('close', (code) => {
    if (code === 0) {
        console.log("[COMPLETE] Orchestration finished successfully!");
    } else {
        console.error(`[FAIL] Orchestrator process exited with code ${code}`);
        process.exit(code);
    }
});
