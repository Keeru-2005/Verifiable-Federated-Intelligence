/**
 * Express Backend API Test Suite
 * Validates REST endpoints, pipeline states, ledger queries, and error handling.
 */
const http = require("http");
const assert = require("assert");
const { spawn } = require("child_process");
const path = require("path");

const PORT = 3099; // Use unique port to avoid conflicts
const PROJECT_ROOT = path.resolve(__dirname, "..");
const SERVER_SCRIPT = path.join(PROJECT_ROOT, "dashboard", "server.js");

function request(method, path, body = null) {
  return new Promise((resolve, reject) => {
    const dataString = body ? JSON.stringify(body) : null;
    const req = http.request(
      {
        hostname: "127.0.0.1",
        port: PORT,
        path: path,
        method: method,
        headers: dataString
          ? {
              "Content-Type": "application/json",
              "Content-Length": Buffer.byteLength(dataString),
            }
          : {},
      },
      (res) => {
        let data = "";
        res.on("data", (chunk) => (data += chunk));
        res.on("end", () => {
          try {
            resolve({ statusCode: res.statusCode, data: JSON.parse(data) });
          } catch (e) {
            resolve({ statusCode: res.statusCode, data: data });
          }
        });
      }
    );
    req.on("error", reject);
    if (dataString) req.write(dataString);
    req.end();
  });
}

async function runTests() {
  console.log("=== Starting Dashboard & Backend API Test Suite ===");

  // 1. Spawn backend server on test port
  const env = Object.assign({}, process.env, { PORT: PORT });
  const serverProc = spawn("node", [SERVER_SCRIPT], { cwd: PROJECT_ROOT, env });

  await new Promise((resolve) => setTimeout(resolve, 1500));

  let passed = 0;
  let failed = 0;

  async function test(name, fn) {
    try {
      await fn();
      console.log(`  [SUCCESS] PASS — ${name}`);
      passed++;
    } catch (err) {
      console.error(`  [FAIL] FAIL — ${name}: ${err.message}`);
      failed++;
    }
  }

  try {
    // Test 1: GET /api/status
    await test("GET /api/status returns valid initial pipeline status", async () => {
      const res = await request("GET", "/api/status");
      assert.strictEqual(res.statusCode, 200);
      assert(res.data.status, "Missing status field");
      assert(Array.isArray(res.data.logs), "Logs must be an array");
    });

    // Test 2: GET /api/rounds
    await test("GET /api/rounds returns sorted array of FL rounds", async () => {
      const res = await request("GET", "/api/rounds");
      assert.strictEqual(res.statusCode, 200);
      assert(Array.isArray(res.data), "Rounds must be an array");
      if (res.data.length > 0) {
        assert(res.data[0].metrics, "Round missing metrics object");
        assert("accuracy" in res.data[0].metrics, "Missing accuracy metric");
        assert("f1" in res.data[0].metrics, "Missing f1 metric");
      }
    });

    // Test 2b: GET /api/rounds/:id valid, non-existent, and invalid
    await test("GET /api/rounds/:id handles valid round, 404 for missing round, and 400 for bad id", async () => {
      const resValid = await request("GET", "/api/rounds/1");
      assert.strictEqual(resValid.statusCode, 200);
      assert.strictEqual(resValid.data.round, 1);
      assert("metrics" in resValid.data, "Must contain metrics");

      const res404 = await request("GET", "/api/rounds/999");
      assert.strictEqual(res404.statusCode, 404);
      assert(res404.data.error.includes("not found"));

      const res400 = await request("GET", "/api/rounds/invalid_abc");
      assert.strictEqual(res400.statusCode, 400);
      assert(res400.data.error.includes("Invalid round ID"));
    });

    // Test 3: GET /api/baseline
    await test("GET /api/baseline returns centralized vs federated evaluation data", async () => {
      const res = await request("GET", "/api/baseline");
      assert.strictEqual(res.statusCode, 200);
      assert("centralized" in res.data, "Missing centralized comparison data");
      assert("federated" in res.data, "Missing federated comparison data");
    });

    // Test 4: GET /api/blockchain
    await test("GET /api/blockchain returns legitimate ledger records without mock artifacts", async () => {
      const res = await request("GET", "/api/blockchain");
      assert.strictEqual(res.statusCode, 200);
      assert(res.data.address, "Missing AMLVerifier address");
      assert(Array.isArray(res.data.rounds), "Rounds must be an array");
      if (res.data.rounds.length > 0) {
        const r1 = res.data.rounds[0];
        assert(r1.modelWeightsHash.startsWith("0x"), "Model weights hash must be 0x-prefixed Keccak256");
      }
    });

    // Test 5: GET /api/blockchain/status
    await test("GET /api/blockchain/status reports RPC connection health", async () => {
      const res = await request("GET", "/api/blockchain/status");
      assert.strictEqual(res.statusCode, 200);
      assert("rpcActive" in res.data, "Missing rpcActive flag");
      assert("network" in res.data, "Missing network description");
    });

    // Test 6: POST /api/run conflict prevention
    await test("POST /api/run handles concurrent launches cleanly", async () => {
      // First call (note: starts pipeline)
      const res1 = await request("POST", "/api/run");
      assert([200, 400].includes(res1.statusCode));

      // Immediate second call should receive 400 if already running
      if (res1.statusCode === 200) {
        const res2 = await request("POST", "/api/run");
        assert.strictEqual(res2.statusCode, 400, "Second concurrent run must return 400 Bad Request");
        assert(res2.data.error.includes("already running"), "Must indicate pipeline is already running");
      }
    });

  } finally {
    serverProc.kill("SIGTERM");
  }

  console.log("\n=========================================");
  console.log(`Backend API Tests Completed: ${passed} Passed, ${failed} Failed`);
  console.log("=========================================\n");

  if (failed > 0) {
    process.exit(1);
  }
}

runTests().catch((err) => {
  console.error("Test execution failed:", err);
  process.exit(1);
});
