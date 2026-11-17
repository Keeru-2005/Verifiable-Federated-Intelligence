
const fs = require("fs");

const SCALE = 1_000_000;
const OFFSET = 2 ** 32;

function encode(weight) {
  const scaled = Math.round(weight * SCALE);
  const encoded = scaled + OFFSET;
  if (encoded < 0) {
    throw new Error(
      `Weight ${weight} underflows OFFSET after scaling (scaled=${scaled}). ` +
      `Increase OFFSET in both this script and aml_verify.circom if this fires.`
    );
  }
  return encoded.toString(); // field elements are passed as strings to snarkjs
}

function main() {
  const [, , clientsPath, roundArg, outputPathArg] = process.argv;
  if (!clientsPath || !roundArg) {
    console.error("Usage: node generate_circuit_input.js round_updates.json round_number [private_output_path]");
    process.exit(1);
  }

  const source = JSON.parse(fs.readFileSync(clientsPath, "utf8"));
  if (Number(source.round) !== Number(roundArg)) {
    throw new Error(`Proof round ${roundArg} does not match update record round ${source.round}.`);
  }
  const { clientWeights, clientCounts, globalOutputLayerParameters } = source;

  if (!Array.isArray(clientWeights) || !Array.isArray(clientCounts)) {
    throw new Error("Round update record must include clientWeights and clientCounts arrays.");
  }
  const numClients = clientWeights.length;
  if (numClients !== 4 || !clientWeights.every(Array.isArray)) {
    throw new Error(`Compiled circuit expects 4 client vectors; got ${numClients}.`);
  }
  const numParams = clientWeights[0].length;
  if (numParams !== 33) {
    throw new Error(`Compiled circuit expects 4 clients × 33 values; got ${numClients} × ${numParams}.`);
  }
  if (clientCounts.length !== numClients) {
    throw new Error("clientCounts length must match number of clients in clientWeights");
  }
  if (!clientCounts.every((count) => Number.isSafeInteger(Number(count)) && Number(count) > 0)) {
    throw new Error("Every client count must be a positive safe integer.");
  }
  if (!source.run_id || typeof source.run_id !== "string") {
    throw new Error("Round update record must include the preprocessing run_id.");
  }
  clientWeights.forEach((row, i) => {
    if (row.length !== numParams) {
      throw new Error(`Client ${i} has ${row.length} params, expected ${numParams} (mismatch across clients)`);
    }
    if (!row.every((weight) => Number.isFinite(Number(weight)))) {
      throw new Error(`Client ${i} contains a non-finite output-layer parameter.`);
    }
  });

  if (!Array.isArray(globalOutputLayerParameters) || globalOutputLayerParameters.length !== numParams) {
    throw new Error("Round record must include the exact 33-parameter Flower aggregate for cross-checking.");
  }
  if (!globalOutputLayerParameters.every((weight) => Number.isFinite(Number(weight)))) {
    throw new Error("Global output layer contains a non-finite parameter.");
  }
  const counts = clientCounts.map(Number);
  const totalSamples = counts.reduce((a, b) => a + b, 0);
  for (let p = 0; p < numParams; p++) {
    const expected = clientWeights.reduce((sum, weights, c) => sum + Number(weights[p]) * counts[c], 0) / totalSamples;
    if (Math.abs(expected - Number(globalOutputLayerParameters[p])) > 2e-5) {
      throw new Error(`Client updates/counts do not reproduce Flower aggregate at parameter ${p}.`);
    }
  }

  const input = {
    roundNumber: roundArg.toString(),
    totalSamples: totalSamples.toString(),
    clientCounts: counts.map(String),
    clientWeights: clientWeights.map((row) => row.map(encode)),
  };

  const path = require("path");
  const circuitDir = __dirname;
  const privateDir = path.join(circuitDir, "private_inputs");
  const outputPath = outputPathArg
    ? path.resolve(outputPathArg)
    : path.join(privateDir, `round_${roundArg}`, "input.json");
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(input, null, 2));

  const contextPath = path.join(path.dirname(outputPath), "context.json");
  fs.writeFileSync(contextPath, JSON.stringify({
    round: Number(roundArg),
    run_id: source.run_id,
    totalSamples,
    clientCounts: counts,
    globalOutputLayerParameters,
    sourceUpdateRecord: path.resolve(clientsPath),
  }, null, 2));

  console.log(
    `Wrote input.json: ${numClients} clients, ${numParams} params/client, ` +
    `totalSamples=${totalSamples}, round=${roundArg}, path=${outputPath}`
  );
  console.log(
    "Reminder: after proving, the circuit's public globalCommitment output is " +
    "offset-encoded too -- to recover the true aggregated weight for parameter p, " +
    "decode as (quotient_p - OFFSET) / SCALE once you've pulled quotient_p out of " +
    "the witness (the commitment itself is a Poseidon hash, so this only applies " +
    "to values you additionally export directly, e.g. for global_model_round_N.json)."
  );
}

main();
