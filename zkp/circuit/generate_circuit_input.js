
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
  const [, , clientsPath, roundArg] = process.argv;
  if (!clientsPath || !roundArg) {
    console.error("Usage: node generate_circuit_input.js clients.json round_number");
    process.exit(1);
  }

  const { clientWeights, clientCounts } = JSON.parse(fs.readFileSync(clientsPath, "utf8"));

  const numClients = clientWeights.length;
  const numParams = clientWeights[0].length;
  if (clientCounts.length !== numClients) {
    throw new Error("clientCounts length must match number of clients in clientWeights");
  }
  clientWeights.forEach((row, i) => {
    if (row.length !== numParams) {
      throw new Error(`Client ${i} has ${row.length} params, expected ${numParams} (mismatch across clients)`);
    }
  });

  const totalSamples = clientCounts.reduce((a, b) => a + b, 0);

  const input = {
    roundNumber: roundArg,
    totalSamples: totalSamples.toString(),
    clientCounts: clientCounts.map(String),
    clientWeights: clientWeights.map((row) => row.map(encode)),
  };

  fs.writeFileSync("input.json", JSON.stringify(input, null, 2));
  console.log(
    `Wrote input.json: ${numClients} clients, ${numParams} params/client, ` +
    `totalSamples=${totalSamples}, round=${roundArg}`
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