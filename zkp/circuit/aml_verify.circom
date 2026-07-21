pragma circom 2.1.6;

include "circomlib/circuits/poseidon.circom";
include "circomlib/circuits/comparators.circom";
include "circomlib/circuits/bitify.circom";

/*
 * aml_verify.circom  (Week 2, Day 6 — Likith)
 * ============================================================================
 * SCOPE — read this before touching anything downstream.
 *
 * This circuit proves ONE thing:
 *
 *   "The published global model for round `roundNumber` is the correct
 *    sample-count-weighted average of the private per-bank weight vectors
 *    that were committed to this proof."
 *
 * It does NOT prove:
 *   - that any bank's local training was performed correctly
 *   - that any bank's raw transaction data was not tampered with
 *   - that the resulting model is accurate, fair, or free of poisoning
 *   - correctness of GraphAwareMLP's forward pass (nonlinearities, BatchNorm,
 *     Dropout, Sigmoid) — proving a full NN forward pass in-circuit is out of
 *     scope for this project's timeline (this is the "don't boil the ocean"
 *     call flagged in the Week 2 plan)
 *
 * In one line for the README: "proves correct aggregation arithmetic across
 * bank submissions, not correctness of what any bank trained."
 *
 * ----------------------------------------------------------------------------
 * WHY WEIGHTED AVERAGE, NOT PLAIN AVERAGE
 * server.py / FedAvg weights each client's contribution by its local sample
 * count (standard FedAvg). This circuit mirrors that: each bank submits its
 * local weights AND its local sample count, and the proof shows the global
 * weights are the sample-count-weighted mean of the four.
 *
 * ----------------------------------------------------------------------------
 * WHY OFFSET-ENCODED FIXED-POINT INTEGERS (documented simplification)
 * circom's field is unsigned. Model weights are signed floats. Rather than
 * build full signed-integer arithmetic under time pressure, weights are:
 *   1. scaled by 10^SCALE (fixed-point quantization) client-side, and
 *   2. shifted by +WEIGHT_OFFSET so they're always non-negative
 * before ever being fed into the circuit. This is a known, explicitly
 * documented simplification — not a hidden assumption. Document it as such
 * in the Day 18 README, don't paper over it.
 *
 * ----------------------------------------------------------------------------
 * WHY numParams IS SMALL BY DEFAULT
 * GraphAwareMLP has ~15,000 parameters end to end. Proving aggregation over
 * all of them is possible but compiles/proves slowly and will eat your Day 8
 * budget. Default NUM_PARAMS below covers the output layer (33 params:
 * Linear(32,1) = 32 weights + 1 bias) as the demo-scale proof target.
 * Scale `NUM_PARAMS` up (and re-run trusted setup) once this is proven to
 * work end-to-end — don't start with the full model.
 * ============================================================================
 */

// Per-parameter weighted-average check with an explicit remainder, since
// circom has no native division: weightedSum = quotient * totalSamples + remainder,
// 0 <= remainder < totalSamples. `quotient` is the aggregated (offset-encoded)
// global weight for this parameter.
template WeightedAvgParam(numClients, bitWidth) {
    signal input clientWeights[numClients];  // offset-encoded, fixed-point ints
    signal input clientCounts[numClients];   // local sample counts (same for every param)
    signal input totalSamples;               // public, = sum(clientCounts)

    signal output globalWeight;              // this parameter's aggregated value

    signal weightedSum;
    signal quotient;
    signal remainder;

    // weightedSum = sum_c clientWeights[c] * clientCounts[c]
    signal partial[numClients];
    var acc = 0;
    for (var c = 0; c < numClients; c++) {
        partial[c] <== clientWeights[c] * clientCounts[c];
        acc += partial[c];
    }
    weightedSum <== acc;

    // Prover supplies quotient/remainder as a witness; circuit constrains them.
    quotient <-- weightedSum \ totalSamples;
    remainder <-- weightedSum % totalSamples;
    weightedSum === quotient * totalSamples + remainder;

    // 0 <= remainder < totalSamples
    component remainderCheck = LessThan(bitWidth);
    remainderCheck.in[0] <== remainder;
    remainderCheck.in[1] <== totalSamples;
    remainderCheck.out === 1;

    // Bound the quotient so it can't silently wrap the field (range check,
    // not a correctness check on the underlying float value itself).
    component quotientBits = Num2Bits(bitWidth);
    quotientBits.in <== quotient;

    globalWeight <== quotient;
}

template AmlVerify(numClients, numParams, bitWidth) {
    // ---- public inputs ----
    signal input roundNumber;
    signal input totalSamples;

    // ---- private inputs ----
    signal input clientCounts[numClients];
    signal input clientWeights[numClients][numParams];

    // ---- public output ----
    // Compact commitment to the aggregated global weight vector, so the
    // full ~15k-float model never has to touch the chain — only this hash
    // does. proof.json's "public signals" = [roundNumber, globalCommitment].
    signal output globalCommitment;

    // 1) sample counts must actually sum to the claimed public total
    var countSum = 0;
    for (var c = 0; c < numClients; c++) {
        countSum += clientCounts[c];
    }
    countSum === totalSamples;

    // 2) range-check every submitted weight so a malicious bank can't submit
    //    an out-of-range value to bias the aggregate via field overflow
    component weightBits[numClients][numParams];
    for (var c = 0; c < numClients; c++) {
        for (var p = 0; p < numParams; p++) {
            weightBits[c][p] = Num2Bits(bitWidth);
            weightBits[c][p].in <== clientWeights[c][p];
        }
    }

    // 3) per-parameter weighted average
    component avg[numParams];
    signal globalWeights[numParams];
    for (var p = 0; p < numParams; p++) {
        avg[p] = WeightedAvgParam(numClients, bitWidth);
        avg[p].totalSamples <== totalSamples;
        for (var c = 0; c < numClients; c++) {
            avg[p].clientWeights[c] <== clientWeights[c][p];
            avg[p].clientCounts[c] <== clientCounts[c];
        }
        globalWeights[p] <== avg[p].globalWeight;
    }

    // 4) fold the aggregated weights + round number into one hash chain
    //    commitment (Poseidon has limited arity, so chain it rather than
    //    passing all numParams+1 signals into a single call)
    component hashers[numParams];
    signal chain[numParams + 1];
    chain[0] <== roundNumber;
    for (var p = 0; p < numParams; p++) {
        hashers[p] = Poseidon(2);
        hashers[p].inputs[0] <== chain[p];
        hashers[p].inputs[1] <== globalWeights[p];
        chain[p + 1] <== hashers[p].out;
    }

    globalCommitment <== chain[numParams];
}

// Demo-scale default: 4 banks, output-layer-only (33 params: 32 weights + 1
// bias from GraphAwareMLP's final Linear(32,1)), 64-bit range checks (plenty
// for fixed-point weights scaled by 10^6 — see generate_circuit_input.js).
component main {public [roundNumber, totalSamples]} = AmlVerify(4, 33, 64);