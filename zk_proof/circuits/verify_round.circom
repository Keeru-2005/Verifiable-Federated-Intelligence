pragma circom 2.0.0;

include "../node_modules/circomlib/circuits/poseidon.circom";

template VerifyRound(N) {
    signal input weights[N];
    signal output hash;

    component poseidon = Poseidon(N);
    for (var i = 0; i < N; i++) {
        poseidon.inputs[i] <== weights[i];
    }
    hash <== poseidon.out;
}

// We hash the first 10 weights as a proof of knowledge of the global model.
// In a full implementation, this could use a Merkle tree for all weights.
component main = VerifyRound(10);
