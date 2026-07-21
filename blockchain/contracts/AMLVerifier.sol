// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./Groth16Verifier.sol";

contract AMLVerifier {
    struct AMLRound {
        uint256 roundNumber;
        string modelWeightsHash;
        uint256 accuracy;      // Scaled by 10000 (e.g., 9542 = 95.42%)
        uint256 precision;     // Scaled by 10000
        uint256 recall;        // Scaled by 10000
        uint256 f1;            // Scaled by 10000
        uint256 clientCount;
        uint256 timestamp;
        string circuitHash;
        address submitter;
    }

    Groth16Verifier public immutable verifierContract;
    address public owner;

    // Mapping from round number to details
    mapping(uint256 => AMLRound) public rounds;
    uint256[] public roundNumbers;

    event RoundVerified(
        uint256 indexed roundNumber,
        string modelWeightsHash,
        uint256 accuracy,
        uint256 precision,
        uint256 recall,
        uint256 f1,
        uint256 clientCount,
        uint256 timestamp,
        string circuitHash,
        address indexed submitter
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can perform this action");
        _;
    }

    constructor(address _verifierAddress) {
        require(_verifierAddress != address(0), "Invalid verifier address");
        verifierContract = Groth16Verifier(_verifierAddress);
        owner = msg.sender;
    }

    /**
     * @notice Verify a federated learning round using ZK proof and log results to on-chain ledger.
     * @param a zk-SNARK Groth16 proof parameter A
     * @param b zk-SNARK Groth16 proof parameter B
     * @param c zk-SNARK Groth16 proof parameter C
     * @param input Public signals from the ZK proof
     * @param roundNumber Federated learning round index
     * @param modelWeightsHash Hash of the global model weights JSON
     * @param accuracy Model accuracy metric scaled by 10000
     * @param precision Model precision metric scaled by 10000
     * @param recall Model recall metric scaled by 10000
     * @param f1 Model F1-score metric scaled by 10000
     * @param clientCount Number of active clients involved in aggregation
     * @param circuitHash IPFS/Git identifier of the verified circuit configuration
     */
    function verifyAndRecordRound(
        uint[2] calldata a,
        uint[2][2] calldata b,
        uint[2] calldata c,
        uint[3] calldata input,
        uint256 roundNumber,
        string calldata modelWeightsHash,
        uint256 accuracy,
        uint256 precision,
        uint256 recall,
        uint256 f1,
        uint256 clientCount,
        string calldata circuitHash
    ) external returns (bool) {
        // 1. Verify the cryptographic proof using the imported Groth16 contract
        bool isProofValid = verifierContract.verifyProof(a, b, c, input);
        require(isProofValid, "Cryptographic zk-SNARK verification failed");

        // 2. Ensure round is not already recorded (or allow updating but log updates)
        require(rounds[roundNumber].timestamp == 0, "Round details already finalized on-chain");

        // 3. Create ledger entry
        rounds[roundNumber] = AMLRound({
            roundNumber: roundNumber,
            modelWeightsHash: modelWeightsHash,
            accuracy: accuracy,
            precision: precision,
            recall: recall,
            f1: f1,
            clientCount: clientCount,
            timestamp: block.timestamp,
            circuitHash: circuitHash,
            submitter: msg.sender
        });

        roundNumbers.push(roundNumber);

        // 4. Emit event for logging and frontend updates
        emit RoundVerified(
            roundNumber,
            modelWeightsHash,
            accuracy,
            precision,
            recall,
            f1,
            clientCount,
            block.timestamp,
            circuitHash,
            msg.sender
        );

        return true;
    }

    /**
     * @notice Get all recorded round numbers.
     */
    function getRoundNumbers() external view returns (uint256[] memory) {
        return roundNumbers;
    }

    /**
     * @notice Get total number of recorded rounds.
     */
    function getRoundsCount() external view returns (uint256) {
        return roundNumbers.length;
    }
}
