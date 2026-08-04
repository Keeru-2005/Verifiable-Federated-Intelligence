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
    mapping(uint256 => bool) public isRoundVerified;
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

    event InvalidProof(
        uint256 indexed roundNumber,
        address indexed submitter,
        string reason
    );

    event ProofRejected(
        uint256 indexed roundNumber,
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

    function verifyOnly(
        uint[2] calldata a,
        uint[2][2] calldata b,
        uint[2] calldata c,
        uint[3] calldata input
    ) external view returns (bool) {
        return verifierContract.verifyProof(a, b, c, input);
    }

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

        require(rounds[roundNumber].timestamp == 0, "Round details already finalized on-chain");

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

        isRoundVerified[roundNumber] = true;
        roundNumbers.push(roundNumber);

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

    function getRoundNumbers() external view returns (uint256[] memory) {
        return roundNumbers;
    }

    function getRoundsCount() external view returns (uint256) {
        return roundNumbers.length;
    }

    function getLatestRound() external view returns (AMLRound memory) {
        require(roundNumbers.length > 0, "No rounds verified yet");
        uint256 latestRoundNumber = roundNumbers[roundNumbers.length - 1];
        return rounds[latestRoundNumber];
    }

    function getLedgerEntry(uint256 roundNumber) external view returns (AMLRound memory) {
        require(isRoundVerified[roundNumber], "Round not verified");
        return rounds[roundNumber];
    }
}
