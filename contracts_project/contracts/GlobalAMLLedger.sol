// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title GlobalAMLLedger
 * @dev Records the verified rounds of the federated learning global model for AML.
 */
contract GlobalAMLLedger {
    struct RoundRecord {
        uint256 roundId;
        string modelHash;
        bool isVerified;
        uint256 timestamp;
        address verifier;
    }

    // Mapping from roundId to its record
    mapping(uint256 => RoundRecord) public rounds;
    
    // Total number of recorded rounds
    uint256 public totalRounds;

    event RoundRecorded(uint256 indexed roundId, string modelHash, bool isVerified, address verifier);

    /**
     * @dev Records a new verified model round.
     * @param _roundId The federated learning round number.
     * @param _modelHash The IPFS CID or Poseidon hash of the global model weights.
     * @param _isVerified Boolean indicating if the ZKP was verified successfully.
     */
    function recordRound(uint256 _roundId, string memory _modelHash, bool _isVerified) external {
        require(rounds[_roundId].timestamp == 0, "Round already recorded");

        rounds[_roundId] = RoundRecord({
            roundId: _roundId,
            modelHash: _modelHash,
            isVerified: _isVerified,
            timestamp: block.timestamp,
            verifier: msg.sender
        });

        totalRounds += 1;

        emit RoundRecorded(_roundId, _modelHash, _isVerified, msg.sender);
    }
}
