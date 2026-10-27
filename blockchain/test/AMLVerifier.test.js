const { expect } = require("chai");
const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

describe("AMLVerifier & Groth16Verifier End-to-End System", function () {
  let groth16Verifier;
  let amlVerifier;
  let owner;
  let otherAccount;
  let mockProofData;

  before(async function () {
    [owner, otherAccount] = await ethers.getSigners();

    // Read frozen proof.json schema to ensure test matches spec
    const proofPath = path.join(__dirname, "../../schemas/proof.json");
    const rawData = fs.readFileSync(proofPath);
    mockProofData = JSON.parse(rawData);
  });

  beforeEach(async function () {
    // Deploy Groth16Verifier
    const Groth16Verifier = await ethers.getContractFactory("Groth16Verifier");
    groth16Verifier = await Groth16Verifier.deploy();
    await groth16Verifier.waitForDeployment();

    // Deploy AMLVerifier passing the verifier contract address
    const AMLVerifier = await ethers.getContractFactory("AMLVerifier");
    amlVerifier = await AMLVerifier.deploy(await groth16Verifier.getAddress());
    await amlVerifier.waitForDeployment();
  });

  it("Should deploy both contracts and establish the correct owner", async function () {
    expect(await amlVerifier.owner()).to.equal(owner.address);
    expect(await amlVerifier.verifierContract()).to.equal(await groth16Verifier.getAddress());
  });

  it("Should successfully verify a valid ZK proof and record round details", async function () {
    // Parse values from mockProofData schema
    const a = mockProofData.proof.pi_a.slice(0, 2);
    const b = [
      [mockProofData.proof.pi_b[0][1], mockProofData.proof.pi_b[0][0]],
      [mockProofData.proof.pi_b[1][1], mockProofData.proof.pi_b[1][0]]
    ];
    const c = mockProofData.proof.pi_c.slice(0, 2);
    const input = mockProofData.publicSignals;

    const roundNumber = mockProofData.round_reference;
    const modelWeightsHash = "0x" + mockProofData.circuit_hash.slice(2);
    const accuracy = 9542; // scaled by 10000 (95.42%)
    const precision = 9213;
    const recall = 9085;
    const f1 = 9149;
    const clientCount = 4;
    const circuitHash = mockProofData.circuit_hash;

    // Submit transaction
    const tx = await amlVerifier.verifyAndRecordRound(
      a,
      b,
      c,
      input,
      roundNumber,
      modelWeightsHash,
      accuracy,
      precision,
      recall,
      f1,
      clientCount,
      circuitHash
    );

    // Assert event emission
    await expect(tx)
      .to.emit(amlVerifier, "RoundVerified")
      .withArgs(
        roundNumber,
        modelWeightsHash,
        accuracy,
        precision,
        recall,
        f1,
        clientCount,
        anyValue => anyValue > 0, // checks block timestamp is set
        circuitHash,
        owner.address
      );

    // Retrieve ledger state from contract mappings
    const roundDetails = await amlVerifier.rounds(roundNumber);
    expect(roundDetails.roundNumber).to.equal(roundNumber);
    expect(roundDetails.modelWeightsHash).to.equal(modelWeightsHash);
    expect(roundDetails.accuracy).to.equal(accuracy);
    expect(roundDetails.precision).to.equal(precision);
    expect(roundDetails.recall).to.equal(recall);
    expect(roundDetails.f1).to.equal(f1);
    expect(roundDetails.clientCount).to.equal(clientCount);
    expect(roundDetails.circuitHash).to.equal(circuitHash);
    expect(roundDetails.submitter).to.equal(owner.address);

    // Check round lists
    const roundNumbersList = await amlVerifier.getRoundNumbers();
    expect(roundNumbersList.length).to.equal(1);
    expect(roundNumbersList[0]).to.equal(roundNumber);
    expect(await amlVerifier.getRoundsCount()).to.equal(1);
  });

  it("Should reject double submission of the same round", async function () {
    const a = mockProofData.proof.pi_a.slice(0, 2);
    const b = [
      [mockProofData.proof.pi_b[0][1], mockProofData.proof.pi_b[0][0]],
      [mockProofData.proof.pi_b[1][1], mockProofData.proof.pi_b[1][0]]
    ];
    const c = mockProofData.proof.pi_c.slice(0, 2);
    const input = mockProofData.publicSignals;

    const roundNumber = mockProofData.round_reference;
    const modelWeightsHash = "0x" + mockProofData.circuit_hash.slice(2);

    // First submission
    await amlVerifier.verifyAndRecordRound(
      a, b, c, input,
      roundNumber, modelWeightsHash,
      9500, 9200, 9000, 9100, 4,
      mockProofData.circuit_hash
    );

    // Second submission should revert
    await expect(
      amlVerifier.verifyAndRecordRound(
        a, b, c, input,
        roundNumber, modelWeightsHash,
        9500, 9200, 9000, 9100, 4,
        mockProofData.circuit_hash
      )
    ).to.be.revertedWith("Round details already finalized on-chain");
  });

  it("Should revert if cryptographic verification fails", async function () {
    const a = mockProofData.proof.pi_a.slice(0, 2);
    const b = [
      [mockProofData.proof.pi_b[0][1], mockProofData.proof.pi_b[0][0]],
      [mockProofData.proof.pi_b[1][1], mockProofData.proof.pi_b[1][0]]
    ];
    const c = mockProofData.proof.pi_c.slice(0, 2);
    // Alter input to fail verifyProof (should revert because input doesn't match verifyProof success paths)
    const invalidInput = ["9", "9", "9"];

    const roundNumber = mockProofData.round_reference + 1; // Different round to not clash
    const modelWeightsHash = "0x" + mockProofData.circuit_hash.slice(2);

    await expect(
      amlVerifier.verifyAndRecordRound(
        a, b, c, invalidInput,
        roundNumber, modelWeightsHash,
        9500, 9200, 9000, 9100, 4,
        mockProofData.circuit_hash
      )
    ).to.be.revertedWith("Cryptographic zk-SNARK verification failed");
  });

  it("Should revert if public commitment is altered", async function () {
    const a = mockProofData.proof.pi_a.slice(0, 2);
    const b = [
      [mockProofData.proof.pi_b[0][1], mockProofData.proof.pi_b[0][0]],
      [mockProofData.proof.pi_b[1][1], mockProofData.proof.pi_b[1][0]]
    ];
    const c = mockProofData.proof.pi_c.slice(0, 2);
    // Tamper the globalCommitment signal
    const tamperedInput = [
      "12345678901234567890123456789012345678901234567890",
      mockProofData.publicSignals[1],
      mockProofData.publicSignals[2]
    ];

    await expect(
      amlVerifier.verifyAndRecordRound(
        a, b, c, tamperedInput,
        999, "0x1234",
        9500, 9200, 9000, 9100, 4,
        mockProofData.circuit_hash
      )
    ).to.be.revertedWith("Cryptographic zk-SNARK verification failed");
  });

  it("Should revert if proof point A is corrupted", async function () {
    const corruptedA = [
      "11111111111111111111111111111111111111111111111111",
      mockProofData.proof.pi_a[1]
    ];
    const b = [
      [mockProofData.proof.pi_b[0][1], mockProofData.proof.pi_b[0][0]],
      [mockProofData.proof.pi_b[1][1], mockProofData.proof.pi_b[1][0]]
    ];
    const c = mockProofData.proof.pi_c.slice(0, 2);
    const input = mockProofData.publicSignals;

    await expect(
      amlVerifier.verifyAndRecordRound(
        corruptedA, b, c, input,
        998, "0x1234",
        9500, 9200, 9000, 9100, 4,
        mockProofData.circuit_hash
      )
    ).to.be.revertedWith("Cryptographic zk-SNARK verification failed");
  });

  it("Should correctly retrieve ledger entry for a verified round and revert for unverified", async function () {
    const a = mockProofData.proof.pi_a.slice(0, 2);
    const b = [
      [mockProofData.proof.pi_b[0][1], mockProofData.proof.pi_b[0][0]],
      [mockProofData.proof.pi_b[1][1], mockProofData.proof.pi_b[1][0]]
    ];
    const c = mockProofData.proof.pi_c.slice(0, 2);
    const input = mockProofData.publicSignals;
    const roundNumber = 50;

    await amlVerifier.verifyAndRecordRound(
      a, b, c, input,
      roundNumber, "0xabcdef",
      9800, 9700, 9600, 9650, 4,
      mockProofData.circuit_hash
    );

    const entry = await amlVerifier.getLedgerEntry(roundNumber);
    expect(entry.roundNumber).to.equal(roundNumber);
    expect(entry.accuracy).to.equal(9800);
    expect(entry.f1).to.equal(9650);

    await expect(amlVerifier.getLedgerEntry(99999)).to.be.revertedWith("Round not verified");
  });

  it("Should revert if deployed with zero address verifier", async function () {
    const AMLVerifierFactory = await ethers.getContractFactory("AMLVerifier");
    await expect(
      AMLVerifierFactory.deploy(ethers.ZeroAddress)
    ).to.be.revertedWith("Invalid verifier address");
  });

  it("Should revert getLatestRound when no rounds have been recorded", async function () {
    const AMLVerifierFactory = await ethers.getContractFactory("AMLVerifier");
    const freshVerifier = await AMLVerifierFactory.deploy(await groth16Verifier.getAddress());
    await expect(freshVerifier.getLatestRound()).to.be.revertedWith("No rounds verified yet");
  });

  it("Should return false from verifyOnly on invalid inputs without reverting", async function () {
    const a = mockProofData.proof.pi_a.slice(0, 2);
    const b = [
      [mockProofData.proof.pi_b[0][1], mockProofData.proof.pi_b[0][0]],
      [mockProofData.proof.pi_b[1][1], mockProofData.proof.pi_b[1][0]]
    ];
    const c = mockProofData.proof.pi_c.slice(0, 2);
    const invalidInput = ["0", "0", "0"];

    const isValid = await amlVerifier.verifyOnly(a, b, c, invalidInput);
    expect(isValid).to.equal(false);
  });

  it("Should accurately track round count and round numbers array", async function () {
    const count = await amlVerifier.getRoundsCount();
    const numbers = await amlVerifier.getRoundNumbers();
    expect(numbers.length).to.equal(Number(count));
  });
});
