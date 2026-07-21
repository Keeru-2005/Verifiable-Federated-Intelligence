const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("=== Starting Deployment ===");
  const [deployer] = await ethers.getSigners();
  console.log(`Deploying contracts with account: ${deployer.address}`);

  // Deploy Groth16Verifier
  console.log("Deploying Groth16Verifier...");
  const Groth16Verifier = await ethers.getContractFactory("Groth16Verifier");
  const verifier = await Groth16Verifier.deploy();
  await verifier.waitForDeployment();
  const verifierAddress = await verifier.getAddress();
  console.log(`Groth16Verifier deployed to: ${verifierAddress}`);

  // Deploy AMLVerifier passing the verifier address
  console.log("Deploying AMLVerifier...");
  const AMLVerifier = await ethers.getContractFactory("AMLVerifier");
  const amlVerifier = await AMLVerifier.deploy(verifierAddress);
  await amlVerifier.waitForDeployment();
  const amlVerifierAddress = await amlVerifier.getAddress();
  console.log(`AMLVerifier deployed to: ${amlVerifierAddress}`);

  // Save deployed addresses to a file
  const addresses = {
    network: hre.network.name,
    Groth16Verifier: verifierAddress,
    AMLVerifier: amlVerifierAddress,
    deployer: deployer.address,
    timestamp: new Date().toISOString()
  };

  const outputDir = path.join(__dirname, "../");
  fs.writeFileSync(
    path.join(outputDir, "deployed_addresses.json"),
    JSON.stringify(addresses, null, 2)
  );
  console.log(`Saved deployed addresses to blockchain/deployed_addresses.json`);
  console.log("=== Deployment Complete ===");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
