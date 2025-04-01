const FraudDetectionFederated = artifacts.require("FraudDetectionFederated");

module.exports = async function (deployer, network, accounts) {
    console.log("Deployment Info:");
    console.log("Timestamp (UTC): 2025-03-31 13:30:04");
    console.log("Deployer: dinhcam89");
    console.log("Network: ", network);

    // Deploy the contract
    await deployer.deploy(FraudDetectionFederated);
    const fraudDetection = await FraudDetectionFederated.deployed();

    console.log("FraudDetectionFederated deployed at:", fraudDetection.address);

    // Set up test data for Ganache
    if (network === 'ganache') {
        try {
            // Register some test institutions using Ganache accounts
            await fraudDetection.registerInstitution(
                accounts[1],
                "Test Institution 1"
            );
            console.log("Registered Test Institution 1:", accounts[1]);

            await fraudDetection.registerInstitution(
                accounts[2],
                "Test Institution 2"
            );
            console.log("Registered Test Institution 2:", accounts[2]);

            // Start first training round
            await fraudDetection.initiateTrainingRound();
            console.log("Initiated first training round");

        } catch (error) {
            console.error("Error setting up test data:", error);
        }
    }
};