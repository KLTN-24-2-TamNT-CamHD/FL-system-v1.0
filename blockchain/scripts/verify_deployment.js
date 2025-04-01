const FraudDetectionFederated = artifacts.require("FraudDetectionFederated");

module.exports = async function(callback) {
  try {
    // Get accounts first
    const accounts = await web3.eth.getAccounts();
    const fraudDetection = await FraudDetectionFederated.deployed();
    
    console.log("\nDeployment Verification:");
    console.log("Timestamp (UTC): 2025-03-31 13:30:04");
    console.log("Deployer: dinhcam89");
    
    // Verify contract owner
    const owner = await fraudDetection.owner();
    console.log("\nContract Owner:", owner);
    
    // Get current round
    const currentRound = await fraudDetection.currentRound();
    console.log("Current Round:", currentRound.toString());
    
    // Get authorized institutions count
    const authorizedInstitutions = await fraudDetection.authorizedInstitutions(0).catch(() => null);
    console.log("\nAuthorized Institutions:");
    
    if (authorizedInstitutions) {
        // Try to get institution details
        try {
            const institution = await fraudDetection.institutions(authorizedInstitutions);
            console.log(`Institution at ${authorizedInstitutions}:`, {
                name: institution.name,
                authorized: institution.authorized
            });
        } catch (err) {
            console.log("No institutions registered yet");
        }
    } else {
        console.log("No institutions registered yet");
    }
    
    // Print available accounts for reference
    console.log("\nAvailable Accounts:");
    accounts.forEach((account, index) => {
        console.log(`Account ${index}: ${account}`);
    });
    
    callback();
  } catch (error) {
    console.error("\nVerification Error:", error);
    callback(error);
  }
};