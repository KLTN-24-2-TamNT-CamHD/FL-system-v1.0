// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Federation {
    // Model structure
    struct Model {
        string ipfsHash;
        uint256 round;
        string version;
        uint256 timestamp;
        uint256 participatingClients;
        address publisher;
        bool isActive;
    }
    
    // Client contribution structure
    struct ClientContribution {
        uint256 contributionCount;
        uint256 totalScore;
        bool isAuthorized;
        uint256 lastContributionTimestamp;
        uint256 rewardsEarned;
        bool rewardsClaimed;
    }
    
    // Contribution record structure
    struct ContributionRecord {
        address clientAddress;
        uint256 round;
        string ipfsHash;
        uint256 accuracy;
        uint256 timestamp;
        uint256 score;
        bool rewarded;
    }
    
    // Main storage for models
    mapping(bytes32 => Model) public models;
    
    // Model IDs by round
    mapping(uint256 => bytes32[]) public modelsByRound;
    
    // Latest model ID for each version prefix
    mapping(string => bytes32) public latestVersions;
    
    // Client registry
    mapping(address => ClientContribution) public clientRegistry;
    
    // Authorized client addresses
    address[] public authorizedClients;
    
    // Contribution records by client address
    mapping(address => ContributionRecord[]) public contributionRecords;
    
    // Contributions by round
    mapping(uint256 => ContributionRecord[]) public roundContributions;
    
    // Total rewards allocated
    uint256 public totalRewardsAllocated;
    
    // Events
    event ModelRegistered(bytes32 indexed modelId, string ipfsHash, uint256 round, string version);
    event ModelUpdated(bytes32 indexed modelId, string ipfsHash, uint256 round, string version);
    event ModelDeactivated(bytes32 indexed modelId);
    event ClientAuthorized(address indexed clientAddress);
    event ClientDeauthorized(address indexed clientAddress);
    event ContributionRecorded(address indexed clientAddress, uint256 round, uint256 score);
    event RewardAllocated(address indexed clientAddress, uint256 amount);
    event RewardClaimed(address indexed clientAddress, uint256 amount);
    
    // Owner of the contract
    address public owner;
    
    constructor() {
        owner = msg.sender;
    }
    
    // Modifier to restrict certain functions to the owner
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }
    
    // Generate a unique model ID from the IPFS hash and round
    function generateModelId(string memory ipfsHash, uint256 round) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(ipfsHash, round));
    }
    

    //////////////////////////////////////////////////////////
    ///////////   MODEL MANAGEMENT FUNCTIONS  ////////////////
    //////////////////////////////////////////////////////////
    // Register a new model
    function registerModel(
        string memory ipfsHash, 
        uint256 round, 
        string memory version,
        uint256 participatingClients
    ) public returns (bytes32) {
        // Generate model ID
        bytes32 modelId = generateModelId(ipfsHash, round);
        
        // Ensure it doesn't already exist
        require(models[modelId].timestamp == 0, "Model already exists");
        
        // Store the model
        models[modelId] = Model({
            ipfsHash: ipfsHash,
            round: round,
            version: version,
            timestamp: block.timestamp,
            participatingClients: participatingClients,
            publisher: msg.sender,
            isActive: true
        });
        
        // Add to round mapping
        modelsByRound[round].push(modelId);
        
        // Update latest version
        string memory versionPrefix = getVersionPrefix(version);
        
        // If there's no latest version for this prefix or this one is newer
        if (latestVersions[versionPrefix] == bytes32(0) || 
            compareVersions(version, getVersionFromModelId(latestVersions[versionPrefix]))) {
            latestVersions[versionPrefix] = modelId;
        }
        
        // Emit event
        emit ModelRegistered(modelId, ipfsHash, round, version);
        
        return modelId;
    }
    
    // Update an existing model by round
    function updateModelByRound(
        string memory ipfsHash, 
        uint256 round, 
        string memory version,
        uint256 participatingClients
    ) public onlyOwner returns (bytes32) {
        // Check if any model exists for this round
        require(modelsByRound[round].length > 0, "No models exist for this round");
        
        // Generate new model ID
        bytes32 modelId = generateModelId(ipfsHash, round);
        
        // Check if this specific model already exists
        if (models[modelId].timestamp > 0) {
            // Model with this exact hash and round already exists
            // Just return the existing ID
            return modelId;
        }
        
        // Store the model
        models[modelId] = Model({
            ipfsHash: ipfsHash,
            round: round,
            version: version,
            timestamp: block.timestamp,
            participatingClients: participatingClients,
            publisher: msg.sender,
            isActive: true
        });
        
        // Add to round mapping
        modelsByRound[round].push(modelId);
        
        // Update latest version
        string memory versionPrefix = getVersionPrefix(version);
        
        // If there's no latest version for this prefix or this one is newer
        if (latestVersions[versionPrefix] == bytes32(0) || 
            compareVersions(version, getVersionFromModelId(latestVersions[versionPrefix]))) {
            latestVersions[versionPrefix] = modelId;
        }
        
        // Emit event
        emit ModelUpdated(modelId, ipfsHash, round, version);
        
        return modelId;
    }

    // Get the latest model for a specific version prefix
    function getLatestModel(string memory versionPrefix) public view returns (
        bytes32 modelId,
        string memory ipfsHash,
        uint256 round,
        string memory version,
        uint256 timestamp,
        uint256 participatingClients
    ) {
        bytes32 id = latestVersions[versionPrefix];
        require(id != bytes32(0), "No models for this version prefix");
        
        Model storage model = models[id];
        return (
            id,
            model.ipfsHash,
            model.round,
            model.version,
            model.timestamp,
            model.participatingClients
        );
    }
    
    // Get all models for a specific round
    function getModelsByRound(uint256 round) public view returns (bytes32[] memory) {
        return modelsByRound[round];
    }
    
    // Get the most recent model for a specific round
    function getLatestModelByRound(uint256 round) public view returns (
        bytes32 modelId,
        string memory ipfsHash,
        string memory version,
        uint256 timestamp,
        uint256 participatingClients,
        address publisher,
        bool isActive
    ) {
        require(modelsByRound[round].length > 0, "No models for this round");
        
        // Get the last model in the array (most recently added)
        bytes32 id = modelsByRound[round][modelsByRound[round].length - 1];
        Model storage model = models[id];
        
        return (
            id,
            model.ipfsHash,
            model.version,
            model.timestamp,
            model.participatingClients,
            model.publisher,
            model.isActive
        );
    }
    
    // Get model details
    function getModelDetails(bytes32 modelId) public view returns (
        string memory ipfsHash,
        uint256 round,
        string memory version,
        uint256 timestamp,
        uint256 participatingClients,
        address publisher,
        bool isActive
    ) {
        Model storage model = models[modelId];
        require(model.timestamp > 0, "Model does not exist");
        
        return (
            model.ipfsHash,
            model.round,
            model.version,
            model.timestamp,
            model.participatingClients,
            model.publisher,
            model.isActive
        );
    }
    
    // Get version prefix (e.g., "1.0" from "1.0.3")
    function getVersionPrefix(string memory version) internal pure returns (string memory) {
        // This is a simplified implementation that assumes format major.minor.patch
        bytes memory versionBytes = bytes(version);
        uint dotCount = 0;
        uint lastDotPos = 0;
        
        for (uint i = 0; i < versionBytes.length; i++) {
            if (versionBytes[i] == '.') {
                dotCount++;
                lastDotPos = i;
                if (dotCount == 2) break;
            }
        }
        
        if (dotCount < 2) return version; // No patch number found
        
        // Create prefix (major.minor)
        bytes memory prefix = new bytes(lastDotPos);
        for (uint i = 0; i < lastDotPos; i++) {
            prefix[i] = versionBytes[i];
        }
        
        return string(prefix);
    }
    
    // Compare versions (returns true if v1 is newer than v2)
    function compareVersions(string memory v1, string memory v2) internal pure returns (bool) {
        // Simple string comparison - assumes properly formatted semantic versions
        // In a production environment, this would parse and compare version components
        bytes memory b1 = bytes(v1);
        bytes memory b2 = bytes(v2);
        
        for (uint i = 0; i < b1.length && i < b2.length; i++) {
            if (b1[i] != b2[i]) {
                return b1[i] > b2[i];
            }
        }
        
        return b1.length > b2.length;
    }
    
    // Get the version from a model ID
    function getVersionFromModelId(bytes32 modelId) public view returns (string memory) {
        return models[modelId].version;
    }
    
    
    
    // Deactivate a model (e.g., if it's found to be problematic)
    function deactivateModel(bytes32 modelId) public onlyOwner {
        require(models[modelId].timestamp > 0, "Model does not exist");
        require(models[modelId].isActive, "Model is already inactive");
        
        models[modelId].isActive = false;
        
        emit ModelDeactivated(modelId);
    }
    
    //////////////////////////////////////////////////////////
    ///////////   MODEL MANAGEMENT FUNCTIONS  ////////////////
    //////////////////////////////////////////////////////////
    // Authorize a client to participate in federated learning
    function authorizeClient(address clientAddress) public onlyOwner {
        require(clientAddress != address(0), "Invalid client address");
        
        ClientContribution storage client = clientRegistry[clientAddress];
        
        if (!client.isAuthorized) {
            client.isAuthorized = true;
            authorizedClients.push(clientAddress);
            
            emit ClientAuthorized(clientAddress);
        }
    }
    
    // Authorize multiple clients at once
    function authorizeClients(address[] memory clientAddresses) public onlyOwner {
        for (uint i = 0; i < clientAddresses.length; i++) {
            authorizeClient(clientAddresses[i]);
        }
    }
    
    // Deauthorize a client
    function deauthorizeClient(address clientAddress) public onlyOwner {
        require(clientRegistry[clientAddress].isAuthorized, "Client is not authorized");
        
        clientRegistry[clientAddress].isAuthorized = false;
        
        // Remove from authorized clients array
        // This is an inefficient implementation but works for moderate numbers of clients
        for (uint i = 0; i < authorizedClients.length; i++) {
            if (authorizedClients[i] == clientAddress) {
                // Move the last element to this position and pop the last element
                authorizedClients[i] = authorizedClients[authorizedClients.length - 1];
                authorizedClients.pop();
                break;
            }
        }
        
        emit ClientDeauthorized(clientAddress);
    }
    
    // Check if a client is authorized
    function isClientAuthorized(address clientAddress) public view returns (bool) {
        return clientRegistry[clientAddress].isAuthorized;
    }
    
    // Get all authorized clients
    function getAllAuthorizedClients() public view returns (address[] memory) {
        return authorizedClients;
    }
    
    // Record a client's contribution
    function recordContribution(
        address clientAddress,
        uint256 round,
        string memory ipfsHash,
        uint256 accuracy // Accuracy multiplied by 10000 (e.g., 95.67% = 9567)
    ) public returns (uint256) {
        // Only owner or the client itself can record a contribution
        require(msg.sender == owner || msg.sender == clientAddress, "Not authorized to record contribution");
        
        // Check if client is authorized
        require(clientRegistry[clientAddress].isAuthorized, "Client is not authorized");
        
        // Calculate contribution score (can be modified with more complex formulas)
        // Simple linear scoring: score = accuracy / 100 (to get a 0-100 scale)
        uint256 score = accuracy / 100;
        
        // Create contribution record
        ContributionRecord memory record = ContributionRecord({
            clientAddress: clientAddress,
            round: round,
            ipfsHash: ipfsHash,
            accuracy: accuracy,
            timestamp: block.timestamp,
            score: score,
            rewarded: false
        });
        
        // Add to client's contribution records
        contributionRecords[clientAddress].push(record);
        
        // Add to round contributions
        roundContributions[round].push(record);
        
        // Update client metrics
        ClientContribution storage client = clientRegistry[clientAddress];
        client.contributionCount++;
        client.totalScore += score;
        client.lastContributionTimestamp = block.timestamp;
        
        emit ContributionRecorded(clientAddress, round, score);
        
        return score;
    }
    
    // Allocate rewards to clients based on their contributions for a round
    function allocateRewardsForRound(uint256 round, uint256 totalReward) public onlyOwner {
        ContributionRecord[] storage contributions = roundContributions[round];
        require(contributions.length > 0, "No contributions for this round");
        
        // Calculate total score for this round
        uint256 totalScore = 0;
        for (uint i = 0; i < contributions.length; i++) {
            totalScore += contributions[i].score;
        }
        
        require(totalScore > 0, "Total score is zero");
        
        // Allocate rewards proportionally to each client's score
        for (uint i = 0; i < contributions.length; i++) {
            if (!contributions[i].rewarded) {
                ContributionRecord storage contribution = contributions[i];
                
                // Calculate reward amount
                uint256 rewardAmount = (contribution.score * totalReward) / totalScore;
                
                // Update contribution record
                contribution.rewarded = true;
                
                // Update client's rewards
                ClientContribution storage client = clientRegistry[contribution.clientAddress];
                client.rewardsEarned += rewardAmount;
                
                // Update total rewards allocated
                totalRewardsAllocated += rewardAmount;
                
                emit RewardAllocated(contribution.clientAddress, rewardAmount);
            }
        }
    }
    
    // Allow client to claim rewards (in a real implementation, this would transfer tokens)
    function claimRewards() public {
        ClientContribution storage client = clientRegistry[msg.sender];
        require(client.isAuthorized, "Client is not authorized");
        require(client.rewardsEarned > 0, "No rewards to claim");
        require(!client.rewardsClaimed, "Rewards already claimed");
        
        uint256 amount = client.rewardsEarned;
        client.rewardsClaimed = true;
        
        // In a real implementation, this would transfer tokens
        // For now, we just mark it as claimed
        
        emit RewardClaimed(msg.sender, amount);
    }
    
    // Get client contribution details
    function getClientContribution(address clientAddress) public view returns (
        uint256 contributionCount,
        uint256 totalScore,
        bool isAuthorized,
        uint256 lastContributionTimestamp,
        uint256 rewardsEarned,
        bool rewardsClaimed
    ) {
        ClientContribution storage client = clientRegistry[clientAddress];
        return (
            client.contributionCount,
            client.totalScore,
            client.isAuthorized,
            client.lastContributionTimestamp,
            client.rewardsEarned,
            client.rewardsClaimed
        );
    }
    
    // Get client contribution records
    function getClientContributionRecords(address clientAddress) public view returns (
        uint256[] memory rounds,
        uint256[] memory accuracies,
        uint256[] memory scores,
        uint256[] memory timestamps,
        bool[] memory rewarded
    ) {
        ContributionRecord[] storage records = contributionRecords[clientAddress];
        uint256 length = records.length;
        
        rounds = new uint256[](length);
        accuracies = new uint256[](length);
        scores = new uint256[](length);
        timestamps = new uint256[](length);
        rewarded = new bool[](length);
        
        for (uint i = 0; i < length; i++) {
            rounds[i] = records[i].round;
            accuracies[i] = records[i].accuracy;
            scores[i] = records[i].score;
            timestamps[i] = records[i].timestamp;
            rewarded[i] = records[i].rewarded;
        }
        
        return (rounds, accuracies, scores, timestamps, rewarded);
    }
    
    // Get contributions for a specific round
    function getRoundContributions(uint256 round) public view returns (
        address[] memory clients,
        uint256[] memory accuracies,
        uint256[] memory scores,
        bool[] memory rewarded
    ) {
        ContributionRecord[] storage records = roundContributions[round];
        uint256 length = records.length;
        
        clients = new address[](length);
        accuracies = new uint256[](length);
        scores = new uint256[](length);
        rewarded = new bool[](length);
        
        for (uint i = 0; i < length; i++) {
            clients[i] = records[i].clientAddress;
            accuracies[i] = records[i].accuracy;
            scores[i] = records[i].score;
            rewarded[i] = records[i].rewarded;
        }
        
        return (clients, accuracies, scores, rewarded);
    }
    
    // Update contract owner
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "New owner cannot be the zero address");
        owner = newOwner;
    }
    
    // Get total number of authorized clients
    function getAuthorizedClientCount() public view returns (uint256) {
        return authorizedClients.length;
    }
}