// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Federation {
    // Gas-optimized Model structure with smaller uint types
    struct Model {
        string ipfsHash;
        uint32 round;
        string version;
        uint64 timestamp;
        uint32 participatingClients;
        address publisher;
        bool isActive;
    }
    
    // Gas-optimized Client contribution structure
    struct ClientContribution {
        uint32 contributionCount;
        uint96 totalScore;
        bool isAuthorized;
        uint64 lastContributionTimestamp;
        uint96 rewardsEarned;
        bool rewardsClaimed;
    }
    
    // Gas-optimized Contribution record structure
    struct ContributionRecord {
        address clientAddress;
        uint32 round;
        string ipfsHash;
        uint32 accuracy;
        uint64 timestamp;
        uint32 score;
        bool rewarded;
    }
    
    // Gas-optimized Round reward pool structure
    struct RoundRewardPool {
        uint128 totalAmount;
        uint128 allocatedAmount;
        bool isFinalized;
    }
    
    // Main storage for models
    mapping(bytes32 => Model) public models;
    
    // Model IDs by round
    mapping(uint32 => bytes32[]) public modelsByRound;
    
    // Latest model ID for each version prefix
    mapping(string => bytes32) public latestVersions;

    // Track total number of models
    uint256 public totalModelCount;

    // Track all model IDs in an array for easier access
    bytes32[] public allModelIds;
    
    // Client registry
    mapping(address => ClientContribution) public clientRegistry;
    
    // Authorized client addresses
    address[] public authorizedClients;
    
    // Contribution records by client address - using mapping instead of array
    mapping(address => mapping(uint32 => ContributionRecord)) public clientContributions;
    mapping(address => uint32) public clientContributionCount;
    
    // Contributions by round - using mapping instead of array for gas efficiency
    mapping(uint32 => mapping(uint32 => ContributionRecord)) public roundContributions;
    mapping(uint32 => uint32) public roundContributionCount;
    
    // Reward pools by round
    mapping(uint32 => RoundRewardPool) public roundRewardPools;
    
    // Total rewards allocated
    uint128 public totalRewardsAllocated;
    
    // Total rewards claimed
    uint128 public totalRewardsClaimed;
    
    // Current contract balance
    uint128 public contractBalance;
    
    // Events
    event ModelRegistered(bytes32 indexed modelId, string ipfsHash, uint32 round, string version);
    event ModelUpdated(bytes32 indexed modelId, string ipfsHash, uint32 round, string version);
    event ModelDeactivated(bytes32 indexed modelId);
    event ClientAuthorized(address indexed clientAddress);
    event ClientDeauthorized(address indexed clientAddress);
    event ContributionRecorded(address indexed clientAddress, uint32 round, uint32 score);
    event RewardAllocated(address indexed clientAddress, uint128 amount, uint32 round);
    event RewardClaimed(address indexed clientAddress, uint128 amount);
    event RewardPoolFunded(uint32 indexed round, uint128 amount);
    event RewardPoolFinalized(uint32 indexed round, uint128 totalAmount);
    event ContractFunded(address indexed from, uint128 amount);
    event EmergencyWithdrawal(address indexed to, uint128 amount);
    
    // Owner of the contract
    address public owner;
    
    constructor() {
        owner = msg.sender;
    }
    
    // Modifier to restrict certain functions to the owner
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    // Receive function to allow the contract to receive ETH
    receive() external payable {
        contractBalance += uint128(msg.value);
        emit ContractFunded(msg.sender, uint128(msg.value));
    }
    
    // Fallback function
    fallback() external payable {
        contractBalance += uint128(msg.value);
        emit ContractFunded(msg.sender, uint128(msg.value));
    }
    
    // Function to fund the contract
    function fundContract() external payable {
        contractBalance += uint128(msg.value);
        emit ContractFunded(msg.sender, uint128(msg.value));
    }
    
    // Function to fund a specific round's reward pool
    function fundRoundRewardPool(uint32 round) external payable onlyOwner {
        require(msg.value > 0, "Zero funding");
        
        RoundRewardPool storage pool = roundRewardPools[round];
        pool.totalAmount += uint128(msg.value);
        contractBalance += uint128(msg.value);
        
        emit RewardPoolFunded(round, uint128(msg.value));
    }
    
    // Finalize a round's reward pool (prevents additional funding)
    function finalizeRoundRewardPool(uint32 round) external onlyOwner {
        RoundRewardPool storage pool = roundRewardPools[round];
        require(pool.totalAmount > 0, "Empty pool");
        require(!pool.isFinalized, "Already finalized");
        
        pool.isFinalized = true;
        
        emit RewardPoolFinalized(round, pool.totalAmount);
    }
    
    // Generate a unique model ID from the IPFS hash and round
    function generateModelId(string calldata ipfsHash, uint32 round) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(ipfsHash, round));
    }
    
    //////////////////////////////////////////////////////////
    ///////////   MODEL MANAGEMENT FUNCTIONS  ////////////////
    //////////////////////////////////////////////////////////
    
    // Register a new model - using calldata instead of memory for parameters
    function registerModel(
        string calldata ipfsHash, 
        uint32 round, 
        string calldata version,
        uint32 participatingClients
    ) external returns (bytes32) {
        // Generate model ID
        bytes32 modelId = generateModelId(ipfsHash, round);
        
        // Ensure it doesn't already exist
        require(models[modelId].timestamp == 0, "Model exists");
        
        // Store the model
        models[modelId] = Model({
            ipfsHash: ipfsHash,
            round: round,
            version: version,
            timestamp: uint64(block.timestamp),
            participatingClients: participatingClients,
            publisher: msg.sender,
            isActive: true
        });
        
        // Add to round mapping
        modelsByRound[round].push(modelId);
        
        // Update latest version
        string memory versionPrefix = getVersionPrefix(version);
        
        allModelIds.push(modelId);
        totalModelCount++;

        // If there's no latest version for this prefix or this one is newer
        if (latestVersions[versionPrefix] == bytes32(0) || 
            compareVersions(version, models[latestVersions[versionPrefix]].version)) {
            latestVersions[versionPrefix] = modelId;
        }
        
        // Emit event
        emit ModelRegistered(modelId, ipfsHash, round, version);
        
        return modelId;
    }
    
    // Update an existing model by round
    function updateModelByRound(
        string calldata ipfsHash, 
        uint32 round, 
        string calldata version,
        uint32 participatingClients
    ) external onlyOwner returns (bytes32) {
        // Check if any model exists for this round
        require(modelsByRound[round].length > 0, "No models for round");
        
        // Generate new model ID
        bytes32 modelId = generateModelId(ipfsHash, round);
        
        // Check if this specific model already exists
        if (models[modelId].timestamp > 0) {
            // Model already exists
            return modelId;
        }
        
        // Store the model
        models[modelId] = Model({
            ipfsHash: ipfsHash,
            round: round,
            version: version,
            timestamp: uint64(block.timestamp),
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
            compareVersions(version, models[latestVersions[versionPrefix]].version)) {
            latestVersions[versionPrefix] = modelId;
        }
        
        // Emit event
        emit ModelUpdated(modelId, ipfsHash, round, version);
        
        return modelId;
    }

    // Get the latest model for a specific version prefix
    function getLatestModel(string calldata versionPrefix) external view returns (
        bytes32 modelId,
        string memory ipfsHash,
        uint32 round,
        string memory version,
        uint64 timestamp,
        uint32 participatingClients
    ) {
        bytes32 id = latestVersions[versionPrefix];
        require(id != bytes32(0), "No models for version");
        
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
    function getModelsByRound(uint32 round) external view returns (bytes32[] memory) {
        return modelsByRound[round];
    }
    
    // Get the most recent model for a specific round
    function getLatestModelByRound(uint32 round) external view returns (
        bytes32 modelId,
        string memory ipfsHash,
        string memory version,
        uint64 timestamp,
        uint32 participatingClients,
        address publisher,
        bool isActive
    ) {
        require(modelsByRound[round].length > 0, "No models for round");
        
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
    function getModelDetails(bytes32 modelId) external view returns (
        string memory ipfsHash,
        uint32 round,
        string memory version,
        uint64 timestamp,
        uint32 participatingClients,
        address publisher,
        bool isActive
    ) {
        Model storage model = models[modelId];
        require(model.timestamp > 0, "Model not exist");
        
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

    // Get models with pagination
    // This function is optimized for gas by using arrays instead of mappings
    function getModels(uint256 offset, uint256 limit) external view returns (
        bytes32[] memory modelIds,
        string[] memory ipfsHashes,
        uint32[] memory rounds,
        uint64[] memory timestamps,
        uint32[] memory participatingClients,
        bool[] memory isActive
    ) {
        // Calculate actual number of records to return
        uint256 count = (offset >= totalModelCount) ? 0 : (
            ((offset + limit) > totalModelCount) ? (totalModelCount - offset) : limit
        );
        
        // Initialize arrays
        modelIds = new bytes32[](count);
        ipfsHashes = new string[](count);
        rounds = new uint32[](count);
        timestamps = new uint64[](count);
        participatingClients = new uint32[](count);
        isActive = new bool[](count);
        
        // Fill arrays with model data
        for (uint256 i = 0; i < count; i++) {
            bytes32 modelId = allModelIds[offset + i];
            Model storage model = models[modelId];
            
            modelIds[i] = modelId;
            ipfsHashes[i] = model.ipfsHash;
            rounds[i] = model.round;
            timestamps[i] = model.timestamp;
            participatingClients[i] = model.participatingClients;
            isActive[i] = model.isActive;
        }
        
        return (modelIds, ipfsHashes, rounds, timestamps, participatingClients, isActive);
    }

    // Get total model count
    function getModelCount() external view returns (uint256) {
        return totalModelCount;
    }
    // Function to get all models for visualization
    function getAllModels() external view returns (
        bytes32[] memory modelIds,
        string[] memory ipfsHashes,
        uint32[] memory rounds,
        uint64[] memory timestamps,
        uint32[] memory participatingClients,
        bool[] memory isActive
    ) {
        // First, determine how many total models we have by counting all rounds
        uint256 totalModels = 0;
        uint32 maxRound = 0;
        
        // Find the maximum round number
        for (uint32 i = 0; i < 1000; i++) { // Assume maximum 1000 rounds for safety
            if (modelsByRound[i].length > 0) {
                totalModels += modelsByRound[i].length;
                if (i > maxRound) maxRound = i;
            }
        }
        
        // Initialize arrays with the correct size
        modelIds = new bytes32[](totalModels);
        ipfsHashes = new string[](totalModels);
        rounds = new uint32[](totalModels);
        timestamps = new uint64[](totalModels);
        participatingClients = new uint32[](totalModels);
        isActive = new bool[](totalModels);
        
        // Fill arrays with model data
        uint256 index = 0;
        
        // Iterate through all rounds
        for (uint32 round = 0; round <= maxRound; round++) {
            bytes32[] memory roundModels = modelsByRound[round];
            
            for (uint256 j = 0; j < roundModels.length; j++) {
                bytes32 modelId = roundModels[j];
                Model storage model = models[modelId];
                
                modelIds[index] = modelId;
                ipfsHashes[index] = model.ipfsHash;
                rounds[index] = model.round;
                timestamps[index] = model.timestamp;
                participatingClients[index] = model.participatingClients;
                isActive[index] = model.isActive;
                
                index++;
            }
        }
        
        return (modelIds, ipfsHashes, rounds, timestamps, participatingClients, isActive);
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
        bytes memory b1 = bytes(v1);
        bytes memory b2 = bytes(v2);
        
        for (uint i = 0; i < b1.length && i < b2.length; i++) {
            if (b1[i] != b2[i]) {
                return b1[i] > b2[i];
            }
        }
        
        return b1.length > b2.length;
    }
    
    // Deactivate a model (e.g., if it's found to be problematic)
    function deactivateModel(bytes32 modelId) external onlyOwner {
        require(models[modelId].timestamp > 0, "Model not exist");
        require(models[modelId].isActive, "Already inactive");
        
        models[modelId].isActive = false;
        
        emit ModelDeactivated(modelId);
    }
    
    //////////////////////////////////////////////////////////
    ///////////   CLIENT MANAGEMENT FUNCTIONS  ///////////////
    //////////////////////////////////////////////////////////
    
    // Authorize a client to participate in federated learning
    function authorizeClient(address clientAddress) public onlyOwner {
        require(clientAddress != address(0), "Invalid address");
        
        ClientContribution storage client = clientRegistry[clientAddress];
        
        if (!client.isAuthorized) {
            client.isAuthorized = true;
            authorizedClients.push(clientAddress);
            
            emit ClientAuthorized(clientAddress);
        }
    }
    
    // Authorize multiple clients at once
    function authorizeClients(address[] calldata clientAddresses) external onlyOwner {
        for (uint i = 0; i < clientAddresses.length; i++) {
            authorizeClient(clientAddresses[i]);
        }
    }
    
    // Deauthorize a client - more gas efficient implementation
    function deauthorizeClient(address clientAddress) external onlyOwner {
        require(clientRegistry[clientAddress].isAuthorized, "Not authorized");
        
        clientRegistry[clientAddress].isAuthorized = false;
        
        // Find and remove from authorized clients array using a more gas efficient approach
        uint256 length = authorizedClients.length;
        for (uint i = 0; i < length; i++) {
            if (authorizedClients[i] == clientAddress) {
                // Swap with the last element and pop
                authorizedClients[i] = authorizedClients[length - 1];
                authorizedClients.pop();
                break;
            }
        }
        
        emit ClientDeauthorized(clientAddress);
    }
    
    // Check if a client is authorized
    function isClientAuthorized(address clientAddress) external view returns (bool) {
        return clientRegistry[clientAddress].isAuthorized;
    }
    
    // Get all authorized clients
    function getAllAuthorizedClients() external view returns (address[] memory) {
        return authorizedClients;
    }
    
    //////////////////////////////////////////////////////////
    ///////////   CONTRIBUTION MANAGEMENT  //////////////////
    //////////////////////////////////////////////////////////
    
    // Record a client's contribution - more gas efficient
    function recordContribution(
        address clientAddress,
        uint32 round,
        string calldata ipfsHash,
        uint32 accuracy // Accuracy multiplied by 10000 (e.g., 95.67% = 9567)
    ) external returns (uint32) {
        // Only owner or the client itself can record a contribution
        require(msg.sender == owner || msg.sender == clientAddress, "Not authorized");
        
        // Check if client is authorized
        require(clientRegistry[clientAddress].isAuthorized, "Not authorized");
        
        // Calculate contribution score (simplified for gas)
        uint32 score = accuracy / 100;
        
        // Create contribution record using mappings instead of arrays
        uint32 clientIndex = clientContributionCount[clientAddress];
        uint32 roundIndex = roundContributionCount[round];
        
        // Store in client contributions mapping
        clientContributions[clientAddress][clientIndex] = ContributionRecord({
            clientAddress: clientAddress,
            round: round,
            ipfsHash: ipfsHash,
            accuracy: accuracy,
            timestamp: uint64(block.timestamp),
            score: score,
            rewarded: false
        });
        
        // Store in round contributions mapping
        roundContributions[round][roundIndex] = ContributionRecord({
            clientAddress: clientAddress,
            round: round,
            ipfsHash: ipfsHash,
            accuracy: accuracy,
            timestamp: uint64(block.timestamp),
            score: score,
            rewarded: false
        });
        
        // Increment counters
        clientContributionCount[clientAddress]++;
        roundContributionCount[round]++;
        
        // Update client metrics
        ClientContribution storage client = clientRegistry[clientAddress];
        client.contributionCount++;
        client.totalScore += score;
        client.lastContributionTimestamp = uint64(block.timestamp);
        
        emit ContributionRecorded(clientAddress, round, score);
        
        return score;
    }
    
    // Get client contribution details
    function getClientContribution(address clientAddress) external view returns (
        uint32 contributionCount,
        uint96 totalScore,
        bool isAuthorized,
        uint64 lastContributionTimestamp,
        uint96 rewardsEarned,
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
    
    // Get client contribution records - with pagination for gas efficiency
    function getClientContributionRecords(
        address clientAddress, 
        uint32 offset, 
        uint32 limit
    ) external view returns (
        uint32[] memory rounds,
        uint32[] memory accuracies,
        uint32[] memory scores,
        uint64[] memory timestamps,
        bool[] memory rewarded
    ) {
        uint32 total = clientContributionCount[clientAddress];
        
        // Calculate actual number of records to return
        uint32 count = (offset >= total) ? 0 : (
            ((offset + limit) > total) ? (total - offset) : limit
        );
        
        // Initialize arrays
        rounds = new uint32[](count);
        accuracies = new uint32[](count);
        scores = new uint32[](count);
        timestamps = new uint64[](count);
        rewarded = new bool[](count);
        
        // Populate arrays
        for (uint32 i = 0; i < count; i++) {
            ContributionRecord storage record = clientContributions[clientAddress][offset + i];
            rounds[i] = record.round;
            accuracies[i] = record.accuracy;
            scores[i] = record.score;
            timestamps[i] = record.timestamp;
            rewarded[i] = record.rewarded;
        }
        
        return (rounds, accuracies, scores, timestamps, rewarded);
    }
    
    // Get contributions for a specific round - with pagination
    function getRoundContributions(
        uint32 round,
        uint32 offset,
        uint32 limit
    ) external view returns (
        address[] memory clients,
        uint32[] memory accuracies,
        uint32[] memory scores,
        bool[] memory rewarded
    ) {
        uint32 total = roundContributionCount[round];
        
        // Calculate actual number of records to return
        uint32 count = (offset >= total) ? 0 : (
            ((offset + limit) > total) ? (total - offset) : limit
        );
        
        // Initialize arrays
        clients = new address[](count);
        accuracies = new uint32[](count);
        scores = new uint32[](count);
        rewarded = new bool[](count);
        
        // Populate arrays
        for (uint32 i = 0; i < count; i++) {
            ContributionRecord storage record = roundContributions[round][offset + i];
            clients[i] = record.clientAddress;
            accuracies[i] = record.accuracy;
            scores[i] = record.score;
            rewarded[i] = record.rewarded;
        }
        
        return (clients, accuracies, scores, rewarded);
    }
    
    //////////////////////////////////////////////////////////
    ///////////   REWARD MANAGEMENT FUNCTIONS  ///////////////
    //////////////////////////////////////////////////////////
    
    // Batch allocate rewards to clients - more gas efficient
    function allocateRewardsForRound(uint32 round) external onlyOwner {
        uint32 count = roundContributionCount[round];
        require(count > 0, "No contributions");
        
        RoundRewardPool storage pool = roundRewardPools[round];
        require(pool.totalAmount > 0, "No rewards");
        require(pool.isFinalized, "Not finalized");
        
        // Calculate in a separate function
        (uint256 totalScore, uint32 unrewarded) = calculateRoundScores(round, count);
        
        require(totalScore > 0, "Zero total score");
        require(unrewarded > 0, "All rewarded");
        
        uint256 availableRewards = pool.totalAmount - pool.allocatedAmount;
        require(availableRewards > 0, "No available rewards");
        
        // Process batches using a separate function
        processRewardBatches(round, count, unrewarded, totalScore, availableRewards, pool);
    }

    // Extract calculations to reduce variable count
    function calculateRoundScores(uint32 round, uint32 count) internal view returns (uint256 totalScore, uint32 unrewarded) {
        totalScore = 0;
        unrewarded = 0;
        
        for (uint32 i = 0; i < count; i++) {
            ContributionRecord storage contribution = roundContributions[round][i];
            if (!contribution.rewarded) {
                totalScore += contribution.score;
                unrewarded++;
            }
        }
        
        return (totalScore, unrewarded);
    }

    // Process rewards in a separate function
    function processRewardBatches(
        uint32 round, 
        uint32 count, 
        uint32 unrewarded, 
        uint256 totalScore, 
        uint256 availableRewards,
        RoundRewardPool storage pool
    ) internal {
        uint32 batchSize = 10;
        uint32 processedCount = 0;
        
        while (processedCount < unrewarded && processedCount < count) {
            uint32 currentBatch = (unrewarded - processedCount) < batchSize ? 
                                (unrewarded - processedCount) : batchSize;
            
            processRewardBatch(round, count, currentBatch, totalScore, availableRewards, pool);
            processedCount += currentBatch;
        }
    }

    // Process a single batch
    function processRewardBatch(
        uint32 round,
        uint32 count,
        uint32 batchSize,
        uint256 totalScore,
        uint256 availableRewards,
        RoundRewardPool storage pool
    ) internal {
        uint32 batchProcessed = 0;
        
        for (uint32 i = 0; i < count && batchProcessed < batchSize; i++) {
            ContributionRecord storage contribution = roundContributions[round][i];
            
            if (!contribution.rewarded) {
                processContributionReward(contribution, round, totalScore, availableRewards, pool);
                batchProcessed++;
            }
        }
    }

    // Process a single contribution's reward
    function processContributionReward(
        ContributionRecord storage contribution,
        uint32 round,
        uint256 totalScore,
        uint256 availableRewards,
        RoundRewardPool storage pool
    ) internal {
        // Calculate reward amount
        uint128 rewardAmount = uint128((uint256(contribution.score) * availableRewards) / totalScore);
        
        // Update contribution record
        contribution.rewarded = true;
        
        // Update reward pool
        pool.allocatedAmount += rewardAmount;
        contractBalance -= rewardAmount;
        
        // Transfer ETH directly to the client
        (bool success, ) = payable(contribution.clientAddress).call{value: rewardAmount}("");
        require(success, "Transfer failed");
        
        // Emit event with fewer variables in scope
        emit RewardAllocated(contribution.clientAddress, rewardAmount, round);
    }
    
    // Allow client to claim rewards (transfers actual ETH)
    function claimRewards() external {
        ClientContribution storage client = clientRegistry[msg.sender];
        require(client.isAuthorized, "Not authorized");
        require(client.rewardsEarned > 0, "No rewards");
        require(!client.rewardsClaimed, "Already claimed");
        
        uint128 amount = client.rewardsEarned;
        require(amount <= contractBalance, "Insufficient balance");
        
        // Mark rewards as claimed before transfer to prevent reentrancy
        client.rewardsClaimed = true;
        contractBalance -= amount;
        totalRewardsClaimed += amount;
        
        // Transfer ETH to the client
        (bool success, ) = payable(msg.sender).call{value: amount}("");
        require(success, "Transfer failed");
        
        emit RewardClaimed(msg.sender, amount);
    }
    
    // Get available rewards for a client
    function getAvailableRewards(address clientAddress) external view returns (uint96) {
        ClientContribution storage client = clientRegistry[clientAddress];
        if (!client.isAuthorized || client.rewardsClaimed) {
            return 0;
        }
        return client.rewardsEarned;
    }
    
    // Emergency withdrawal function for the owner
    function emergencyWithdraw(uint128 amount) external onlyOwner {
        require(amount <= contractBalance, "Insufficient balance");
        
        contractBalance -= amount;
        
        // Transfer ETH to the owner
        (bool success, ) = payable(owner).call{value: amount}("");
        require(success, "Transfer failed");
        
        emit EmergencyWithdrawal(owner, amount);
    }
    
    // Get round reward pool details
    function getRoundRewardPool(uint32 round) external view returns (
        uint128 totalAmount,
        uint128 allocatedAmount,
        uint128 remainingAmount,
        bool isFinalized
    ) {
        RoundRewardPool storage pool = roundRewardPools[round];
        return (
            pool.totalAmount,
            pool.allocatedAmount,
            pool.totalAmount - pool.allocatedAmount,
            pool.isFinalized
        );
    }
    
    // Update contract owner
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Zero address");
        owner = newOwner;
    }
    
    // Get total number of authorized clients
    function getAuthorizedClientCount() external view returns (uint256) {
        return authorizedClients.length;
    }
}