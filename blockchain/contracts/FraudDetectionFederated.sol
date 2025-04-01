
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FraudDetectionFederated {
    struct Institution {
        address addr;
        string name;
        bool authorized;
    }
    
    struct ModelUpdate {
        string ipfsHash;
        string metrics;
        uint256 timestamp;
    }
    
    struct TrainingRound {
        uint256 roundId;
        uint256 startTime;
        uint256 endTime;
        string globalModelHash;
        mapping(address => ModelUpdate) updates;
        address[] participants;
        bool completed;
    }
    
    address public owner;
    mapping(address => Institution) public institutions;
    address[] public authorizedInstitutions;
    mapping(uint256 => TrainingRound) public trainingRounds;
    uint256 public currentRound;
    
    // Events for monitoring
    event InstitutionRegistered(address indexed institution, string name);
    event TrainingRoundInitiated(uint256 indexed roundId);
    event ModelUpdateSubmitted(address indexed institution, uint256 indexed roundId, string ipfsHash);
    event GlobalModelUpdated(uint256 indexed roundId, string ipfsHash);
    event ModelEvaluationSubmitted(address indexed institution, uint256 indexed roundId, string metrics);
    
    constructor() {
        owner = msg.sender;
        currentRound = 0;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can perform this action");
        _;
    }
    
    modifier onlyAuthorized() {
        require(institutions[msg.sender].authorized, "Institution not authorized");
        _;
    }
    
    function registerInstitution(address _institution, string memory _name) public onlyOwner {
        require(!institutions[_institution].authorized, "Institution already registered");
        
        institutions[_institution] = Institution({
            addr: _institution,
            name: _name,
            authorized: true
        });
        
        authorizedInstitutions.push(_institution);
        emit InstitutionRegistered(_institution, _name);
    }
    
    function initiateTrainingRound() public onlyOwner {
        currentRound++;
        
        TrainingRound storage round = trainingRounds[currentRound];
        round.roundId = currentRound;
        round.startTime = block.timestamp;
        round.completed = false;
        
        emit TrainingRoundInitiated(currentRound);
    }
    
    function submitModelUpdate(uint256 _roundId, string memory _ipfsHash, string memory _metrics) public onlyAuthorized {
        require(_roundId == currentRound, "Invalid round ID");
        require(!trainingRounds[_roundId].completed, "Round already completed");
        
        TrainingRound storage round = trainingRounds[_roundId];
        round.updates[msg.sender] = ModelUpdate({
            ipfsHash: _ipfsHash,
            metrics: _metrics,
            timestamp: block.timestamp
        });
        
        // Add participant if not already in the list
        bool found = false;
        for (uint i = 0; i < round.participants.length; i++) {
            if (round.participants[i] == msg.sender) {
                found = true;
                break;
            }
        }
        
        if (!found) {
            round.participants.push(msg.sender);
        }
        
        emit ModelUpdateSubmitted(msg.sender, _roundId, _ipfsHash);
    }
    
    function completeRound(uint256 _roundId, string memory _globalModelHash) public onlyOwner {
        require(_roundId == currentRound, "Invalid round ID");
        require(!trainingRounds[_roundId].completed, "Round already completed");
        
        TrainingRound storage round = trainingRounds[_roundId];
        round.completed = true;
        round.globalModelHash = _globalModelHash;
        round.endTime = block.timestamp;
        
        emit GlobalModelUpdated(_roundId, _globalModelHash);
    }
    
    function submitEvaluation(uint256 _roundId, 
                             uint256 _loss, 
                             uint256 _accuracy, 
                             uint256 _auc, 
                             uint256 _precision, 
                             uint256 _recall) public onlyAuthorized {
        require(trainingRounds[_roundId].completed, "Round not completed yet");
        
        string memory metrics = string(abi.encodePacked(
            '{"loss":', _loss,
            ',"accuracy":', _accuracy,
            ',"auc":', _auc,
            ',"precision":', _precision,
            ',"recall":', _recall, '}'
        ));
        
        emit ModelEvaluationSubmitted(msg.sender, _roundId, metrics);
    }
    
    function getParticipantCount(uint256 _roundId) public view returns (uint256) {
        return trainingRounds[_roundId].participants.length;
    }
    
    function getParticipantAtIndex(uint256 _roundId, uint256 _index) public view returns (address) {
        require(_index < trainingRounds[_roundId].participants.length, "Index out of bounds");
        return trainingRounds[_roundId].participants[_index];
    }
    
    function getModelUpdateByInstitution(uint256 _roundId, address _institution) public view returns (string memory, string memory, uint256) {
        ModelUpdate storage update = trainingRounds[_roundId].updates[_institution];
        return (update.ipfsHash, update.metrics, update.timestamp);
    }
}