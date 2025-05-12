# Federated Learning with GA-Stacking, Blockchain and IPFS

This project implements an enhanced federated learning system for credit card fraud detection that leverages Genetic Algorithm-based Stacking (GA-Stacking) ensemble methods, blockchain for client authorization and contribution tracking, and IPFS for distributed model storage.

## Overview

The system consists of a federated learning architecture where:

1. **Server** coordinates training across clients, aggregates model updates, and manages the global model
2. **Clients** train local ensemble models and optimize them using GA-Stacking
3. **IPFS** stores and distributes model parameters
4. **Blockchain** provides client authentication, tracks contributions, and distributes rewards

The main innovation is the combination of ensemble learning with genetic algorithms (GA-Stacking) to optimize model weights, rewarding clients based on their contribution quality.

## System Architecture

![System Architecture](architecture_diagram.png)

### Components

- **EnhancedFedAvgWithGA**: Server-side federated averaging strategy with GA-Stacking support
- **GAStackingClient**: Client implementation with ensemble model training capabilities
- **IPFSConnector**: Interface for storing and retrieving models from IPFS
- **BlockchainConnector**: Interface for blockchain interactions
- **EnsembleAggregator**: Handles aggregation of ensemble models from clients
- **GAStackingRewardSystem**: Blockchain-based system for tracking and rewarding client contributions

## Features

- **GA-Stacking Ensemble Optimization**: Uses genetic algorithms to find optimal ensemble weights
- **Blockchain Integration**: 
  - Client authorization
  - Contribution tracking
  - Merit-based reward distribution
- **IPFS Model Storage**: Distributed storage for model parameters
- **Fraud Detection Specialization**: Models and parameters optimized for fraud detection tasks
- **Reward System**: Incentivizes high-quality client contributions

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Flower (Federated Learning framework)
- IPFS daemon
- Ganache (for local blockchain)
- Web3.py

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/federated-ga-stacking.git
cd federated-ga-stacking

# Install dependencies
pip install -r requirements.txt

# Start IPFS daemon
ipfs daemon

# Start Ganache (for local blockchain)
ganache-cli
```

### Running the Server

```bash
python server.py --server-address 0.0.0.0:8088 \
                 --rounds 5 \
                 --min-fit-clients 2 \
                 --ipfs-url http://127.0.0.1:5001/api/v0 \
                 --ganache-url http://127.0.0.1:7545 \
                 --deploy-contract \
                 --authorized-clients-only
```

### Running Clients

```bash
python client.py --server-address 127.0.0.1:8088 \
                 --ipfs-url http://127.0.0.1:5001/api/v0 \
                 --ganache-url http://127.0.0.1:7545 \
                 --wallet-address 0x123... \
                 --private-key 0xabc... \
                 --client-id client-1
```

## Key Files

- `server.py`: Implementation of the federated learning server
- `client.py`: Implementation of GA-Stacking client
- `ipfs_connector.py`: Interface to IPFS
- `blockchain_connector.py`: Interface to blockchain
- `ensemble_aggregation.py`: Logic for aggregating ensemble models
- `ga_stacking.py`: Implementation of GA-Stacking algorithm
- `ga_stacking_reward_system.py`: Reward system for client contributions
- `base_models.py`: Implementation of various base models for the ensemble

## Project Structure

```
v2-fl/
├── server.py               # FL server implementation
├── client.py               # GA-Stacking client
├── ipfs_connector.py       # IPFS interface
├── blockchain_connector.py # Blockchain interface
├── ensemble_aggregation.py # Ensemble aggregation logic
├── ga_stacking.py          # GA-Stacking implementation
├── base_models.py          # Base models for ensemble
├── metrics/                # Metrics storage directory
└── contract/               # Blockchain smart contracts
```

## How It Works

### GA-Stacking Process

1. **Base Model Training**: Each client trains individual base models on local data
2. **Meta-Learner Training**: Meta-learners are trained on base model predictions
3. **Genetic Optimization**: GA finds optimal ensemble weights to maximize performance
4. **Ensemble Evaluation**: Final ensemble is evaluated on test data
5. **Model Sharing**: Models are stored in IPFS and references shared

### Federated Learning Workflow

1. **Server Initialization**: Server initializes the global ensemble model
2. **Client Selection**: Server selects clients for the current round
3. **Local Training**: Clients train and optimize local ensembles using GA-Stacking
4. **Model Aggregation**: Server aggregates client models into a new global model
5. **Evaluation**: Global model is evaluated on client test data
6. **Reward Distribution**: Contributions are assessed and rewards distributed via blockchain

## Smart Contract

The blockchain smart contract provides:

- **Client Authorization**: Only authorized clients can participate
- **Model Registration**: Global models are registered on-chain
- **Contribution Tracking**: Client contributions are tracked with quality metrics
- **Reward System**: Tokens/ETH are distributed based on contribution quality

## Advanced Configuration

### Server Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--server-address` | Server address (host:port) | 0.0.0.0:8088 |
| `--rounds` | Number of federated learning rounds | 3 |
| `--min-fit-clients` | Minimum number of clients for training | 2 |
| `--min-evaluate-clients` | Minimum number of clients for evaluation | 2 |
| `--ipfs-url` | IPFS API URL | http://127.0.0.1:5001/api/v0 |
| `--ganache-url` | Ganache blockchain URL | http://127.0.0.1:7545 |
| `--contract-address` | Federation contract address | None |
| `--deploy-contract` | Deploy a new contract if address not provided | False |
| `--authorized-clients-only` | Only accept contributions from authorized clients | True |
| `--round-rewards` | Reward points to distribute each round | 1000 |
| `--device` | Device to use for computation | cpu |

### Client Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--server-address` | Server address (host:port) | 127.0.0.1:8088 |
| `--ipfs-url` | IPFS API URL | http://127.0.0.1:5001/api/v0 |
| `--ganache-url` | Ganache blockchain URL | http://127.0.0.1:7545 |
| `--contract-address` | Federation contract address | None |
| `--wallet-address` | Client's Ethereum wallet address | None |
| `--private-key` | Client's private key | None |
| `--client-id` | Client identifier | None |
| `--input-dim` | Input dimension for the model | 10 |
| `--output-dim` | Output dimension for the model | 1 |
| `--ensemble-size` | Number of models in the ensemble | 5 |
| `--device` | Device for training | cpu |
| `--ga-generations` | Number of GA generations to run | 20 |
| `--ga-population-size` | Size of GA population | 30 |

## Metrics and Evaluation

The system tracks various metrics:

- Ensemble accuracy
- Model diversity
- Generalization score
- Convergence rate
- Individual base model performance
- Final weighted score

These metrics are used both for evaluating model performance and determining client rewards.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Flower](https://flower.dev/) - Federated Learning framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [IPFS](https://ipfs.io/) - Distributed file system
- [Web3.py](https://web3py.readthedocs.io/) - Ethereum interface
- [Ganache](https://www.trufflesuite.com/ganache) - Local Ethereum blockchain
