// src/blockchain.js
import Web3 from 'web3';

const web3 = new Web3('http://127.0.0.1:7545'); // Replace with your RPC URL

// Example: Fetch transaction data
export const fetchTransactionData = async (transactionHash) => {
    try {
        const transaction = await web3.eth.getTransaction(transactionHash);
        return transaction;
    } catch (error) {
        console.error('Error fetching transaction data:', error);
        throw error;
    }
};

// Example: Fetch model hash from smart contract
export const fetchModelHash = async (contractAddress, contractABI) => {
    try {
        const contract = new web3.eth.Contract(contractABI, contractAddress);
        const modelHash = await contract.methods.getModelHash().call(); // Use .call() to read data
        return modelHash;
    } catch (error) {
        console.error('Error fetching Model Hash:', error);
        throw error;
    }
};

// Example: Send a transaction (e.g., to update the model hash)
export const updateModelHash = async (contractAddress, contractABI, newHash, accountPrivateKey) => {
    try {
        const account = web3.eth.accounts.privateKeyToAccount(accountPrivateKey);
        web3.eth.accounts.wallet.add(account);

        const contract = new web3.eth.Contract(contractABI, contractAddress);
        const gas = await contract.methods.updateModelHash(newHash).estimateGas({ from: account.address });

        const transaction = {
            from: account.address,
            to: contractAddress,
            gas: gas,
            data: contract.methods.updateModelHash(newHash).encodeABI(),
        };

        const signedTransaction = await web3.eth.accounts.signTransaction(transaction, accountPrivateKey);
        const receipt = await web3.eth.sendSignedTransaction(signedTransaction.rawTransaction);

        return receipt;
    } catch (error) {
        console.error('Error updating model hash:', error);
        throw error;
    }
};

// ... other blockchain functions ...