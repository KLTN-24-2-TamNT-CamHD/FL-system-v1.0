import React, { useState } from 'react';
import axios from 'axios';
import { Form } from '../styles';

const BlockchainInteraction = () => {
    const [transactionHash, setTransactionHash] = useState('');
    const [transactionData, setTransactionData] = useState(null);

    const handleSendTransaction = (event) => {
        event.preventDefault();
        axios.post('http://127.0.0.1:8000/send_transaction', { data: 'Sample Data' }) // Replace with your FastAPI endpoint
            .then(response => {
                setTransactionHash(response.data.transaction_hash);
            })
            .catch(error => console.error('Error sending transaction:', error));
    };

    const handleGetTransaction = (event) => {
        event.preventDefault();
        axios.get(`http://127.0.0.1:8000/get_transaction/${transactionHash}`) // Replace with your FastAPI endpoint
            .then(response => {
                setTransactionData(response.data);
            })
            .catch(error => console.error('Error getting transaction:', error));
    };

    return (
        <div>
            <h2>Blockchain Interaction</h2>
            <Form onSubmit={handleSendTransaction}>
                <button type="submit">Send Transaction</button>
            </Form>

            {transactionHash && (
                <div>
                    <p>Transaction Hash: {transactionHash}</p>
                    <Form onSubmit={handleGetTransaction}>
                        <button type="submit">Get Transaction Data</button>
                    </Form>
                </div>
            )}

            {transactionData && (
                <div>
                    <h3>Transaction Data</h3>
                    <pre>{JSON.stringify(transactionData, null, 2)}</pre>
                </div>
            )}
        </div>
    );
};

export default BlockchainInteraction;