import React, { useState } from 'react';
import axios from 'axios';
import { Form } from '../styles';

const TrainingSession = () => {
    const [rounds, setRounds] = useState(10);
    const [numClients, setNumClients] = useState(5);
    const [roundId, setRoundId] = useState(null);

    const handleInitiateTraining = (event) => {
        event.preventDefault();
        axios.post('/api/training/rounds/initiate', { rounds, numClients }) // Adjusted URL
            .then(response => {
                console.log('Training initiated:', response.data);
                setRoundId(response.data.round_id); // Assuming the API returns round_id
            })
            .catch(error => console.error('Error initiating training:', error));
    };

    const handleCompleteRound = (event) => {
        event.preventDefault();
        axios.post(`/api/training/rounds/${roundId}/complete`).then(response => {
            console.log("Round completed", response.data);
        }).catch(error => console.error("Error completing round", error));
    }

    return (
        <div>
            <h2>Start Training Session</h2>
            <Form onSubmit={handleInitiateTraining}>
                <label>Training Rounds:</label>
                <input type="number" value={rounds} onChange={(e) => setRounds(e.target.value)} />
                <label>Number of Clients:</label>
                <input type="number" value={numClients} onChange={(e) => setNumClients(e.target.value)} />
                <button type="submit">Initiate Training</button>
            </Form>
            {roundId && <div>
                <p>Round ID: {roundId}</p>
                <button onClick={handleCompleteRound}>Complete Round</button>
            </div>}
        </div>
    );
};

export default TrainingSession;