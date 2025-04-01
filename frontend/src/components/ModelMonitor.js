import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Table, Form } from '../styles';

const ModelMonitor = () => {
    const [modelMetrics, setModelMetrics] = useState([]);
    const [roundId, setRoundId] = useState('');
    const [evaluation, setEvaluation] = useState('');

    useEffect(() => {
        if (roundId) {
            axios.get(`/api/training/rounds/${roundId}`) // Adjusted URL
                .then(response => setModelMetrics(response.data))
                .catch(error => console.error('Error fetching round info:', error));
        }
    }, [roundId]);

    const handleEvaluation = (event) => {
        event.preventDefault();
        axios.post(`/api/training/rounds/${roundId}/evaluate`, { evaluation }).then(response => {
            console.log("Evaluation Submitted", response.data);
        }).catch(error => console.error("Error submitting evaluation", error));
    }

    return (
        <div>
            <h2>Model Monitor</h2>
            <Form>
                <label>Round ID:</label>
                <input type="text" value={roundId} onChange={(e) => setRoundId(e.target.value)} />
            </Form>
            <Table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    {Object.entries(modelMetrics).map(([key, value]) => (
                        <tr key={key}>
                            <td>{key}</td>
                            <td>{value}</td>
                        </tr>
                    ))}
                </tbody>
            </Table>
            {roundId && <Form onSubmit={handleEvaluation}>
                <label>Evaluation:</label>
                <input type="text" value={evaluation} onChange={(e) => setEvaluation(e.target.value)} />
                <button type="submit">Submit Evaluation</button>
            </Form>}
        </div>
    );
};

export default ModelMonitor;