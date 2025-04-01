import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Form } from '../styles';

const FLTraining = () => {
    const [flStatus, setFLStatus] = useState(null);
    const [error, setError] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        fetchFLStatus(); // Fetch status on component mount
    }, []);

    const fetchFLStatus = () => {
        setLoading(true);
        axios
            .get('http://localhost:8000/api/training/fl/status')
            .then((response) => {
                setFLStatus(response.data);
                setError(null);
            })
            .catch((err) => {
                setError(err.message || 'Failed to fetch FL status');
                setFLStatus(null);
            })
            .finally(() => setLoading(false));
    };

    const handleStartFLTraining = () => {
        setLoading(true);
        axios
            .post('http://localhost:8000/api/training/fl/start')
            .then(() => {
                fetchFLStatus(); // Refresh status after starting
            })
            .catch((err) => {
                setError(err.message || 'Failed to start FL training');
            })
            .finally(() => setLoading(false));
    };

    const handleStopFLTraining = () => {
        setLoading(true);
        axios
            .post('http://localhost:8000/api/training/fl/stop')
            .then(() => {
                fetchFLStatus(); // Refresh status after stopping
            })
            .catch((err) => {
                setError(err.message || 'Failed to stop FL training');
            })
            .finally(() => setLoading(false));
    };

    return (
        <div>
            <h2>Federated Learning Training</h2>
            {loading && <p>Loading...</p>}
            {error && <p style={{ color: 'red' }}>{error}</p>}
            {flStatus && (
                <div>
                    <h3>FL Status</h3>
                    <pre>{JSON.stringify(flStatus, null, 2)}</pre>
                </div>
            )}
            <Form>
                <button type="button" onClick={handleStartFLTraining}>
                    Start FL Training
                </button>
                <button type="button" onClick={handleStopFLTraining}>
                    Stop FL Training
                </button>
            </Form>
        </div>
    );
};

export default FLTraining;