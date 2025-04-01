// src/api.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'; // Replace with your FastAPI URL

// Institution Management
export const registerInstitution = async (institutionData) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/api/institutions/register`, institutionData);
        return response.data;
    } catch (error) {
        console.error('Error registering institution:', error);
        throw error;
    }
};

// Training Management
export const initiateTrainingRound = async (trainingData) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/api/training/rounds/initiate`, trainingData);
        return response.data;
    } catch (error) {
        console.error('Error initiating training round:', error);
        throw error;
    }
};

export const completeTrainingRound = async (roundId) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/api/training/rounds/${roundId}/complete`);
        return response.data;
    } catch (error) {
        console.error('Error completing training round:', error);
        throw error;
    }
};

export const getTrainingRoundInfo = async (roundId) => {
    try {
        const response = await axios.get(`${API_BASE_URL}/api/training/rounds/${roundId}`);
        return response.data;
    } catch (error) {
        console.error('Error getting training round info:', error);
        throw error;
    }
};

export const submitEvaluation = async (roundId, evaluationData) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/api/training/rounds/${roundId}/evaluate`, evaluationData);
        return response.data;
    } catch (error) {
        console.error('Error submitting evaluation:', error);
        throw error;
    }
};

// Federated Learning (Flower)
export const startFLTraining = async () => {
    try {
        const response = await axios.post(`${API_BASE_URL}/api/training/fl/start`);
        return response.data;
    } catch (error) {
        console.error('Error starting FL training:', error);
        throw error;
    }
};

export const getFLStatus = async () => {
    try {
        const response = await axios.get(`${API_BASE_URL}/api/training/fl/status`);
        return response.data;
    } catch (error) {
        console.error('Error getting FL status:', error);
        throw error;
    }
};

export const stopFLTraining = async () => {
    try {
        const response = await axios.post(`${API_BASE_URL}/api/training/fl/stop`);
        return response.data;
    } catch (error) {
        console.error('Error stopping FL training:', error);
        throw error;
    }
};

// Existing API calls (adjust paths if needed)
export const startTrainingSession = async (trainingData) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/start_training`, trainingData);
        return response.data;
    } catch (error) {
        console.error('Error starting training:', error);
        throw error;
    }
};

export const getClientStatus = async () => {
    try {
        const response = await axios.get(`${API_BASE_URL}/client_status`);
        return response.data;
    } catch (error) {
        console.error('Error getting client status:', error);
        throw error;
    }
};

export const getModelMetrics = async () => {
    try {
        const response = await axios.get(`${API_BASE_URL}/model_metrics`);
        return response.data;
    } catch (error) {
        console.error('Error getting model metrics:', error);
        throw error;
    }
};

export const getBlockchainData = async () => {
    try {
        const response = await axios.get(`${API_BASE_URL}/blockchain_data`);
        return response.data;
    } catch (error) {
        console.error("Error getting blockchain data:", error);
        throw error;
    }
};