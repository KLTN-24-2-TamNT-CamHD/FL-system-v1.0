// src/api/flService.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

export const startTraining = async (trainingConfig) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/start-training`, trainingConfig);
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};

export const stopTraining = async () => {
  try {
    const response = await axios.post(`${API_BASE_URL}/stop-training`);
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};

export const getTrainingStatus = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/training-status`);
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};

export const getServerLogs = async (numLines = 100) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/server-logs`, {
      params: { num_lines: numLines }
    });
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};

export const registerClient = async (clientData) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/register-client`, clientData);
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};

export const getSystemStatus = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/system-status`);
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};

export const getTrainingHistory = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/training-history`);
    return response.data;
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};
