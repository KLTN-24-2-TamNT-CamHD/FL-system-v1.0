// src/utils.js
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000', // Set your FastAPI server URL here
});

export default api;