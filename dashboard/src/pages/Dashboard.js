// src/pages/Dashboard.js
import React, { useState, useEffect } from 'react';
import { getTrainingStatus, getSystemStatus, getServerLogs } from '../api/flService';
import TrainingStatusCard from '../components/TrainingStatusCard';
import SystemStatusCard from '../components/SystemStatusCard';
import LogsViewer from '../components/LogsViewer';
import './Dashboard.css';

const Dashboard = () => {
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [systemStatus, setSystemStatus] = useState(null);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Fetch data in parallel
        const [statusRes, systemRes, logsRes] = await Promise.all([
          getTrainingStatus(),
          getSystemStatus(),
          getServerLogs(20)
        ]);
        
        setTrainingStatus(statusRes);
        setSystemStatus(systemRes);
        setLogs(logsRes.logs);
        setError(null);
      } catch (err) {
        setError('Failed to fetch dashboard data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    
    // Set up polling interval
    const intervalId = setInterval(fetchData, 5000);
    
    return () => clearInterval(intervalId);
  }, []);

  if (loading && !trainingStatus) {
    return <div className="loading">Loading dashboard data...</div>;
  }

  if (error) {
    return <div className="error-message">{error}</div>;
  }

  return (
    <div className="dashboard">
      <h1>Federated Learning Dashboard</h1>
      
      <div className="dashboard-grid">
        <div className="dashboard-column">
          <TrainingStatusCard status={trainingStatus} />
          <SystemStatusCard status={systemStatus} />
        </div>
        <div className="dashboard-column">
          <div className="card logs-card">
            <h2>Recent Logs</h2>
            <LogsViewer logs={logs} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
