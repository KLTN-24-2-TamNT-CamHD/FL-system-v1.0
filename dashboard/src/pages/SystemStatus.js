// src/pages/SystemStatus.js
import React, { useState, useEffect } from 'react';
import { getSystemStatus, getServerLogs } from '../api/flService';
import LogsViewer from '../components/LogsViewer';
import './SystemStatus.css';

const SystemStatus = () => {
  const [systemStatus, setSystemStatus] = useState(null);
  const [logs, setLogs] = useState([]);
  const [numLogLines, setNumLogLines] = useState(100);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Fetch status and logs
        const [statusRes, logsRes] = await Promise.all([
          getSystemStatus(),
          getServerLogs(numLogLines)
        ]);
        
        setSystemStatus(statusRes);
        setLogs(logsRes.logs);
        setError(null);
      } catch (err) {
        setError('Failed to fetch system status data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    
    // Set up polling interval
    const intervalId = setInterval(fetchData, 10000);
    
    return () => clearInterval(intervalId);
  }, [numLogLines]);

  const handleRefreshLogs = async () => {
    try {
      const logsRes = await getServerLogs(numLogLines);
      setLogs(logsRes.logs);
    } catch (err) {
      console.error('Failed to refresh logs:', err);
    }
  };

  if (loading && !systemStatus) {
    return <div className="loading">Loading system status...</div>;
  }

  if (error) {
    return <div className="error-message">{error}</div>;
  }

  return (
    <div className="system-status-page">
      <h1>System Status</h1>
      
      <div className="card status-overview">
        <h2>System Overview</h2>
        
        {systemStatus && (
          <div className="status-details">
            <div className="status-item">
              <span className="status-label">Flower Server:</span>
              <span className={`status-value ${systemStatus.flower_status.is_running ? 'active' : 'inactive'}`}>
                {systemStatus.flower_status.is_running ? 'Running' : 'Stopped'}
              </span>
            </div>
            
            {systemStatus.flower_status.is_running && (
              <div className="status-item">
                <span className="status-label">Current Training:</span>
                <span className="status-value">{systemStatus.flower_status.current_training_id}</span>
              </div>
            )}
            
            <div className="status-item">
              <span className="status-label">Latest Training:</span>
              <span className="status-value">{systemStatus.latest_training_id || 'None'}</span>
            </div>
          </div>
        )}
      </div>
      
      <div className="card logs-card">
        <div className="logs-header">
          <h2>Server Logs</h2>
          <div className="logs-controls">
            <div className="form-group">
              <label htmlFor="numLines">Lines:</label>
              <select
                id="numLines"
                value={numLogLines}
                onChange={(e) => setNumLogLines(Number(e.target.value))}
              >
                <option value="50">50</option>
                <option value="100">100</option>
                <option value="200">200</option>
                <option value="500">500</option>
              </select>
            </div>
            <button
              className="refresh-btn"
              onClick={handleRefreshLogs}
            >
              Refresh
            </button>
          </div>
        </div>
        
        <LogsViewer logs={logs} />
      </div>
    </div>
  );
};

export default SystemStatus;
