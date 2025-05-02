// src/components/SystemStatusCard.js
import React from 'react';
import './SystemStatusCard.css';

const SystemStatusCard = ({ status }) => {
  if (!status) {
    return (
      <div className="card system-card">
        <h2>System Status</h2>
        <div className="status-content">
          <p>Unable to fetch system status</p>
        </div>
      </div>
    );
  }

  const { flower_status, latest_training_id } = status;

  return (
    <div className="card system-card">
      <h2>System Status</h2>
      <div className="status-content">
        <div className="status-item">
          <span className="status-label">Flower Server:</span>
          <span className={`status-value ${flower_status.is_running ? 'active' : 'inactive'}`}>
            {flower_status.is_running ? 'Running' : 'Stopped'}
          </span>
        </div>
        
        {flower_status.is_running && (
          <div className="status-item">
            <span className="status-label">Current Training:</span>
            <span className="status-value">{flower_status.current_training_id}</span>
          </div>
        )}
        
        <div className="status-item">
          <span className="status-label">Latest Training:</span>
          <span className="status-value">{latest_training_id || 'None'}</span>
        </div>
      </div>
    </div>
  );
};

export default SystemStatusCard;
