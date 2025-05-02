// src/components/TrainingStatusCard.js
import React from 'react';
import './TrainingStatusCard.css';

const TrainingStatusCard = ({ status }) => {
  if (!status) {
    return (
      <div className="card status-card">
        <h2>Training Status</h2>
        <div className="status-content">
          <p>Unable to fetch training status</p>
        </div>
      </div>
    );
  }

  const { is_training, current_round, total_rounds, connected_clients, start_time, elapsed_time } = status;

  return (
    <div className="card status-card">
      <h2>Training Status</h2>
      <div className="status-content">
        <div className="status-item">
          <span className="status-label">Status:</span>
          <span className={`status-value ${is_training ? 'active' : 'inactive'}`}>
            {is_training ? 'Active' : 'Inactive'}
          </span>
        </div>
        
        {is_training && (
          <>
            <div className="status-item">
              <span className="status-label">Progress:</span>
              <div className="progress-container">
                <div 
                  className="progress-bar" 
                  style={{ width: `${(current_round / total_rounds) * 100}%` }}
                ></div>
              </div>
              <span className="status-value">{`${current_round} / ${total_rounds} rounds`}</span>
            </div>
            
            <div className="status-item">
              <span className="status-label">Started:</span>
              <span className="status-value">{new Date(start_time).toLocaleString()}</span>
            </div>
            
            <div className="status-item">
              <span className="status-label">Elapsed:</span>
              <span className="status-value">{elapsed_time}</span>
            </div>
            
            <div className="status-item">
              <span className="status-label">Connected Clients:</span>
              <span className="status-value">{connected_clients.length}</span>
            </div>
            
            <div className="client-list">
              {connected_clients.map((client, index) => (
                <div key={index} className="client-badge">{client}</div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default TrainingStatusCard;
