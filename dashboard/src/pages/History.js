// src/pages/History.js
import React, { useState, useEffect } from 'react';
import { getTrainingHistory } from '../api/flService';
import './History.css';

const History = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        setLoading(true);
        const response = await getTrainingHistory();
        setHistory(response.history);
        setError(null);
      } catch (err) {
        setError('Failed to fetch training history');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, []);

  if (loading) {
    return <div className="loading">Loading training history...</div>;
  }

  if (error) {
    return <div className="error-message">{error}</div>;
  }

  if (history.length === 0) {
    return (
      <div className="history">
        <h1>Training History</h1>
        <div className="no-history">No training sessions found</div>
      </div>
    );
  }

  return (
    <div className="history">
      <h1>Training History</h1>
      
      <div className="history-table-container">
        <table className="history-table">
          <thead>
            <tr>
              <th>Training ID</th>
              <th>Start Time</th>
              <th>End Time</th>
              <th>Rounds</th>
              <th>Final Accuracy</th>
              <th>Contract Address</th>
            </tr>
          </thead>
          <tbody>
            {history.map((session, index) => (
              <tr key={index}>
                <td>{session.training_id}</td>
                <td>{new Date(session.start_time).toLocaleString()}</td>
                <td>{session.end_time ? new Date(session.end_time).toLocaleString() : 'N/A'}</td>
                <td>{session.num_rounds}</td>
                <td>{session.final_accuracy ? `${(session.final_accuracy * 100).toFixed(2)}%` : 'N/A'}</td>
                <td>
                  <span className="contract-address" title={session.contract_address}>
                    {session.contract_address.substring(0, 10)}...
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default History;
