// src/components/LogsViewer.js
import React from 'react';
import './LogsViewer.css';

const LogsViewer = ({ logs }) => {
  if (!logs || logs.length === 0) {
    return <div className="logs-empty">No logs available</div>;
  }

  const getLogClass = (log) => {
    if (log.includes('ERROR')) return 'log-error';
    if (log.includes('WARNING')) return 'log-warning';
    if (log.includes('INFO')) return 'log-info';
    return '';
  };

  return (
    <div className="logs-container">
      {logs.map((log, index) => (
        <div key={index} className={`log-line ${getLogClass(log)}`}>
          {log}
        </div>
      ))}
    </div>
  );
};

export default LogsViewer;
