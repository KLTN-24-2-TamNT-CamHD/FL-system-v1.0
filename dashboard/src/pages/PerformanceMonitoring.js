// src/pages/PerformanceMonitoring.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';
import './PerformanceMonitoring.css';

const API_BASE_URL = 'http://localhost:8000/api';

// This is a mock function that would be replaced with your actual API endpoint
const getPerformanceData = async (trainingId) => {
  try {
    // In a real application, you would implement this endpoint on your API
    // For now, we'll mock the data for demonstration purposes
    // const response = await axios.get(`${API_BASE_URL}/performance/${trainingId}`);
    // return response.data;
    
    // Mock data for demonstration
    return {
      status: 'success',
      training_id: trainingId,
      metrics: {
        accuracy: [0.45, 0.62, 0.71, 0.78, 0.82, 0.85, 0.87, 0.89, 0.90, 0.91],
        loss: [2.1, 1.7, 1.4, 1.1, 0.9, 0.7, 0.6, 0.5, 0.45, 0.42],
        rounds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        client_metrics: [
          {
            client_id: 'client_1',
            accuracy: [0.41, 0.58, 0.67, 0.75, 0.79, 0.82, 0.84, 0.86, 0.87, 0.88],
            loss: [2.3, 1.9, 1.6, 1.3, 1.0, 0.8, 0.7, 0.6, 0.55, 0.51]
          },
          {
            client_id: 'client_2',
            accuracy: [0.43, 0.60, 0.69, 0.76, 0.81, 0.84, 0.86, 0.88, 0.89, 0.90],
            loss: [2.2, 1.8, 1.5, 1.2, 0.95, 0.75, 0.65, 0.55, 0.5, 0.47]
          }
        ]
      }
    };
  } catch (error) {
    throw error.response ? error.response.data : error;
  }
};

// This is a mock function to get the list of training sessions
const getTrainingSessions = async () => {
  try {
    // In a real application, you would implement this endpoint on your API
    // const response = await axios.get(`${API_BASE_URL}/training-history`);
    // return response.data.history;
    
    // Mock data for demonstration
    return [
      {
        training_id: 'training_20250501142355',
        start_time: '2025-05-01T14:23:55',
        end_time: '2025-05-01T14:45:22',
        num_rounds: 10,
        final_accuracy: 0.91,
        contract_address: '0x58a6e4bb07c90b331b314cbbaca320022a8404fa56c48dc0c557f34e21587bdd'
      },
      {
        training_id: 'training_20250430093012',
        start_time: '2025-04-30T09:30:12',
        end_time: '2025-04-30T10:15:45',
        num_rounds: 8,
        final_accuracy: 0.85,
        contract_address: '0x7a1b4c5d2e3f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e'
      }
    ];
  } catch (error) {
    throw error;
  }
};

const PerformanceMonitoring = () => {
  const [sessions, setSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState('');
  const [performanceData, setPerformanceData] = useState(null);
  const [activeMetric, setActiveMetric] = useState('accuracy');
  const [showAllClients, setShowAllClients] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchSessions = async () => {
      try {
        setLoading(true);
        const sessionsData = await getTrainingSessions();
        setSessions(sessionsData);
        
        if (sessionsData.length > 0) {
          setSelectedSession(sessionsData[0].training_id);
        }
        
        setError(null);
      } catch (err) {
        setError('Failed to fetch training sessions');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchSessions();
  }, []);

  useEffect(() => {
    const fetchPerformanceData = async () => {
      if (selectedSession) {
        try {
          setLoading(true);
          const data = await getPerformanceData(selectedSession);
          setPerformanceData(data);
          setError(null);
        } catch (err) {
          setError('Failed to fetch performance data');
          console.error(err);
        } finally {
          setLoading(false);
        }
      }
    };

    fetchPerformanceData();
  }, [selectedSession]);

  const prepareChartData = () => {
    if (!performanceData || !performanceData.metrics) return [];
    
    const { rounds, accuracy, loss, client_metrics } = performanceData.metrics;
    
    return rounds.map((round, index) => {
      const dataPoint = {
        round,
        globalAccuracy: accuracy[index],
        globalLoss: loss[index],
      };
      
      if (showAllClients && client_metrics) {
        client_metrics.forEach(client => {
          dataPoint[`${client.client_id}_accuracy`] = client.accuracy[index];
          dataPoint[`${client.client_id}_loss`] = client.loss[index];
        });
      }
      
      return dataPoint;
    });
  };

  const chartData = prepareChartData();
  
  const getSessionById = (id) => {
    return sessions.find(session => session.training_id === id);
  };

  const currentSession = selectedSession ? getSessionById(selectedSession) : null;

  if (loading && !sessions.length) {
    return <div className="loading">Loading performance data...</div>;
  }

  if (error) {
    return <div className="error-message">{error}</div>;
  }

  if (!sessions.length) {
    return (
      <div className="performance-monitoring">
        <h1>Performance Monitoring</h1>
        <div className="card">
          <div className="no-data">No training sessions available</div>
        </div>
      </div>
    );
  }

  return (
    <div className="performance-monitoring">
      <h1>Performance Monitoring</h1>
      
      <div className="card">
        <div className="session-selector">
          <label htmlFor="sessionSelect">Select Training Session:</label>
          <select
            id="sessionSelect"
            value={selectedSession}
            onChange={(e) => setSelectedSession(e.target.value)}
          >
            {sessions.map((session) => (
              <option key={session.training_id} value={session.training_id}>
                {session.training_id} - {new Date(session.start_time).toLocaleString()}
              </option>
            ))}
          </select>
        </div>
        
        {currentSession && (
          <div className="session-details">
            <div className="detail-item">
              <span className="detail-label">Start Time:</span>
              <span className="detail-value">{new Date(currentSession.start_time).toLocaleString()}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">End Time:</span>
              <span className="detail-value">
                {currentSession.end_time ? new Date(currentSession.end_time).toLocaleString() : 'Ongoing'}
              </span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Rounds:</span>
              <span className="detail-value">{currentSession.num_rounds}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Final Accuracy:</span>
              <span className="detail-value">
                {currentSession.final_accuracy ? `${(currentSession.final_accuracy * 100).toFixed(2)}%` : 'N/A'}
              </span>
            </div>
          </div>
        )}
      </div>
      
      <div className="card chart-card">
        <div className="chart-controls">
          <div className="metric-selector">
            <span>Metric:</span>
            <div className="toggle-buttons">
              <button
                className={activeMetric === 'accuracy' ? 'active' : ''}
                onClick={() => setActiveMetric('accuracy')}
              >
                Accuracy
              </button>
              <button
                className={activeMetric === 'loss' ? 'active' : ''}
                onClick={() => setActiveMetric('loss')}
              >
                Loss
              </button>
            </div>
          </div>
          
          {performanceData && performanceData.metrics && performanceData.metrics.client_metrics && (
            <div className="client-toggle">
              <label>
                <input
                  type="checkbox"
                  checked={showAllClients}
                  onChange={() => setShowAllClients(!showAllClients)}
                />
                Show Individual Clients
              </label>
            </div>
          )}
        </div>
        
        <div className="chart-container">
          {performanceData && chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="round" label={{ value: 'Round', position: 'bottom', offset: 0 }} />
                <YAxis 
                  domain={activeMetric === 'accuracy' ? [0, 1] : ['auto', 'auto']}
                  label={{ 
                    value: activeMetric === 'accuracy' ? 'Accuracy' : 'Loss', 
                    angle: -90, 
                    position: 'insideLeft' 
                  }}
                />
                <Tooltip formatter={(value) => activeMetric === 'accuracy' ? `${(value * 100).toFixed(2)}%` : value.toFixed(4)} />
                <Legend />
                
                {/* Global Metrics Line */}
                <Line
                  type="monotone"
                  dataKey={activeMetric === 'accuracy' ? 'globalAccuracy' : 'globalLoss'}
                  name="Global Model"
                  stroke="#3f51b5"
                  strokeWidth={3}
                  dot={{ r: 4 }}
                  activeDot={{ r: 6 }}
                />
                
                {/* Client Metrics Lines (if showing all clients) */}
                {showAllClients && performanceData.metrics.client_metrics.map((client, idx) => (
                  <Line
                    key={client.client_id}
                    type="monotone"
                    dataKey={activeMetric === 'accuracy' ? `${client.client_id}_accuracy` : `${client.client_id}_loss`}
                    name={client.client_id}
                    stroke={idx === 0 ? '#f50057' : '#00bcd4'}
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="no-data">No performance data available for this session</div>
          )}
        </div>
      </div>
      
      {/* Model performance metrics details */}
      {performanceData && performanceData.metrics && (
        <div className="card metrics-card">
          <h2>Detailed Metrics</h2>
          
          <div className="metrics-table-container">
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>Round</th>
                  <th>Global Accuracy</th>
                  <th>Global Loss</th>
                  {showAllClients && performanceData.metrics.client_metrics.map(client => (
                    <React.Fragment key={client.client_id}>
                      <th>{client.client_id} Accuracy</th>
                      <th>{client.client_id} Loss</th>
                    </React.Fragment>
                  ))}
                </tr>
              </thead>
              <tbody>
                {chartData.map((data, idx) => (
                  <tr key={idx}>
                    <td>{data.round}</td>
                    <td>{(data.globalAccuracy * 100).toFixed(2)}%</td>
                    <td>{data.globalLoss.toFixed(4)}</td>
                    {showAllClients && performanceData.metrics.client_metrics.map(client => (
                      <React.Fragment key={client.client_id}>
                        <td>{(data[`${client.client_id}_accuracy`] * 100).toFixed(2)}%</td>
                        <td>{data[`${client.client_id}_loss`].toFixed(4)}</td>
                      </React.Fragment>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default PerformanceMonitoring;
