// src/pages/Clients.js
import React, { useState, useEffect } from 'react';
import { toast } from 'react-toastify';
import { registerClient } from '../api/flService';
import axios from 'axios';
import './Clients.css';

const Clients = () => {
  const [clientId, setClientId] = useState('');
  const [deviceType, setDeviceType] = useState('');
  const [modelType, setModelType] = useState('');
  const [location, setLocation] = useState('');
  const [loading, setLoading] = useState(false);
  const [clientAddresses, setClientAddresses] = useState([]);
  const [fetchingClients, setFetchingClients] = useState(true);
  const [fetchError, setFetchError] = useState(null);

  // Fetch authorized clients from the blockchain API
  useEffect(() => {
    const fetchAuthorizedClients = async () => {
      try {
        setFetchingClients(true);
        const response = await axios.get('http://192.168.80.180:8000/api/authorized-clients');
        
        console.log('API Response:', response.data);
        
        // Check if response has the clients array
        if (response.data && response.data.clients && Array.isArray(response.data.clients)) {
          setClientAddresses(response.data.clients);
          setFetchError(null);
        } else {
          setClientAddresses([]);
          setFetchError('Invalid response format from API');
        }
      } catch (error) {
        console.error('Failed to fetch authorized clients:', error);
        setFetchError('Failed to load clients from blockchain. Please try again.');
        toast.error('Failed to load authorized clients');
        setClientAddresses([]);
      } finally {
        setFetchingClients(false);
      }
    };

    fetchAuthorizedClients();
    
    // Set up periodic refresh (optional)
    const intervalId = setInterval(fetchAuthorizedClients, 30000);
    
    return () => clearInterval(intervalId);
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!clientId.trim()) {
      toast.error('Client ID is required');
      return;
    }
    
    try {
      setLoading(true);
      
      const clientData = {
        client_id: clientId,
        client_info: {
          device_type: deviceType,
          model_type: modelType,
          location: location
        }
      };
      
      const response = await registerClient(clientData);
      toast.success(`Client registered: ${response.client_id}`);
      
      // Refresh the client list after registering
      try {
        const clientsResponse = await axios.get('http://192.168.80.180:8000/api/authorized-clients');
        if (clientsResponse.data && clientsResponse.data.clients && Array.isArray(clientsResponse.data.clients)) {
          setClientAddresses(clientsResponse.data.clients);
        }
      } catch (refreshError) {
        console.error('Failed to refresh client list:', refreshError);
      }
      
      // Clear the form
      setClientId('');
      setDeviceType('');
      setModelType('');
      setLocation('');
    } catch (error) {
      toast.error(`Failed to register client: ${error.message || 'Unknown error'}`);
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="clients-page">
      <h1>Clients Management</h1>
      
      <div className="clients-grid">
        <div className="card register-card">
          <h2>Register New Client</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="clientId">Client ID:</label>
              <input
                type="text"
                id="clientId"
                value={clientId}
                onChange={(e) => setClientId(e.target.value)}
                placeholder="Enter client ID"
                required
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="deviceType">Device Type:</label>
              <input
                type="text"
                id="deviceType"
                value={deviceType}
                onChange={(e) => setDeviceType(e.target.value)}
                placeholder="e.g., raspberry_pi, jetson_nano"
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="modelType">Model Type:</label>
              <input
                type="text"
                id="modelType"
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                placeholder="e.g., cnn, lstm, mlp"
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="location">Location:</label>
              <input
                type="text"
                id="location"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                placeholder="e.g., edge_device_1"
              />
            </div>
            
            <button
              type="submit"
              className="register-btn"
              disabled={loading}
            >
              {loading ? 'Registering...' : 'Register Client'}
            </button>
          </form>
        </div>
        
        <div className="card clients-list-card">
          <h2>Authorized Clients</h2>
          
          {fetchingClients && <div className="loading-clients">Loading clients data...</div>}
          
          {fetchError && <div className="error-message">{fetchError}</div>}
          
          {!fetchingClients && !fetchError && clientAddresses.length === 0 ? (
            <div className="no-clients">No authorized clients found</div>
          ) : (
            <div className="clients-table-container">
              <table className="clients-table">
                <thead>
                  <tr>
                    <th>Client Address</th>
                  </tr>
                </thead>
                <tbody>
                  {Array.isArray(clientAddresses) && clientAddresses.map((address, index) => (
                    <tr key={index}>
                      <td className="client-address">{address}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          
          <div className="refresh-container">
            <button 
              className="refresh-btn"
              onClick={async () => {
                try {
                  setFetchingClients(true);
                  const response = await axios.get('http://192.168.80.180:8000/api/authorized-clients');
                  console.log('Manual Refresh Response:', response.data);
                  
                  if (response.data && response.data.clients && Array.isArray(response.data.clients)) {
                    setClientAddresses(response.data.clients);
                    setFetchError(null);
                    toast.info('Client list refreshed');
                  } else {
                    setFetchError('Invalid response format from API');
                  }
                } catch (error) {
                  console.error('Failed to refresh client list:', error);
                  setFetchError('Failed to refresh clients data');
                  toast.error('Failed to refresh clients data');
                } finally {
                  setFetchingClients(false);
                }
              }}
            >
              Refresh Clients
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Clients;
