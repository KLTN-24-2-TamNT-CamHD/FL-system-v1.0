// src/pages/TrainingControl.js
import React, { useState, useEffect } from 'react';
import { toast } from 'react-toastify';
import { startTraining, stopTraining, getTrainingStatus } from '../api/flService';
import { FaPlay, FaStop, FaChevronDown, FaChevronUp, FaPlus, FaTrash } from 'react-icons/fa';
import './TrainingControl.css';

const defaultConfig = {
  num_rounds: 3,
  min_fit_clients: 2,
  min_evaluate_clients: 2,
  fraction_fit: 1.0,
  ipfs_url: "http://127.0.0.1:5001/api/v0",
  ganache_url: "http://192.168.1.146:7545",
  deploy_contract: true,
  version_prefix: "1.0",
  authorized_clients_only: false,
  round_rewards: 1000,
  device: "cpu"
};

const TrainingControl = () => {
  const [config, setConfig] = useState(defaultConfig);
  const [isTraining, setIsTraining] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [authorizedClients, setAuthorizedClients] = useState(['']);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const checkTrainingStatus = async () => {
      try {
        const response = await getTrainingStatus();
        // Make sure we're checking the is_training property correctly
        if (response && typeof response.is_training === 'boolean') {
          setIsTraining(response.is_training);
        }
      } catch (error) {
        console.error("Failed to fetch training status:", error);
        // Don't reset isTraining here to avoid resetting when there's just a network error
      }
    };

    checkTrainingStatus();
  }, []);

  const handleInputChange = (e) => {
    const { name, value, type } = e.target;
    
    if (type === 'number') {
      setConfig({
        ...config,
        [name]: name === 'fraction_fit' ? parseFloat(value) : parseInt(value, 10)
      });
    } else if (type === 'checkbox') {
      setConfig({
        ...config,
        [name]: e.target.checked
      });
    } else {
      setConfig({
        ...config,
        [name]: value
      });
    }
  };

  const handleClientChange = (index, value) => {
    const newClients = [...authorizedClients];
    newClients[index] = value;
    setAuthorizedClients(newClients);
  };

  const addClientField = () => {
    setAuthorizedClients([...authorizedClients, '']);
  };

  const removeClientField = (index) => {
    const newClients = authorizedClients.filter((_, i) => i !== index);
    setAuthorizedClients(newClients);
  };

  const handleStartTraining = async () => {
    try {
      setLoading(true);
      
      // Filter out empty client names
      const filteredClients = authorizedClients.filter(client => client.trim() !== '');
      
      const trainingConfig = {
        ...config,
        authorized_clients: config.authorized_clients_only ? filteredClients : []
      };
      
      const response = await startTraining(trainingConfig);
      toast.success(`Training started: ${response.training_id}`);
      setIsTraining(true);
    } catch (error) {
      toast.error(`Failed to start training: ${error.message || 'Unknown error'}`);
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleStopTraining = async () => {
    try {
      setLoading(true);
      const response = await stopTraining();
      toast.info(response.message);
      setIsTraining(false);
    } catch (error) {
      toast.error(`Failed to stop training: ${error.message || 'Unknown error'}`);
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="training-control">
      <h1>Training Control</h1>
      
      <div className="card form-container">
        <h2>Training Configuration</h2>
        
        <div className="form-grid">
          <h3 className="section-title">Basic Settings</h3>
          
          <div className="form-group">
            <label htmlFor="num_rounds">Number of Rounds:</label>
            <input
              type="number"
              id="num_rounds"
              name="num_rounds"
              value={config.num_rounds}
              onChange={handleInputChange}
              min="1"
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="min_fit_clients">Minimum Fit Clients:</label>
            <input
              type="number"
              id="min_fit_clients"
              name="min_fit_clients"
              value={config.min_fit_clients}
              onChange={handleInputChange}
              min="1"
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="min_evaluate_clients">Minimum Evaluate Clients:</label>
            <input
              type="number"
              id="min_evaluate_clients"
              name="min_evaluate_clients"
              value={config.min_evaluate_clients}
              onChange={handleInputChange}
              min="1"
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="fraction_fit">Fraction Fit:</label>
            <input
              type="number"
              id="fraction_fit"
              name="fraction_fit"
              value={config.fraction_fit}
              onChange={handleInputChange}
              min="0"
              max="1"
              step="0.1"
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="device">Device:</label>
            <select
              id="device"
              name="device"
              value={config.device}
              onChange={handleInputChange}
            >
              <option value="cpu">CPU</option>
              <option value="cuda">CUDA (GPU)</option>
            </select>
          </div>
          
          <div className="form-group">
            <label htmlFor="round_rewards">Round Rewards:</label>
            <input
              type="number"
              id="round_rewards"
              name="round_rewards"
              value={config.round_rewards}
              onChange={handleInputChange}
              min="0"
            />
          </div>
          
          <button 
            className="toggle-advanced-btn" 
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            {showAdvanced ? <FaChevronUp style={{ marginRight: '8px' }} /> : <FaChevronDown style={{ marginRight: '8px' }} />}
            {showAdvanced ? 'Hide Advanced Settings' : 'Show Advanced Settings'}
          </button>
          
          {showAdvanced && (
            <div className="advanced-settings">
              <h3 className="section-title">Advanced Settings</h3>
              
              <div className="form-grid">
                <div className="form-group">
                  <label htmlFor="ipfs_url">IPFS URL:</label>
                  <input
                    type="text"
                    id="ipfs_url"
                    name="ipfs_url"
                    value={config.ipfs_url}
                    onChange={handleInputChange}
                  />
                </div>
                
                <div className="form-group">
                  <label htmlFor="ganache_url">Ganache URL:</label>
                  <input
                    type="text"
                    id="ganache_url"
                    name="ganache_url"
                    value={config.ganache_url}
                    onChange={handleInputChange}
                  />
                </div>
                
                <div className="form-group">
                  <label htmlFor="contract_address">Contract Address (Optional):</label>
                  <input
                    type="text"
                    id="contract_address"
                    name="contract_address"
                    value={config.contract_address || ''}
                    onChange={handleInputChange}
                    placeholder="0x123abc..."
                  />
                </div>
                
                <div className="form-group">
                  <label htmlFor="private_key">Private Key (Optional):</label>
                  <input
                    type="password"
                    id="private_key"
                    name="private_key"
                    value={config.private_key || ''}
                    onChange={handleInputChange}
                    placeholder="0x..."
                  />
                </div>
                
                <div className="form-group">
                  <label htmlFor="version_prefix">Version Prefix:</label>
                  <input
                    type="text"
                    id="version_prefix"
                    name="version_prefix"
                    value={config.version_prefix}
                    onChange={handleInputChange}
                  />
                </div>
                
                <div className="form-group checkbox">
                  <input
                    type="checkbox"
                    id="deploy_contract"
                    name="deploy_contract"
                    checked={config.deploy_contract}
                    onChange={handleInputChange}
                  />
                  <label htmlFor="deploy_contract">Deploy New Contract</label>
                </div>
                
                <div className="form-group checkbox">
                  <input
                    type="checkbox"
                    id="authorized_clients_only"
                    name="authorized_clients_only"
                    checked={config.authorized_clients_only}
                    onChange={handleInputChange}
                  />
                  <label htmlFor="authorized_clients_only">Authorized Clients Only</label>
                </div>
                
                {config.authorized_clients_only && (
                  <div className="authorized-clients">
                    <h3 className="section-title">Authorized Clients</h3>
                    {authorizedClients.map((client, index) => (
                      <div key={index} className="client-input">
                        <input
                          type="text"
                          value={client}
                          onChange={(e) => handleClientChange(index, e.target.value)}
                          placeholder="Client ID"
                        />
                        <button
                          type="button"
                          onClick={() => removeClientField(index)}
                          className="remove-btn"
                        >
                          <FaTrash style={{ marginRight: '4px' }} /> Remove
                        </button>
                      </div>
                    ))}
                    <button
                      type="button"
                      onClick={addClientField}
                      className="add-btn"
                    >
                      <FaPlus style={{ marginRight: '4px' }} /> Add Client
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
          
          <div className="action-buttons">
            {!isTraining ? (
              <button
                onClick={handleStartTraining}
                disabled={loading}
                className="start-btn"
              >
                <FaPlay style={{ marginRight: '8px' }} />
                {loading ? 'Starting...' : 'Start Training'}
              </button>
            ) : (
              <button
                onClick={handleStopTraining}
                disabled={loading}
                className="stop-btn"
              >
                <FaStop style={{ marginRight: '8px' }} />
                {loading ? 'Stopping...' : 'Stop Training'}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingControl;
