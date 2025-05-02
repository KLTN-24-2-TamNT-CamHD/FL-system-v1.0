import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import TrainingControl from './pages/TrainingControl';
import History from './pages/History';
import Clients from './pages/Clients';
import SystemStatus from './pages/SystemStatus';
import PerformanceMonitoring from './pages/PerformanceMonitoring';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <div className="content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/training" element={<TrainingControl />} />
            <Route path="/history" element={<History />} />
            <Route path="/clients" element={<Clients />} />
            <Route path="/status" element={<SystemStatus />} />
            <Route path="/performance" element={<PerformanceMonitoring />} />
          </Routes>
        </div>
        <ToastContainer position="bottom-right" />
      </div>
    </Router>
  );
}

export default App;
