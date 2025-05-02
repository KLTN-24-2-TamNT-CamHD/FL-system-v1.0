// src/components/Navbar.js
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
  const location = useLocation();

  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <Link to="/">Federated Learning Dashboard</Link>
      </div>
      <ul className="navbar-nav">
        <li className={location.pathname === '/' ? 'active' : ''}>
          <Link to="/">Dashboard</Link>
        </li>
        <li className={location.pathname === '/training' ? 'active' : ''}>
          <Link to="/training">Training Control</Link>
        </li>
        <li className={location.pathname === '/performance' ? 'active' : ''}>
          <Link to="/performance">Performance</Link>
        </li>
        <li className={location.pathname === '/history' ? 'active' : ''}>
          <Link to="/history">Training History</Link>
        </li>
        <li className={location.pathname === '/clients' ? 'active' : ''}>
          <Link to="/clients">Clients</Link>
        </li>
        <li className={location.pathname === '/status' ? 'active' : ''}>
          <Link to="/status">System Status</Link>
        </li>
      </ul>
    </nav>
  );
};

export default Navbar;
