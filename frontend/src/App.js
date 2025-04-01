import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import ClientList from './components/ClientList';
import TrainingSession from './components/TrainingSession';
import ModelMonitor from './components/ModelMonitor';
import BlockchainInteraction from './components/BlockchainInteraction';
import InstitutionRegistration from './components/InstitutionRegistration';
import FLTraining from './components/FLTraining';
import { Container, MainContent, GlobalStyle } from './styles';

function App() {
  return (
    <Router>
      <GlobalStyle />
      <Container>
        <Sidebar />
        <MainContent>
          <Routes>
            <Route path="/clients" element={<ClientList />} />
            <Route path="/training" element={<TrainingSession />} />
            <Route path="/model" element={<ModelMonitor />} />
            <Route path="/blockchain" element={<BlockchainInteraction />} />
            <Route path="/register" element={<InstitutionRegistration />} /> 
            <Route path="/fl-training" element={<FLTraining />} /> 
            <Route path="/" element={<ClientList />} />
          </Routes>
        </MainContent>
      </Container>
    </Router>
  );
}

export default App;