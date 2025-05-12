module.exports = {
  networks: {
    ganache: {
      host: "192.168.1.146",
      port: 7545,  // Default Ganache GUI port (or 8545 for ganache-cli)
      network_id: "5777",
      from: "0x1D5102A04086332530dbBf906F6D7290EE9302e6" // Get this from Ganache UI
    }
  },
  
  // Configure your compilers
  compilers: {
    solc: {
      version: "0.8.0",
      settings: {
        optimizer: {
          enabled: true,
          runs: 200
        }
      }
    }
  }
};