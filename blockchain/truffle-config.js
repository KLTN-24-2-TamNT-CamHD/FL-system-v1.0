module.exports = {
  networks: {
    ganache: {
      host: "192.168.80.1",
      port: 7545,  // Default Ganache GUI port (or 8545 for ganache-cli)
      network_id: "5777",
      from: "0xb6845ba34Fc592266921419ef65b9b4D943e2797" // Get this from Ganache UI
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