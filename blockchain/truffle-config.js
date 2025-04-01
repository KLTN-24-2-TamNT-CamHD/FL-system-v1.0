module.exports = {
  networks: {
    ganache: {
      host: "192.168.1.146",
      port: 7545,  // Default Ganache GUI port (or 8545 for ganache-cli)
      network_id: "5777",
      from: "0xF9C1965243B05861937EB9d50fFF1Aba60F27061" // Get this from Ganache UI
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