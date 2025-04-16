module.exports = {
  networks: {
    ganache: {
      host: "192.168.1.146",
      port: 7545,  // Default Ganache GUI port (or 8545 for ganache-cli)
      network_id: "5777",
      from: "0xdE1a8A52252a7724aDfb1AcaC300Ea1b2c4eaFE0" // Get this from Ganache UI
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