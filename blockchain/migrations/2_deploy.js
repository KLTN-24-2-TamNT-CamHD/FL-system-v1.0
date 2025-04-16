const Federation = artifacts.require("Federation");

module.exports = function(deployer) {
  deployer.deploy(Federation);
};