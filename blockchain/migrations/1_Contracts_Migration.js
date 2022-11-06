var Contribution = artifacts.require("Contribution");
var Federatation = artifacts.require("Federation");

module.exports = function (deployer) {
    deployer.deploy(Contribution);
    deployer.deploy(Federatation);
};
