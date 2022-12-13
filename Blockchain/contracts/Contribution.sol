pragma solidity >=0.5.0 <0.9.0;

contract Contribution{
    
    struct Work {
        bool finished;
        uint dataSize;
    }

    mapping(address => mapping(uint => Work)) public dataContributions;
    mapping(address => uint) public dataContributionsCount;
    mapping(address => uint) public balances;
    address[] public clientAddresses;
    uint[] public rNos;
    event contributeEvent(uint _rNo, address _address);

    function calculateContribution(uint _rNo, bool _finished, uint _dataSize, uint _payment)  public 
    {
        require(
            (dataContributions[msg.sender][_rNo].finished == true || dataContributions[msg.sender][_rNo].finished == false)
            ,
            "Edge already contributed."
        );

        if (_finished == true) {
            if(_dataSize > 500){
                dataContributions[msg.sender][_rNo] = Work(_finished, _dataSize);
                balances[msg.sender] = balances[msg.sender] + _payment;
                clientAddresses.push(msg.sender);
                rNos.push(_rNo);
            }
            dataContributionsCount[msg.sender] =  dataContributionsCount[msg.sender]+_dataSize;                
        }

       emit contributeEvent(_rNo, msg.sender);
    }

    function getClientAddresses() view public returns (address[] memory){
        return clientAddresses;
    }

    function get_rNos() view public returns (uint[] memory){
        return rNos;
    }

    function getContribution(address _clientAddress, uint _rNo) view public returns (bool finished_, uint dataSize_, uint balance_){
        finished_ = dataContributions[_clientAddress][_rNo].finished;
        dataSize_ = dataContributions[_clientAddress][_rNo].dataSize;
        balance_ = balances[_clientAddress];

        return (finished_, dataSize_, balance_);
    }

    function getBalance(address _clientAddress) public view returns(uint balance_){
       return balances[_clientAddress];
    }


}
