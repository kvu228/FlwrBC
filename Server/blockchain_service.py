from web3 import Web3
import json


rpcServer = 'HTTP://127.0.0.1:8545'
w3 = Web3(Web3.HTTPProvider(rpcServer))

contributionSC = open('..//blockchain//build//contracts//Contribution.json')
contributionData = json.load(contributionSC)
contributionAbi = contributionData['abi']
addressContribution = contributionData['networks']['5777']['address']
contribution_contract_instance = w3.eth.contract(address=addressContribution, abi=contributionAbi)


federationSC = open('..//blockchain//build//contracts//Federation.json')
federationData = json.load(federationSC)
federationAbi = federationData['abi']
addressFederation = federationData['networks']['5777']['address']
federation_contract_instance = w3.eth.contract(address=addressFederation, abi=federationAbi)

event_filter = contribution_contract_instance.events.contributeEvent.createFilter(fromBlock='latest')
poll_interval = 2

class BlockchainService():
    def getContributions(self):
        clientAddresses = contribution_contract_instance.functions.getClientAddresses().call()
        rNos = contribution_contract_instance.functions.get_rNos().call()
        results = []
        for client in clientAddresses:
            for round in rNos:
                contribution = contribution_contract_instance.functions.getContribution(client,round).call()
                contrib_dic={}
                # Bool: Work status
                contrib_dic[0]= contribution[0]
                # Uint: Data size
                contrib_dic[1]= contribution[1]
                # Uint: Account Balance
                contrib_dic[2]= contribution[2]
                # Uint: Number of Round
                contrib_dic[3] = round
                # Address: Client Address
                contrib_dic[4] = client

                results.append(contrib_dic)
        return results

    def addStrategy(self, _session:int, _algoName: str, _numRounds: int, _numClients: int):
        server_account = w3.eth.accounts[0]
        federation_contract_instance.functions.addStartegy(_session, _algoName, _numRounds, _numClients).transact({'from': server_account})
        strategy = federation_contract_instance.functions.getStrategy(_session).call()
        return strategy
    
    def addContribution(self, _rNo: int, _dataSize: int, _client_address: str):
        server_account = w3.eth.accounts[0]
        contribution_contract_instance.functions.calculateContribution(_rNo, True, _dataSize).transact({"from":_client_address})
        w3.eth.send_transaction({
            'from': server_account,
            'to': _client_address,
            'value': w3.toWei(_dataSize/500,'ether')
        })
        contribution = contribution_contract_instance.functions.getContribution(_client_address, _rNo)
        return contribution

    def addModel(self, _session:int, _round_num:int, _filePath:str, _fileHash:str):
        server_account = w3.eth.accounts[0]
        federation_contract_instance.functions.addModel(_session,_round_num,_filePath,_fileHash).transact({'from':server_account})
        model = federation_contract_instance.functions.getModel(_session,_round_num).call()
        return model
    
    def getModel(self, _session:int, _round_num:int):
        model = federation_contract_instance.functions.getModel(_session,_round_num).call()
        return model

    def getTrainingSessions(self):
        clientAddresses = contribution_contract_instance.functions.getClientAddresses().call()
        rNos = contribution_contract_instance.functions.get_rNos().call()
        results = []
        for round in rNos:
            session_dict ={}
            dataSize = 0
            contributers = 0
            for client in clientAddresses:
                contribution = contribution_contract_instance.functions.getContribution(client, round).call()
                if contribution[1] > 0:
                    dataSize = dataSize + int(contribution[1])
                    contributers +=1
            session_dict["roundNumber"] = round
            session_dict["contributers"] = contributers
            session_dict["dataSize"] = dataSize
            results.append(session_dict)
        return results

    def getLastModel(self,_session:int, _round_num:int):
        model = federation_contract_instance.functions.getModel(_session,_round_num).call()
        return model
