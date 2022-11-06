from web3 import Web3
import json
import time

from threading import Thread


rpcServer = 'HTTP://172.25.112.1:7545'
w3 = Web3(Web3.HTTPProvider(rpcServer))

contributionSC = open('../blockchain/build/contracts/Contribution.json')
contributionData = json.load(contributionSC)
contributionAbi = contributionData['abi']
addressContribution = contributionData['networks']['5777']['address']
contribution_contract_instance = w3.eth.contract(address=addressContribution, abi=contributionAbi)


federationSC = open('../blockchain/build/contracts/Federation.json')
federationData = json.load(federationSC)
federationAbi = federationData['abi']
addressFederation = federationData['networks']['5777']['address']
federation_contract_instance = w3.eth.contract(address=addressFederation, abi=federationAbi)



class BlockchainService():
    def addWeight(self, _session: int, _round_num: int, _dataSize: int, _filePath: str, _fileHash: str):
        default_account = w3.eth.accounts[1]
        federation_contract_instance.functions.addWeight(_session, _round_num, _dataSize, _filePath, _fileHash).transact({'from': default_account})
        result = federation_contract_instance.getWeight(_session,_round_num).call()
        return result

    def getAddress(self):
        return w3.eth.accounts[1]

    def getContributions(client_address):
        roundNumbers = contribution_contract_instance.functions.get_rNos().call()
        contributions = []
        for rNo in roundNumbers:
            contribution = contribution_contract_instance.functions.get_contribution(client_address, rNo).call()
            if contribution[1]>0:
                contrib_dic={}
                # Bool: Work status
                contrib_dic[0]= contribution[0]
                # Uint: Data size
                contrib_dic[1]= contribution[1]
                # Uint: Account Balance
                contrib_dic[2]= contribution[2]
                # Uint: Number of Round
                contrib_dic[3]= rNo
                contributions.append(contrib_dic)
        return contributions



def handle_event(event):
    print(event)
    from client_api import FLlaunch
    FLlaunch = FLlaunch()
    FLlaunch.start()

def log_loop(event_filter, poll_interval):
    print(event_filter.get_new_entries())
    while True:
        for event in event_filter.get_new_entries():
            handle_event(event)
        time.sleep(poll_interval)


def main():
    block_filter = federation_contract_instance.events.addStrategyEvent.createFilter(fromBlock='latest')
    worker = Thread(target=log_loop, args=(block_filter, 5), daemon=True)
    worker.start()
        
main()



