![image](https://github.com/kvu228/FlwrBC/assets/63705942/472eaf2f-e996-4598-8109-212319efb1cf=150x150)



# Description 📜

This demo demonstrates a federated learning using Flwr framework with a private Etherum blockchain to classify images in CIFAR-10 dataset.

The project contains the following:

<li> The code related to smart contracts is placed under <em>./Blockchain</em> </li>
<li> The code related to server organisation is placed under <em>./Server</em> </li>
<li> The code related to client organisation is placed under <em>./Client</em> </li>
</br>

# Requirement ⚙️

To run this project successfully, you need to install the following:

<li>Anacoda</li>
<li>Set up enviroment by using <code>venv.yml</code> with Anacoda promp
<code>conda env create --file venv.yml</code> </li>
<li>NodeJS</li>
<li>Truffle suite and Ganache UI to simulate the blockchain</li>  
</br>

# How to use 🧑‍🏫

## Setup blockchain

In <em>./Blockchain</em>, compile the smart contracts by <code>truffle compile</code>. </br>
After compilation, run the promp <code>truffle migrate --network development</code> to mirgrate the contract on the network.</br>
Please help check the configuration in <code>truffle-config.js</code> </br>
Ganache should be opened before running the project
</br>

## For IPFS

Change your keys in <code>api_key.json</code>
</br></br>

## For Server organisation

<ol>
    <li>Change the direction to <em>./Server</em></li>
    <li>Use uvicorn promp to start a server: <code>uvicorn server_api:server --reload</code>. The server will run default on <code>localhost:8000</code>
    <li>Access <code>localhost:8000/docs</code> to interact with the API
</ol> 
</br>

## For Client organisation

<ol>
    <li>Change the direction to <em>./Client</em></li>
    <li>Use uvicorn promp to start a client: <code>uvicorn client_api:app --reload --port PORT_NUMBER</code>. The client will run default on <code>localhost:PORT_NUMBER</code>
    <li>Access <code>localhost:PORT_NUMBER/docs</code> to interact with the API
</ol>
</br>

# Demo 📹

[Demo Video](https://1drv.ms/u/s!AmrmLtdR9dfIotMXE75GICu_J5Zf5w?e=Y8xvvO)
