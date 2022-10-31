#!/bin/bash

echo "Starting server"

python server/server.py &
sleep 5  # Sleep for 5s to give the server enough time to start

# Ensure that the Keras dataset used in client.py is already cached.
# python -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data()"

for i in `seq 0 1`; do
    echo "Starting client $i"
    python client/client.py --partition=${i} &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait