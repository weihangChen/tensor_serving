# tensor_serving

we experiment [tensor serving](https://www.tensorflow.org/serving/) at a Hackerton, we successfully spin up the service, build a client interacting with 
the server. The explanation here is short, but our implementation is neat and precise. 

### Steps required to spin up tensor serving and invoke the client script to interact with server  
1.	Run **tfserving_model_create.py** to generate model
1.	Move the model to **server/trained**
2.	Run the **docker-compose** file, that’s all

This project consists of four major parts

- tfserving_model_create.py (mandatory) - this is a separate script that can be run on its own. This script creates three tf tensors, define the graph, use tf.saved_model to persist and restore the graph, and it feeds input to the restored graph yielding an output. It is important to make sure that the restored graph is able to consume the input and yield output before moving forward, because the same failure here will occur in tensor serving as well.

- serving (no script required,mandatory) - persisted model files should be put under ‘server/trained’. Our Dockerfile is located under ‘server’, once run, it will spin up the tensorflow serving server and host the persisted model

- client (mandatory) - **client.py** contains the minimal amount of script to do a grpc call , feeding the input to tensor serving and get response. There is no need to dockerize the client code, but we do it here

- docker-compose.yml (optional) - Since we dockerize both the client and server folder, we need a compose file to orchestra these two containers. Once ‘’docker-compose up --build’ is run, server will be up first, then client script will be invoked to get response from server.  

