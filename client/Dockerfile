FROM python:2.7

RUN pip install grpcio-tools
RUN pip install numpy
RUN pip install tensorflow
RUN pip install tensorflow-serving-api

ADD mnist-client.py mnist-client.py
ADD mnist_input_data.py mnist_input_data.py

ENTRYPOINT [ "python", "mnist-client.py", "--num_tests=1000", "--server=server:9000" ]
