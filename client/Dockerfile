FROM python:2.7

RUN pip install grpcio-tools
RUN pip install numpy
RUN pip install tensorflow
RUN pip install tensorflow-serving-api

ADD client.py client.py

ENTRYPOINT [ "python", "client.py", "--server=server:9000" ]
