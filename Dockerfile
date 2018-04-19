FROM tensorflow/tensorflow

WORKDIR /tensor-serving

RUN apt-get update \
 && apt-get -y install python-tk

RUN pip install matplotlib
RUN pip install scikit-image
RUN pip install natsort
RUN pip install mnist

ADD app.py app.py
ADD class_activation_map.py class_activation_map.py
ADD utils.py utils.py
ADD variables.py variables.py
ADD lenet_slim.py lenet_slim.py

ADD trained-model trained-model
ADD images images

ENTRYPOINT [ "python", "app.py" ]
