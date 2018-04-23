FROM ubuntu:17.10

RUN apt-get update \
 && apt-get -y install curl

RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" \
      | tee /etc/apt/sources.list.d/tensorflow-serving.list \
 && curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -

RUN apt-get update \
 && apt-get -y install tensorflow-model-server \
 && apt-get clean

WORKDIR /server

ADD trained trained

EXPOSE 9000

ENTRYPOINT [ "tensorflow_model_server", "--port=9000", "--model_name=mymodel", "--model_base_path=/server/trained" ]
