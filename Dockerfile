FROM azul/zulu-openjdk:8-latest
RUN apt-get -y update && \
    apt-get install -y build-essential cmake python3 git neovim maven python-is-python3 curl
ENTRYPOINT ["/bin/bash"]
