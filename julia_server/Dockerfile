FROM julia:1.10-bullseye

WORKDIR /app
COPY julia_server/Project.toml /app/Project.toml
COPY julia_server/src /app/src

RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate(); Pkg.precompile()'

EXPOSE 8088 8089
CMD ["julia", "-e", "using ChaosServer; ChaosServer.start()"]
