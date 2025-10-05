cursor/bc-f408c7bd-bc2a-48a4-bc8d-0989f628ad52-ef2e
FROM julia:1.10-bullseye

WORKDIR /app
COPY julia_server/Project.toml /app/Project.toml
COPY julia_server/src /app/src

RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate(); Pkg.precompile()'

EXPOSE 8088 8089
CMD ["julia", "-e", "using ChaosServer; ChaosServer.start()"]
=======
FROM julia:1.10
WORKDIR /app
COPY julia_server/Project.toml /app/Project.toml
RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
COPY julia_server/src /app/src
EXPOSE 8088 8089
CMD ["julia", "-e", "include(\"src/Server.jl\"); using .ChaosServer; ChaosServer.start()"]
main
