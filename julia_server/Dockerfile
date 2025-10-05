FROM julia:1.10
WORKDIR /app
COPY julia_server/Project.toml /app/Project.toml
RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
COPY julia_server/src /app/src
EXPOSE 8088 8089
CMD ["julia", "-e", "include(\"src/Server.jl\"); using .ChaosServer; ChaosServer.start()"]
