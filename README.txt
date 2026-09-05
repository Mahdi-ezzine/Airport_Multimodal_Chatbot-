# Airport Passenger Assistance — how to run

Requires Docker Desktop (Windows/macOS) or Docker Engine (Linux), running,
and about 10 GB of free disk space.

    docker build -t airport-assistant .
    docker run -p 8501:8501 airport-assistant

Then open http://localhost:8501

The first build takes 10 to 20 minutes, almost all of it downloading PyTorch
and the model weights. The first question takes about a minute while the
models load into memory; later ones are fast.

If port 8501 is already in use:

    docker run -p 8600:8501 airport-assistant

then open http://localhost:8600

Press Ctrl+C in the terminal to stop.
