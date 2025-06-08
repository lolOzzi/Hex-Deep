FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Install GUI dependencies and useful packages
RUN apt update && apt install -y \
    libgl1 \
    libglib2.0-0 \
    python3-tk \
    sudo \
    x11-xserver-utils && \
    rm -rf /var/lib/apt/lists/*

# Add a non-root user with sudo
ARG USERNAME=devuser
RUN useradd -m -s /bin/bash $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Uninstall headless OpenCV and install full OpenCV with matplotlib
RUN pip uninstall -y opencv-python-headless && \
    pip install opencv-python matplotlib

# Switch to user (you can remove USER ... if you want to run as root)
USER $USERNAME
WORKDIR /Hex

# Default command
CMD ["bash"]
