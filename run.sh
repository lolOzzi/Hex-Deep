#!/bin/bash
export DISPLAY=:0  # Usually set by WSLg, but do it just in case

sudo docker run --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=20g \
  -e DISPLAY=$DISPLAY \
  -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$(pwd)/Hex:/Hex" \
  -it --rm tf25-gui
