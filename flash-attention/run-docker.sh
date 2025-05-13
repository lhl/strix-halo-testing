podman run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --privileged \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH \
  -v /home/lhl:/home/lhl \
  docker.io/scottt/therock:pytorch-vision-dev-f41
