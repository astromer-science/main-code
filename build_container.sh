docker build -t astromer \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) .
