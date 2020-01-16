This repo contains my (current record holding) entry to the [comma.ai speed challenge](https://github.com/commaai/speedchallenge).  The speed challenge is a challenge to predict the speed of a car based solely on dashcam video footage.

My entry uses a resnet-based posenet trained on the [comma ai 2k19 dataset](https://github.com/commaai/comma2k19).

Future work:

  - train on the waymo open dataset for better low speed predictions
  - use deeper, more modern architectures (replace the resnet18 encoder with an SE resnext50 encoder)
