# Bonsai Gym Wrapper

This repository provides a simple Bonsai wrapper for OpenAI gym environments.

## Steps for Use

1. update `gym_bridge.py` (whatever bridge code you want) to run your environment.
2. test it works locally: `python gym_bridge.py --test-local True`
3. write the dockerfile with all your requirements
4. build the package: `az acr build --image <image_name>:<image_version> --file <dockerfile> --registry <acr-registory> .`
5. register sim-package with bonsai workspace: `bonsai simulator package add -n "FrozenLakePackage" -u alizaidihappytreehaus.azurecr.io/frozenlake:latest -i 100 -r 1 -m 1 --os-type Linux`
6. start-training: add `package "PackageName"`