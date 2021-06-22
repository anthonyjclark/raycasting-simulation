import sys

sys.path.append("/home/ajc/UE4/unrealcv/client/python")

from unrealcv import client  # type: ignore

client.connect()

if not client.isconnected():
    print("UnrealCV server is not running.")
    raise SystemExit


# filename = client.request("vget /camera/0/lit")
# filename = client.request("vget /camera/0/depth depth.exr")

print(client.request("vget /unrealcv/status"))
# print(client.request("vget /unrealcv/help"))
