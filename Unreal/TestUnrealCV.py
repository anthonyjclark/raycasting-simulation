import sys

sys.path.append("/home/eoca2018/unrealcv/client/python")

#from unrealcv import client  # type: ignore
import unrealcv

client = unrealcv.Client(("localhost", 9000), None)
client.connect(timeout=5)

if not client.isconnected():
    print("UnrealCV server is not running.")
    raise SystemExit


# filename = client.request("vget /camera/0/lit")
# filename = client.request("vget /camera/0/depth depth.exr")

print(client.request("vget /unrealcv/status"))
# print(client.request("vget /unrealcv/help"))
