from audio import listener
from websocket import API
import time

server = API()
l = listener(server)

try:
    while 1:
        time.sleep(.1)
except KeyboardInterrupt:
    print("cc")
    #server.stop_threads = True
    l.stop_threads = True