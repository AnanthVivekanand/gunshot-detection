from simple_websocket_server import WebSocketServer, WebSocket
import threading
import json
import psutil
from time import localtime, strftime
import numpy as np
import soundfile as sf
import os

classifications = []
associated_times = []
raw_samples = []

class Handler(WebSocket):
    def handle(self):
        print("[client message] " + self.data)
        try:
            msg = json.loads(self.data)
            if (msg["status"] == "event_audio_request"):
                y = np.array(raw_samples[int(msg["data"])])
                y = y.flatten()
                if not (os.path.exists("audio_cache/" + str(hash(str(y))) + ".wav")):
                    sf.write("audio_cache/" + str(hash(str(y))) + ".wav", y, int(len(y) / 4), subtype='PCM_24')
                self.send_message(json.dumps({
                    "status": "audio_file",
                    "data": [int(msg["data"]), "../audio_cache/" + str(hash(str(y))) + ".wav"]
                }))
                
        except Exception as e:
            print("[Exception in handling] " + e)
        #echo message back to client
        #self.send_message(self.data)
        return

    def connected(self):
        print(self.address, 'connected')
        self.send_message(json.dumps({
            "status": "PING"
        }))
        self.send_message(json.dumps({
            "status": "past_classifications",
            "data": [classifications, associated_times]
        }))
        
    def handle_close(self):
        print(self.address, 'closed')

    def new_classification(self, c):
        self.send_message(json.dumps({
            "status": "classification",
            "data": [c, strftime("%H:%M:%S", localtime())]
        }))
        self.send_message(json.dumps({
            "status": "device_status",
            "data": {
                "cpu": psutil.cpu_percent(),
                "memory": psutil.virtual_memory().percent
            }
        }))        


class API(WebSocketServer):

    stop_threads = False

    def __init__(self):
        super().__init__('', 8000, Handler)
        # self.serve_forever()
        t = threading.Thread(target=self.serve_forever, daemon=True)
        t.start()

    def new_classification(self, c):
        classifications.append(c)
        associated_times.append(strftime("%H:%M:%S", localtime()))
        for handler in list(self.connections.values()):
            handler.new_classification(c)

    def new_sample(self, s):
        raw_samples.append(s)

#server = WebSocketServer('', 8000, Handler)
# print(server.connections)
# server.serve_forever()
