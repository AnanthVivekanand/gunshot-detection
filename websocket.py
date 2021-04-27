from simple_websocket_server import WebSocketServer, WebSocket
import threading
import json
import psutil
from time import localtime, strftime

classifications = []
associated_times = []

class Handler(WebSocket):
    def handle(self):
        print(self.data)
        # echo message back to client
        # self.send_message(self.data)
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

#server = WebSocketServer('', 8000, Handler)
# print(server.connections)
# server.serve_forever()
