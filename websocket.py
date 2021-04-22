from simple_websocket_server import WebSocketServer, WebSocket
import threading
import json

class Handler(WebSocket):
    classifications = []

    def handle(self):
        # echo message back to client
        # self.send_message(self.data)
        return

    def connected(self):
        print(self.address, 'connected')
        self.send_message(json.dumps({
            "status": "PING"
        }))

    def handle_close(self):
        print(self.address, 'closed')

    def new_classification(self, c):
        self.classifications.append(c)
        self.send_message(json.dumps(c))

class API(WebSocketServer):
    
    stop_threads = False        

    def __init__(self):
        super().__init__('', 8000, Handler)
        t = threading.Thread(target=self.serve_forever)
        t.daemon = True
        t.start()
    
    def new_classification(self, c):
        for handler in list(self.connections.values()):
            handler.new_classification(c)

#server = WebSocketServer('', 8000, Handler)
#print(server.connections)
#server.serve_forever()