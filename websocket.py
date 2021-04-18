from simple_websocket_server import WebSocketServer, WebSocket

class API(WebSocket):
    classifications = []

    def handle(self):
        # echo message back to client
        self.send_message(self.data)

    def connected(self):
        print(self.address, 'connected')
        self.send_message("PING")

    def handle_close(self):
        print(self.address, 'closed')


server = WebSocketServer('', 8000, API)
server.serve_forever()
