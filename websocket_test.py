# Import WebSocket client library
import websocket
import _thread
import time
from websocket import create_connection

# Connect to WebSocket API and subscribe to trade feed for XBT/USD and XRP/USD
ws = create_connection("wss://ws.kraken.com/")
ws.send('{"event":"subscribe", "subscription":{"name":"trade"}, "pair":["XBT/USD","XRP/USD"]}')


# Infinite loop waiting for WebSocket data
while True:
    print(ws.recv())



#################################################
desired = False
if desired:
    # Define WebSocket callback functions
    def ws_message(ws, message):
        # print(type(message))
        print("WebSocket thread: %s" % message)

    def ws_open(ws):
        # response: [channelID, [price, volume, time, side (buy vs sell), order type (market vs limit)]], channel name, asset pair
        ws.send('{"event":"subscribe", "subscription":{"name":"trade"}, "pair":["XBT/USD","XRP/USD"]}')


    def ws_thread(*args):
        ws = websocket.WebSocketApp("wss://ws.kraken.com/", on_open = ws_open, on_message = ws_message)
        ws.run_forever()

    # Start a new thread for the WebSocket interface
    _thread.start_new_thread(ws_thread, ())

    # Continue other (non WebSocket) tasks in the main thread
    while True:
        time.sleep(5)
        print("Main thread: %d" % time.time())


