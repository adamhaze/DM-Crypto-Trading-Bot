# Import WebSocket client library
import websocket
import _thread
import time
import json
import pandas as pd
from websocket import create_connection

# Connect to WebSocket API and subscribe to trade feed for XBT/USD and XRP/USD
ws = create_connection("wss://ws.kraken.com/")
# ws.send('{"event":"subscribe", "pair":["XBT/USD","XRP/USD"], "subscription":{"name":"ticker"}}')
ws.send('{"event":"subscribe", "pair":["XBT/USD"], "subscription":{"name":"ohlc"}}')

# Infinite loop waiting for WebSocket data
intervalData = pd.DataFrame(columns = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume'])
intervalStart = time.time()
while True:
    payload = json.loads(ws.recv())
    print(type(payload))
    if isinstance(payload, list):
        print(payload[1])
        intervalData = intervalData.append({'Date': payload[1][0], 'Symbol': payload[3], 'Open': payload[1][2], 'High': payload[1][3], 'Low': payload[1][4], 'Close': payload[1][5], 'Volume': payload[1][7]}, ignore_index=True)
    if (time.time() - intervalStart > 60):
        print(intervalData)
        # Send it somewhere
        intervalData = intervalData[0:0]
        intervalStart = time.time()
                                    

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


