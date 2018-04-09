from __future__ import print_function

import sys
import json

from satori.rtm.client import make_client, SubscriptionMode

endpoint = "wss://open-data.api.satori.com"
appkey = "dC6c33Fbb5ECdAC1Ef2aB77dcBfBB0B0"
channel = "cryptocurrency-market-data"

def main():
    file = open("messages.txt","w") 
    with make_client(endpoint=endpoint, appkey=appkey) as client:
        print('Connected to Satori RTM!')

        class SubscriptionObserver(object):
            def on_subscription_data(self, data):
                for message in data['messages']:
                    print("Got message:", message)
                    json.dump(message, file)
                    file.write("\n")

        subscription_observer = SubscriptionObserver()
        client.subscribe(
            channel,
            SubscriptionMode.SIMPLE,
            subscription_observer,
            {'filter':'select * from `cryptocurrency-market-data` where exchange = "Bitstamp" and basecurrency = "USD" and cryptocurrency="BTC"'})

        try:
            while True:
                pass
        except KeyboardInterrupt:
            file.close()


if __name__ == '__main__':
    main()