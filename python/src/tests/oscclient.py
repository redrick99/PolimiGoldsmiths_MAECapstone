from pythonosc.udp_client import SimpleUDPClient
from pythonosc import osc_message_builder

ip = "127.0.0.1"
port = 1337

client = SimpleUDPClient(ip, port)  # Create client

while True:
    user_input = input("Input (start/stop/channel): ")
    if user_input == "start": client.send_message("/START", 123)
    if user_input == "channel":
        channel = input("Channel Number: ")
        instrument = input("Instrument (VOICE/GUITAR/PIANO/STRINGS/DRUMS): ")
        msg = osc_message_builder.OscMessageBuilder("/ch_settings")
        msg.add_arg(int(channel), osc_message_builder.OscMessageBuilder.ARG_TYPE_INT)
        msg.add_arg(str(instrument), osc_message_builder.OscMessageBuilder.ARG_TYPE_STRING)
        client.send(msg.build())
    if user_input == "stop": 
        client.send_message("/STOP", 123)
        break
