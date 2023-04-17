from abc import ABC, abstractmethod
from multiprocessing import Lock
from pythonosc import udp_client, osc_message_builder, osc_message
from pythonosc.dispatcher import Dispatcher
from modules.utilities import Instruments, Debugger
import modules.utilities as u

"""
Class declarations
"""
class Message(ABC):
    """
    Abstract Class representing the message with output data to be sent to the visualizer
    """

    def __init__(self, data, channel: int):
        """
        Constructor
        """
        self._data = data
        self.channel = channel
        self.address = "/blank_ch"+str(channel)

    @abstractmethod
    def to_osc(self):
        """
        Abstract method to convert the message into its OSC representation
        """
        pass
    
class LFAudioMessage(Message):
    """
    Message containing Low Level Features
    """

    def __init__(self, data, channel: int, instrument: Instruments):
        """
        Constructor

        Parameters:
        - data: data to send
        - channel: corresponding channel of the sound card
        - instrument: instrument of the track
        """
        super().__init__(data, channel=channel)
        self.address = "/LFmsg_ch"+str(channel)
        self._instrument = instrument

    def to_osc(self):
        """
        Converts the message into its OSC representation with its own OSC address.
        Appends the instrument type as a string argument and all the Low Level Features as floats
        """
        msg = osc_message_builder.OscMessageBuilder(self.address)
        msg.add_arg(self._instrument.get_string(), osc_message_builder.OscMessageBuilder.ARG_TYPE_STRING)
        for d in self._data:
            msg.add_arg(float(d), osc_message_builder.OscMessageBuilder.ARG_TYPE_FLOAT)
        return msg.build()
    
class HFAudioMessage(Message):
    """
    Message containing High Level Features
    """

    def __init__(self, data, channel:int):
        """
        Constructor

        Parameters:
        - data: data to send
        - channel: corresponding channel of the sound card
        """
        super().__init__(data, channel=channel)
        self.address = "/HFmsg_ch"+str(channel)

    def to_osc(self):
        """
        Converts the message into its OSC representation with its own OSC address.
        """
        msg = osc_message_builder.OscMessageBuilder(self.address)
        msg.add_arg(self._data, osc_message_builder.OscMessageBuilder.ARG_TYPE_STRING)
        
        return msg.build()

class ConnectionHandler(ABC):    
    """
    Abstract class to represent the handler of the connection from the python script to the visualization program
    """
    def __init__(self, address:str, port:int):
        """
        Constructor

        Parameters:
        - address: net address of the receiver as a string
        - port: net port of the receiver as an int
        """
        self._address = address
        self._port = port
        self._lock = Lock()

    @staticmethod
    @abstractmethod
    def get_instance(address:str, port:int):
        """
        Abstract method to get the Singleton instance of the class
        """
        pass
    
    @abstractmethod
    def send_message(self, message: Message):
        """
        Abstract method to send messages
        """
        pass

class OSCConnectionHandler(ConnectionHandler):
    """
    Singleton class that handles the communication between the python script and the visualization program via OSC messages
    """
    __instance = None

    def __init__(self, address:str, port:int):
        """
        Singleton constructor. Starts the OSC communication channel
        """
        if OSCConnectionHandler.__instance is not None:
            raise Exception("Tried to instantiate ConnectionHandler multiple times")
        OSCConnectionHandler.__instance = self

        super().__init__(address=address, port=port)
        try:
            self.__client = udp_client.SimpleUDPClient(self._address, self._port)
        except Exception as e:
            raise e
            raise Exception("Failed to initialize client socket")
        
    @staticmethod
    def get_instance(address:str, port:int):
        """
        Returns the running Singleton Instance of the class
        """
        if OSCConnectionHandler.__instance is None:
            OSCConnectionHandler(address, port)
        return OSCConnectionHandler.__instance
    
    def send_message(self, message: Message):
        """
        Sends a message over the net as an OSC message

        Parameters:
        - message: message to be sent
        """
        self._lock.acquire()

        try:
            self.__client.send_message(message.address, message.to_osc())
        except Exception as e:
            print("Error while sending message")
        finally:
            self._lock.release()

"""
Incoming OSC Message Handlers
"""
def default_handler(address, *args):
    dbg = Debugger()
    dbg.print_warning("Received message with unrecognized address")

def handler_ch_settings(address, fixed_args, *osc_args):
    dbg = Debugger()
    dbg.print_info("Received ch_settings message")
    queues = fixed_args[0]
    try:
        track = osc_args[0]
        if track < 0 or track >= u.CHANNELS: raise Exception("Invalid channel number (was "+str(track)+")")
        instrument = Instruments.from_string(osc_args[1])
        setting = [track, instrument]
    except Exception as e:
        dbg.print_error(e)
        return

    for q in queues:
        q.put(setting)

def handler_start(address, *args):
    dbg = Debugger()
    print()
    dbg.print_success("Received starting message\n", True)
    u.EXTERNAL_OSC_CONTROLLER_CONNECTED = True

def handler_stop(address, *args):
    dbg = Debugger()
    print()
    dbg.print_info("Received stopping message", True)
    u.STOP = True

def create_dispatcher(settings_queues):
    dispatcher = Dispatcher()
    dispatcher.map(u.OSC_START_ADDRESS, handler_start)
    dispatcher.map(u.OSC_STOP_ADDRESS, handler_stop)
    dispatcher.map(u.OSC_CHANNEL_SETTINGS_ADDRESS, handler_ch_settings, settings_queues)
    dispatcher.set_default_handler(default_handler)
    return dispatcher
    