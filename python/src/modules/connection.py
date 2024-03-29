from __future__ import annotations
from abc import ABC, abstractmethod
from multiprocessing import Lock
from pythonosc import udp_client, osc_message_builder, osc_message
from pythonosc.dispatcher import Dispatcher
import modules.utilities as ut
import modules.custom_exceptions as ce
from modules.default_parameters import OSC_MESSAGES_PARAMETERS


class Message(ABC):
    """Abstract Class representing the message with output data to be sent to the visualizer. Can be inherited to
    implement custom message types with custom parameters.
    """

    def __init__(self, data, channel: int):
        """Constructor for the Message class.

        **Args:**

        `data`: Data to send.

        `channel`: Index of the track or input channel.

        **Class Attributes:**

        `_data`: Data to send.

        `channel`: Index of the track or input channel.

        `address`: `OSC` address of the message.
        """
        self._data = data
        self.channel = channel
        self.address = "/blank_ch"+str(channel)

    @abstractmethod
    def to_osc(self) -> osc_message.OscMessage:
        """Abstract method to convert a message into its `OSC` representation.

        **Returns:**

        The `OSC` representation of the message.
        """
        pass


class LFAudioMessage(Message):
    """Message containing Low-level Features.
    """

    def __init__(self, data, channel: int, instrument: ut.Instruments):
        """Constructor for the LFAudioMessage class.

        **Args:**

        `data`: Data to send.

        `channel`: Index of the track or input channel.

        `instrument`: Instrument of the channel.
        """
        super().__init__(data, channel=channel)
        self.address = "/LFmsg_ch"+str(channel)
        self._instrument = instrument

    def to_osc(self) -> osc_message.OscMessage:
        """Converts message into its OSC representation with its own OSC address.
        Appends the instrument type as a string argument and all the Low-level features as floats.

        **Returns:**

        The `OSC` representation of the message.
        """
        msg = osc_message_builder.OscMessageBuilder(self.address)
        msg.add_arg(self._instrument.get_string(), osc_message_builder.OscMessageBuilder.ARG_TYPE_STRING)
        for d in self._data:
            msg.add_arg(float(d), osc_message_builder.OscMessageBuilder.ARG_TYPE_FLOAT)
        return msg.build()


class HFAudioMessage(Message):
    """Message containing High-level Features.
    """

    def __init__(self, data, channel: int):
        """Constructor for the HFAudioMessage class.

        **Args:**

        `data`: Data to send.

        `channel`: Index of the track or input channel.
        """
        super().__init__(data, channel=channel)
        self.address = "/HFmsg_ch"+str(channel)

    def to_osc(self) -> osc_message.OscMessage:
        """Converts message into its `OSC` representation with its own `OSC` address.

        **Returns:**

        The `OSC` representation of the message.
        """
        msg = osc_message_builder.OscMessageBuilder(self.address)
        for d in self._data:
            msg.add_arg(d, osc_message_builder.OscMessageBuilder.ARG_TYPE_FLOAT)
        
        return msg.build()


class ConnectionHandler(ABC):    
    """Abstract class to handle the connection between the python script and the external world. Can be inherited to
    implement custom methods of sending messages.
    """
    def __init__(self, address: str, port: int):
        """Constructor for the ConnectionHandler class.

        **Args:**

        `address`: Net address of the receiver as a string.

        `port`: Net port of the receiver as an int.

        **Class Attributes:**

        `_address`: Net address of the receiver as a string.

        `_port`: Net port of the receiver as an int.

        `_lock`: Mutex lock used for synchronization purposes.
        """
        self._address = address
        self._port = port
        self._lock = Lock()

    @staticmethod
    @abstractmethod
    def get_instance(address: str, port: int) -> ConnectionHandler:
        """Abstract method to get the Singleton instance of the class.

        **Args:**

        `address`: Net address to assign to the ConnectionHandler Singleton instance.

        `address`: Net port to assign to the ConnectionHandler Singleton instance.

        **Returns:**

        The Singleton instance of the class.
        """
        pass
    
    @abstractmethod
    def send_message(self, message: Message):
        """Abstract method used to send a message.

        **Args:**

        `message`: Message to send over the network.
        """
        pass


class OSCConnectionHandler(ConnectionHandler):
    """Singleton class that handles the communication between the python script and the external world via `OSC`
    messages.
    """
    __instance = None

    def __init__(self, address: str, port: int):
        """Singleton constructor. Starts the `OSC` communication channel.

        **Args:**

        `address`: Net address to assign to the ConnectionHandler Singleton instance.

        `address`: Net port to assign to the ConnectionHandler Singleton instance.
        """
        if OSCConnectionHandler.__instance is not None:
            raise ce.SetupException("Tried to instantiate ConnectionHandler multiple times")
        OSCConnectionHandler.__instance = self

        super().__init__(address=address, port=port)
        self.__client = udp_client.SimpleUDPClient(self._address, self._port)
        
    @staticmethod
    def get_instance(address: str, port: int) -> OSCConnectionHandler:
        """Returns the currently running Singleton Instance of the class.

        **Args:**

        `address`: Net address to assign to the ConnectionHandler Singleton instance.

        `address`: Net port to assign to the ConnectionHandler Singleton instance.

        **Returns:**

        The Singleton instance of the class.
        """
        if OSCConnectionHandler.__instance is None:
            OSCConnectionHandler(address, port)
        return OSCConnectionHandler.__instance
    
    def send_message(self, message: Message):
        """Sends a message over the network as an `OSC` message.

        **Args:**

        `message`: Message to be sent.
        """
        self._lock.acquire()

        try:
            self.__client.send_message(message.address, message.to_osc())
        except Exception:
            print("Error while sending message")
        finally:
            self._lock.release()


# Incoming OSC Message Handlers
def default_handler(address, *args):
    """Handles OSC messages with unrecognized addresses.

    **Args:**
    `address`: OSC address of the received message.
    `*args`: Arguments of the OSC message.
    """
    ut.print_warning("Received message with unrecognized address")


def handler_ch_settings(address, fixed_args, *osc_args):
    """Handles an incoming OSC message to set settings.

    **Args:**
    `address`: OSC address of the received message.
    `fixed_args`: Function arguments passed down from the calling higher function.
    `*osc_args`: Arguments of the OSC message.
    """
    ut.print_info("Received ch_settings message")
    queues = fixed_args[0]
    channels = fixed_args[1]
    try:
        track = osc_args[0]
        if track < 0 or track >= channels: raise ce.MessageReceiveException("Invalid channel number (was " + str(track) + ")")
        instrument = ut.Instruments.from_string(osc_args[1])
        setting = [track, instrument]
    except Exception as e:
        ut.print_error("Something bad happened while handling Channel Settings message")
        ut.print_dbg(e)
        return

    for q in queues:
        q.put(setting)


def handler_start(address, *args):
    """Starts execution of the application when a start message is received

    **Args:**
    `address`: OSC address of the received message.
    `*args`: Arguments of the OSC message.
    """
    print()
    ut.print_success("Received starting message\n")


def handler_stop(address, *args):
    """Stops execution of the application when a stop message is received

    **Args:**
    `address`: OSC address of the received message.
    `*args`: Arguments of the OSC message.
    """
    print()
    ut.print_info("Received stopping message")


def create_dispatcher(settings_queues, channels):
    """Creates a dispatcher to map incoming OSC messages into functions.

    **Args:**

    `settings_queues`: Multiprocessing queues in which to put incoming settings.
    `channels`: Max number of channels currently being processed.

    **Returns:**

    A dispatcher for an OSC server.
    """
    dispatcher = Dispatcher()
    dispatcher.map(OSC_MESSAGES_PARAMETERS['inStart'], handler_start)
    dispatcher.map(OSC_MESSAGES_PARAMETERS['inStop'], handler_stop)
    dispatcher.map(OSC_MESSAGES_PARAMETERS['inChannelSettings'], handler_ch_settings, settings_queues, channels)
    dispatcher.set_default_handler(default_handler)
    return dispatcher