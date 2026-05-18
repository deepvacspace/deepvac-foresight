from opcua import Server, ua
import time
import random

server = Server()

server.set_endpoint("opc.tcp://127.0.0.1:12345")

uri = "http://dummy.opcua.server"
idx = server.register_namespace(uri)

objects = server.get_objects_node()
state = objects.add_object(idx, "State")

temp = state.add_variable(ua.NodeId("temp", idx), "temp", 25.0)
temp_raw = state.add_variable(ua.NodeId("temp_raw", idx), "temp_raw", 25000)
period_ms = state.add_variable(ua.NodeId("period_ms", idx), "period_ms", 5000)

temp.set_writable()
temp_raw.set_writable()
period_ms.set_writable()

server.start()
print("Dummy OPC UA server running at opc.tcp://127.0.0.1:12345")

try:
    while True:
        temp.set_value(temp.get_value() + random.uniform(-0.2, 0.2))
        temp_raw.set_value(int(temp.get_value() * 1000))
        time.sleep(0.5)
except KeyboardInterrupt:
    print("Stopping server...")
finally:
    server.stop()
