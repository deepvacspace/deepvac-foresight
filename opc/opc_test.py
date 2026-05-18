from opcua import Client
import json
import time

client = Client("opc.tcp://192.168.88.174:12345") # Real PC 192.168.88.144:12345
client.connect()

readings = []

start_time = time.time()

while time.time() - start_time < 30.0:
    reads = {
        "timestamp": time.time(),
        "temp": client.get_node("ns=2;s=Testa chamber.temp").get_value(),
        "temp_raw": client.get_node("ns=2;s=Testa chamber.temp_raw").get_value(),
        "temp_ref": client.get_node("ns=2;s=Testa chamber.temp_ref").get_value(),
        "state": client.get_node("ns=2;s=Testa chamber.state").get_value(),
        "temp_u": client.get_node("ns=2;s=Testa chamber.temp_u").get_value(),
        "temp_u_p": client.get_node("ns=2;s=Testa chamber.temp_u_p").get_value(),
        "temp_u_i": client.get_node("ns=2;s=Testa chamber.temp_u_i").get_value(),
        "temp_u_d": client.get_node("ns=2;s=Testa chamber.temp_u_d").get_value(),
        "temp_kp": client.get_node("ns=2;s=Testa chamber.temp_kp").get_value(),
        "temp_ki": client.get_node("ns=2;s=Testa chamber.temp_ki").get_value(),
        "temp_kd": client.get_node("ns=2;s=Testa chamber.temp_kd").get_value(),
        "temp_pid_idx": client.get_node("ns=2;s=Testa chamber.temp_pid_idx").get_value()
    }
    readings.append(reads)
    print(reads)
    time.sleep(1.0)

json_data = json.dumps(readings, indent=2)
print(json_data)

with open("dummy.json", "w") as f:
    json.dump(readings, f, indent=2)

#client.get_node("ns=2;s=Testa chamber.temp").set_value(118.8)
#final_temp = client.get_node("ns=2;s=Testa chamber.temp").get_value()

#print("Final temp " + str(final_temp))

client.disconnect()


#temp = client.get_node("ns=2;s=temp").get_value()
#temp_raw = client.get_node("ns=2;s=temp_raw").get_value()
# chamber = 200 ms < 500 ms
#client.get_node("ns=2;s=temp1").set_value(35.0)
