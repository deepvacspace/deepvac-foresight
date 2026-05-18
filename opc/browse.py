from opcua import Client

# 192.168.88.166:12345
#dummy 127.0.0.1:12345
ENDPOINT = "opc.tcp://192.168.88.174:12345"  

client = Client(ENDPOINT)
client.connect()

try:
    print("Connected to:", ENDPOINT)

    # 1) Show namespaces so you know the correct ns index
    ns = client.get_namespace_array()
    print("\nNamespaces:")
    for i, uri in enumerate(ns):
        print(f"  ns={i}: {uri}")

    # 2) Browse Objects tree (top-level)
    objects = client.get_objects_node()
    print("\nObjects children:")
    for ch in objects.get_children():
        print(" ", ch.get_browse_name(), "->", ch.nodeid)

    # 3) Deep-browse a bit (common place where variables live)
    print("\nDeep browse (2 levels):")
    for ch in objects.get_children():
        try:
            for gch in ch.get_children():
                print(" ", ch.get_browse_name(), "/", gch.get_browse_name(), "->", gch.nodeid)
        except Exception:
            pass

finally:
    client.disconnect()
