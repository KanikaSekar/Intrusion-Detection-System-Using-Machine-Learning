from scapy.all import sniff, IP
from collections import defaultdict

packet_count = defaultdict(int)
alerted_ips = set()

THRESHOLD = 200

def packet_callback(packet):
    if IP in packet:
        src = packet[IP].src

        packet_count[src] += 1

        print(f"{src} -> Packet Count: {packet_count[src]}")

        if packet_count[src] > THRESHOLD and src not in alerted_ips:
            print("\n⚠ ALERT: Suspicious Activity Detected!")
            print(f"Source IP: {src}")
            print(f"Packets: {packet_count[src]}")
            print("-" * 50)

            alerted_ips.add(src)

print("Real-Time IDS Started...")
sniff(prn=packet_callback, count=50,store=False)