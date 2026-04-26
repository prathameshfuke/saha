import sounddevice as sd

print("\n  ALL AUDIO DEVICES ON YOUR SYSTEM:\n")
print(sd.query_devices())
print("\n")
print(f"  Default INPUT device:  {sd.query_devices(kind='input')['name']}")
print(f"  Default OUTPUT device: {sd.query_devices(kind='output')['name']}")