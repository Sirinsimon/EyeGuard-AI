"""
Simple test script to verify alarm functionality
"""
import time
import numpy as np

print("Testing alarm methods...\n")

# Test 1: Windows beep (most reliable on Windows)
print("Test 1: Windows beep (winsound)")
try:
    import winsound
    print("Playing beep at 880Hz for 500ms...")
    winsound.Beep(880, 500)
    print("✓ Windows beep successful!\n")
    time.sleep(1)
except Exception as e:
    print(f"✗ Windows beep failed: {e}\n")

# Test 2: Pygame alarm
print("Test 2: Pygame alarm")
try:
    import pygame
    pygame.init()
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    
    # Generate sound
    sample_rate = 22050
    duration = 0.5
    frequency = 880
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = np.sin(2 * np.pi * frequency * t) * np.sin(2 * np.pi * 4 * t)
    audio_data = (wave * 32767 * 0.8).astype(np.int16)
    stereo_data = np.column_stack((audio_data, audio_data))
    
    sound = pygame.sndarray.make_sound(stereo_data)
    print("Playing pygame sound...")
    sound.play()
    time.sleep(1)
    print("✓ Pygame alarm successful!\n")
except Exception as e:
    print(f"✗ Pygame alarm failed: {e}\n")

# Test 3: Multiple beeps
print("Test 3: Multiple beeps (simulating sleep detection)")
try:
    import winsound
    for i in range(3):
        print(f"Beep {i+1}/3...")
        winsound.Beep(880, 500)
        time.sleep(0.8)
    print("✓ Multiple beeps successful!\n")
except Exception as e:
    print(f"✗ Multiple beeps failed: {e}\n")

print("Alarm test complete!")
print("\nIf you heard beeps, the alarm system is working correctly.")
print("If not, check your system volume and audio settings.")
