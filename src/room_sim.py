import numpy as np
import pyroomacoustics as pra

def simulate_two_mic_room(speech, noise, fs, rt60=0.15,
                           room_dim=(4.0, 3.0, 2.5),
                           speech_pos=(1.5, 1.5, 1.2),
                           noise_pos=(3.2, 2.5, 1.0),
                           primary_mic=(1.6, 1.55, 1.2),
                           reference_mic=(1.6, 1.35, 1.2)):
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption),
                        max_order=max_order)
    room.add_source(speech_pos, signal=speech)
    room.add_source(noise_pos, signal=noise)
    room.add_microphone_array(np.c_[primary_mic, reference_mic])
    room.simulate()
    sig = room.mic_array.signals
    n = min(sig.shape[1], len(speech))
    return sig[0][:n], sig[1][:n]