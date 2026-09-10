Here it is with **only formatting cleaned up**—no content added or changed:

# ADAPTIVE NOISE CANCELLATION

## CIRCUIT SCHEMATIC

Complete circuit diagram mapping external electret/MEMS reference and error microphones to the differential inputs of an I2S codec, interfaced via SPI/I2S to an ESP32 host controller with an inverted DAC output driving an anti-noise speaker.

## DASHBOARD

Diagnostic dashboard summarizing system performance in terms of STOI input, output, delta, and SNR gain.

## SIMULATION OF ANTINOISE

A simulation of the dashboard showing various acoustic scenarios such as ambient noisy and different combinations of noise and clean voice, showing how hypertuning the filter parameters generates the anti-noise that improves the speech metrics.

## GRAPH

The spectrogram received in one case to visually show how the addition of anti-noise does noise suppression and cleaning.


