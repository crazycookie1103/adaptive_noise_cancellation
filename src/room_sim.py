import numpy as np
import pyroomacoustics as pra


def _simulate_sources(
    speech,
    noise,
    fs,
    rt60,
    room_dim,
    speech_pos,
    noise_pos,
    primary_mic,
    reference_mic,
    use_speech,
    use_noise
):
    e_absorption, max_order = pra.inverse_sabine(
        rt60,
        room_dim
    )

    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials=pra.Material(e_absorption),
        max_order=max_order
    )

    if use_speech:
        room.add_source(
            speech_pos,
            signal=speech
        )

    if use_noise:
        room.add_source(
            noise_pos,
            signal=noise
        )

    room.add_microphone_array(
        np.c_[primary_mic, reference_mic]
    )

    room.simulate()

    return room.mic_array.signals


def simulate_two_mic_room(
    speech,
    noise,
    fs,
    rt60=0.15,
    room_dim=(4.0, 3.0, 2.5),
    speech_pos=(1.5, 1.5, 1.2),
    noise_pos=(3.2, 2.5, 1.0),
    primary_mic=(1.6, 1.55, 1.2),
    reference_mic=(1.6, 1.35, 1.2),
    target_snr_db=None
):

    # ---------------------------------------------------------
    # 1. Simulate SPEECH only
    # ---------------------------------------------------------
    speech_signals = _simulate_sources(
        speech,
        noise,
        fs,
        rt60,
        room_dim,
        speech_pos,
        noise_pos,
        primary_mic,
        reference_mic,
        use_speech=True,
        use_noise=False
    )

    # ---------------------------------------------------------
    # 2. Simulate NOISE only
    # ---------------------------------------------------------
    noise_signals = _simulate_sources(
        speech,
        noise,
        fs,
        rt60,
        room_dim,
        speech_pos,
        noise_pos,
        primary_mic,
        reference_mic,
        use_speech=False,
        use_noise=True
    )

    # ---------------------------------------------------------
    # 3. Make all signals the same length
    # ---------------------------------------------------------
    n = min(
        len(speech),
        speech_signals.shape[1],
        noise_signals.shape[1]
    )

    clean_primary = speech_signals[0][:n]

    noise_primary = noise_signals[0][:n]

    noise_reference = noise_signals[1][:n]

    # ---------------------------------------------------------
    # 4. Re-establish target SNR at PRIMARY microphone
    #
    # Room propagation can change speech/noise energy because
    # the sources are at different distances from the mics.
    # ---------------------------------------------------------
    if target_snr_db is not None:

        speech_rms = np.sqrt(
            np.mean(clean_primary ** 2) + 1e-12
        )

        noise_rms = np.sqrt(
            np.mean(noise_primary ** 2) + 1e-12
        )

        target_noise_rms = (
            speech_rms /
            (10 ** (target_snr_db / 20.0))
        )

        scale = target_noise_rms / noise_rms

        noise_primary = noise_primary * scale
        noise_reference = noise_reference * scale

    # ---------------------------------------------------------
    # 5. Construct primary microphone signal
    # ---------------------------------------------------------
    primary = clean_primary + noise_primary

    return primary, noise_reference, clean_primary