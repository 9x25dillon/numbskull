# wavecaster

Software-defined burst modem: FEC → modulation → PHY (sync) → radio I/O, each layer usable on its own.

```
pip install -e packages/wavecaster            # numpy + scipy only
pip install -e "packages/wavecaster[audio,serial]"   # sound card + serial PTT
# SDR bindings come from system packages: python3-soapysdr / python3-uhd (apt) or conda-forge
```

## Layers

```
 bytes ──► CRC32 ──► FEC encode ──► scramble ──► Modem.modulate ──┐
                                                                  │ payload segment
 header(8B) ──► conv K7 ──► BPSK ─┐                               │
 m-sequence preamble (BPSK) ──────┴─► RRC ─► SC segment ──────────┴──► burst ──► RadioDevice.transmit
                                                                                       │
 RadioDevice.read ─► Receiver.feed ─► MF ─► differential detect ─► segmented verify ─► CFO cascade
   ─► gain/SNR ─► header Viterbi+CRC16 ─► data-aided refine ─► Modem.demodulate ─► descramble
   ─► FEC decode ─► CRC32 ─► Frame
```

| Module | Contents |
|---|---|
| `fec/` | `hamming74`, `rs:n,k` (Berlekamp–Massey/Chien/Forney, errors+erasures, GMD soft retries, shortened codes), `ldpc:n,rate[,dv,seed]` (PEG construction, alist import/export, RREF encoder, sum-product or normalised min-sum), `turbo:K[,iters[,p]]` (LTE 13/15 RSC, S-random interleaver, log-MAP BCJR, optional puncturing), `conv:k` (K=7 171/133 soft Viterbi) |
| `modulation/` | `Constellation` (points + labels → map, exact/max-log LLR demap), PSK/QAM/PAM/APSK factories, JSON specs; waveforms: single-carrier RRC with pilot tracking, DSSS, CP-OFDM (training + CPE pilots), noncoherent CP-M-FSK |
| `phy` | Framing, two-stage acquisition, CFO lag cascade, streaming `Receiver`, named `PROFILES` |
| `radio/` | `SoapyDevice` (USRP via SoapyUHD, HackRF, LimeSDR, Pluto, bladeRF…), `UHDDevice` (native, timed TX), `AudioDevice` (sound card + PTT: serial RTS/DTR, Hamlib `rigctld`), `KissTNC` + AX.25 UI, `FileDevice` (cf32/cs16/cu8/cs8), `WavDevice`, `LoopbackDevice` |
| `link` | Threaded real-time TX/RX, message segmentation, low-latency stream transport |

## Live transmission

> Transmitting requires authorisation for the frequency, power and emission type you use (e.g. an amateur licence on amateur bands, or ISM-band rules). Identify as your licence requires; `kiss-send` and AX.25 framing carry callsigns.

**USRP (native UHD)**
```bash
wavecaster rx --profile sdr-narrow --device uhd:type=b200 --freq 915e6 --rx-gain 40
wavecaster tx --profile sdr-narrow --device uhd:type=b200 --freq 915e6 --tx-gain 50 --text "hello"
```

**Any SoapySDR device** (USRP through SoapyUHD, HackRF, LimeSDR, PlutoSDR):
```bash
wavecaster tx --profile sdr-ofdm --device soapy:driver=hackrf --freq 433.92e6 --tx-gain 20 --file data.bin
```

**Transceiver via sound card** (DigiRig/SignaLink/AIOC; PTT through Hamlib):
```bash
rigctld -m <model> -r /dev/ttyUSB0 &          # or: --device "audio:ptt=serial:/dev/ttyUSB0:rts"
wavecaster tx --profile audio-1200 --device "audio:out=3,ptt=rigctld" --text "CQ de N0CALL"
wavecaster rx --profile audio-1200 --device "audio:in=3"
```

**Real-time audio streaming** (Codec2 voice at 1300 bit/s needs a profile with a higher net rate, e.g. `audio-2400`):
```bash
arecord -f S16_LE -r 8000 | c2enc 1300 - - | wavecaster tx --profile audio-2400 --device audio:ptt=rigctld --stream
wavecaster rx --profile audio-2400 --device audio --stream | c2dec 1300 - - | aplay -f S16_LE -r 8000
```

**External hardware/software modems over KISS** (Direwolf, Mobilinkd, TNC-Pi):
```bash
wavecaster kiss-send --tnc localhost:8001 --src N0CALL --dst APRS --path WIDE1-1 --text "test"
wavecaster kiss-listen --tnc localhost:8001
```

**Offline / lab**: `--device file:tx=burst.cf32` produces GNU Radio-compatible IQ files; `--device wav:tx=out.wav` produces audio you can play into any radio.

Python:
```python
from wavecaster.link import Link
from wavecaster.phy import get_profile
from wavecaster.radio import open_device

cfg = get_profile("sdr-narrow")
dev = open_device("uhd:type=b200", sample_rate=cfg.sample_rate, center_freq=915e6, rx_gain=40, tx_gain=50)
with Link(dev, cfg) as link:
    link.send(b"hello")
    for frame in link.frames(timeout=10):
        print(frame.payload, frame.snr_db, frame.cfo_hz)
```

## Custom modulation

A linear modulation is data: points plus bit labels.

```json
{"name": "hex8", "points": [[1,0],[0.5,0.87],[-0.5,0.87],[-1,0],[-0.5,-0.87],[0.5,-0.87],[0,0],[2,0]],
 "labels": [0,1,3,2,6,7,5,4]}
```
```bash
wavecaster tx --modem custom:file=hex8.json --fec ldpc:1024,0.5 ...
```

Or register it in code, which also makes it available inside OFDM and DSSS:
```python
from wavecaster.modulation import Constellation, register_constellation
register_constellation("hex8", lambda: Constellation.from_spec(spec))
# usable as: "hex8", "ofdm:const=hex8", "dsss:const=hex8,degree=6"
```

For a new *waveform* (CPM, chirp spread spectrum, OTFS…), subclass `Modem` (`modulate`, `demodulate` → LLRs, `num_samples`, `bits_per_symbol`) and call `register_modem(name, factory)`. Out-of-tree packages can expose factories under the `wavecaster.modems` / `wavecaster.constellations` entry-point groups. See `examples/custom_modulation.py`.

Contract every modem must satisfy (`modulation/modems.py`): a segment has unit average power per sample; `demodulate` receives it time-aligned, coarse-CFO-corrected and gain-normalised; `noise_var` is the per-sample complex noise variance after normalisation.

## Measured performance

Full PHY including acquisition, AWGN, CFO 1e-4 cycles/sample, random phase and delay, 128-byte payloads, 30 frames per point (`wavecaster bench`). FER resolution is ±1/30, so treat single-digit percentages as indicative.

| Profile | Waveform / FEC | Rate (S/s) | Net bit/s | FER=0 at Es/N0 | FER at lower Es/N0 |
|---|---|---|---|---|---|
| `sdr-robust` | BPSK / turbo r≈1/3, 511-sym preamble, 4× header | 1 M | 21.8 k | −2 dB | 33 % @ −3 dB |
| `sdr-narrow` | QPSK / LDPC(1024,512) | 250 k | 17.1 k | 3 dB | 67 % @ 2 dB |
| `sdr-wide` | 16-QAM / turbo punctured r≈1/2 | 2 M | 380 k | 8 dB | 97 % @ 6 dB |
| `sdr-ofdm` | OFDM 48×16-QAM / LDPC r½ | 2 M | 782 k | 16 dB* | 23 % @ 14 dB |
| `sdr-dsss` | DSSS-31 BPSK / conv K7 | 1 M | 1.26 k | 4 dB | 7 % @ 2 dB (header-limited) |
| `audio-1200` | 1200 Bd QPSK / conv K7 | 48 k | 734 | 5 dB | 7 % @ 4 dB |
| `audio-2400` | 2400 Bd 8PSK / LDPC(512,256) | 48 k | 2.06 k | 8 dB | 40 % @ 6 dB |
| `audio-ofdm` | 48-carrier OFDM QPSK / LDPC r½ | 48 k | 861 | ≤12 dB | |
| `audio-afsk` | 1200 Bd CP-FSK / conv K7 | 48 k | 425 | ≤8 dB | |

\*Es/N0 is referenced to the single-carrier preamble. OFDM spreads the same power over ~4.4× more bandwidth at sps=4, so its per-carrier SNR is ~6 dB lower.

Receiver throughput (acquisition on a continuous noise stream, one CPU core, numpy): audio profiles 54–72× real time, `sdr-narrow` 15×, `sdr-dsss` 4.8×, `sdr-wide`/`sdr-ofdm` (2 MS/s) ~2.3×, `sdr-robust` 1.8×.

## Known limits (engineering, Level 0)

- **No symbol-timing recovery or clock-offset tracking.** Timing comes from the preamble peak (±½ sample). A sample-clock offset of *p* ppm drifts by *p*·N·1e-6 samples over an N-sample burst. Keep bursts under ~10⁵ samples at 20 ppm, or use an SDR with a GPSDO/shared reference. A Gardner or Mueller–Müller loop is the next PHY addition.
- **The DSSS preamble/header are not spread.** Acquisition sensitivity is therefore the unspread header's (≈0–2 dB Es/N0 per chip), not the payload's 15 dB processing gain. Spread acquisition needs a 2-D time×frequency search.
- **CFO acquisition range:** detection tolerates up to ±Rs/2. The estimator is accurate from preamble length and SNR (lag cascade, near-CRB). Multipath is equalised only by OFDM (single-tap gain for single-carrier).
- **OFDM** uses one training symbol with a 3-carrier moving-average channel estimate: fine for delay spread ≪ CP, lossy for highly frequency-selective channels.
- **Hardware drivers are verified against fake SDK modules** that assert the documented call sequences: stream setup, END_BURST, start/end-of-burst metadata, overflow handling. They are not yet verified on physical radios. First hardware bring-up should use `rx` with a known signal source and `bench`-style loopback via attenuators.
