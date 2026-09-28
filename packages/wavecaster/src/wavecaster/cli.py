"""``wavecaster`` command line.

Examples::

    wavecaster info                                  # modems, codecs, profiles, devices
    wavecaster tx --profile sdr-narrow --device soapy:driver=uhd --freq 915e6 --text "hello"
    wavecaster rx --profile sdr-narrow --device uhd:type=b200 --freq 915e6
    wavecaster tx --profile audio-1200 --device "audio:ptt=rigctld" --file notes.txt
    c2enc 1300 - - < mic.raw | wavecaster tx --profile audio-1200 --device audio --stream
    wavecaster rx --profile audio-1200 --device audio --stream | c2dec 1300 - - | aplay -f S16_LE
    wavecaster tx --profile audio-1200 --device wav:tx=out.wav --text hi   # offline
    wavecaster bench --modem 16qam --fec ldpc:1024,0.5 --esn0 6 8 10
    wavecaster kiss-send --tnc localhost:8001 --src N0CALL --dst CQ --text "hi"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np

from . import __version__
from .channel import Channel
from .fec import available_codecs
from .link import Link
from .modulation import available_modems
from .phy import PROFILES, PHYConfig, Receiver, Transmitter, get_profile
from .radio import open_device


def _cfg(a) -> PHYConfig:
    cfg = get_profile(a.profile) if a.profile else PHYConfig()
    over = {}
    for k in ("modem", "fec", "sps"):
        v = getattr(a, k, None)
        if v is not None:
            over[k] = v
    if a.rate is not None:
        over["sample_rate"] = a.rate
    return cfg.with_(**over) if over else cfg


def _device(a, cfg):
    return open_device(a.device, sample_rate=cfg.sample_rate, center_freq=a.freq, sps=cfg.sps,
                       rx_gain=a.rx_gain, tx_gain=a.tx_gain, rx_antenna=a.rx_antenna,
                       tx_antenna=a.tx_antenna, bandwidth=a.bandwidth, tx_amplitude=a.tx_amplitude)


def _common(p):
    p.add_argument("--profile", choices=sorted(PROFILES), help="named PHY operating point")
    p.add_argument("--modem", help="modem spec, e.g. qpsk, 16qam, ofdm:const=16qam, custom:file=c.json")
    p.add_argument("--fec", help="codec spec, e.g. ldpc:1024,0.5 turbo:512,8 rs:255,223 conv:256")
    p.add_argument("--sps", type=int)
    p.add_argument("--rate", type=float, help="complex sample rate (overrides profile)")
    p.add_argument("--device", default="loopback", help="device URI (see wavecaster.radio)")
    p.add_argument("--freq", type=float, default=915e6, help="RF centre frequency (SDR)")
    p.add_argument("--rx-gain", type=float)
    p.add_argument("--tx-gain", type=float)
    p.add_argument("--rx-antenna")
    p.add_argument("--tx-antenna")
    p.add_argument("--bandwidth", type=float)
    p.add_argument("--tx-amplitude", type=float)


def cmd_info(a) -> int:
    info = {"version": __version__, "modems": available_modems(), "codecs": available_codecs(),
            "profiles": {k: {"sample_rate": v.sample_rate, "symbol_rate": v.symbol_rate, "modem": v.modem,
                             "fec": v.fec} for k, v in PROFILES.items()}}
    try:
        from .radio import enumerate_sdrs
        info["sdrs"] = enumerate_sdrs()
    except Exception as e:  # pragma: no cover
        info["sdrs"] = f"error: {e}"
    try:
        from .radio.audio import list_audio_devices
        info["audio"] = list_audio_devices()
    except Exception:
        info["audio"] = "sounddevice not installed"
    print(json.dumps(info, indent=2, default=str))
    return 0


def cmd_tx(a) -> int:
    cfg = _cfg(a)
    dev = _device(a, cfg).open()
    try:
        tx = Transmitter(cfg)
        if a.stream:
            link = Link(dev, cfg)
            src = iter(lambda: sys.stdin.buffer.read(a.chunk), b"")
            n = link.stream_out(src, mtu=a.mtu)
            logging.info("streamed %d segments", n)
            return 0
        if a.text is not None:
            data = a.text.encode()
        elif a.file:
            with open(a.file, "rb") as f:
                data = f.read()
        else:
            data = sys.stdin.buffer.read()
        if len(data) > a.mtu:
            link = Link(dev, cfg)
            n = link.send_message(data, mtu=a.mtu)
            logging.info("sent %d bytes in %d frames", len(data), n)
        else:
            for i in range(a.repeat):
                dev.transmit(tx.modulate(data, seq=i, pad=cfg.sps * 16))
                if a.interval and i + 1 < a.repeat:
                    time.sleep(a.interval)
            logging.info("sent %d x %d bytes (%d samples/frame, %.3f s airtime)", a.repeat, len(data),
                         tx.frame_samples(len(data)), tx.frame_samples(len(data)) / cfg.sample_rate)
    finally:
        dev.close()
    return 0


def cmd_rx(a) -> int:
    cfg = _cfg(a)
    dev = _device(a, cfg)
    with Link(dev, cfg, deliver_bad=a.show_bad) as link:
        if a.stream:
            for data in link.stream_in(timeout=a.timeout):
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            return 0
        n = 0
        for fr in link.frames(timeout=a.timeout, max_frames=a.count):
            n += 1
            rec = {"seq": fr.seq, "ok": fr.ok, "len": len(fr.payload), "snr_db": round(fr.snr_db, 2),
                   "cfo_hz": round(fr.cfo_hz, 1), "t": round(fr.sample_index / cfg.sample_rate, 4),
                   "fec_iter": fr.fec_iterations}
            if a.raw:
                sys.stdout.buffer.write(fr.payload)
                sys.stdout.buffer.flush()
            else:
                rec["payload"] = fr.payload.decode("utf-8", "replace")
                print(json.dumps(rec), flush=True)
        logging.info("received %d frames; receiver stats %s", n, link.rx.stats)
    return 0


def cmd_bench(a) -> int:
    """Monte-Carlo frame error rate through the full PHY (sync included)."""
    cfg = _cfg(a)
    rng = np.random.default_rng(a.seed)
    tx, results = Transmitter(cfg), []
    for esn0 in a.esn0:
        ok = 0
        t0 = time.time()
        for i in range(a.frames):
            rx = Receiver(cfg)
            payload = rng.bytes(a.size)
            ch = Channel(esn0, cfg.sps, cfo=a.cfo, phase=rng.uniform(0, 2 * np.pi),
                         delay=int(rng.integers(0, 4 * cfg.sps * 100)), seed=int(rng.integers(2**31)))
            y = ch(tx.modulate(payload, seq=i))
            frames = rx.feed(y) + rx.flush()
            ok += any(f.ok and f.payload == payload for f in frames)
        res = {"esn0_db": esn0, "fer": round(1 - ok / a.frames, 4), "frames": a.frames,
               "net_bitrate": round(8 * a.size / (tx.frame_samples(a.size) / cfg.sample_rate), 1),
               "sec_per_frame": round((time.time() - t0) / a.frames, 4)}
        results.append(res)
        print(json.dumps(res), flush=True)
    return 0


def cmd_kiss_send(a) -> int:
    from .radio.kiss import AX25Frame, KissTNC, ax25_encode
    tnc = KissTNC(a.tnc)
    try:
        info = a.text.encode() if a.text is not None else sys.stdin.buffer.read()
        tnc.send(ax25_encode(AX25Frame(a.dst, a.src, tuple(a.path or ()), info)), port=a.port)
    finally:
        tnc.close()
    return 0


def cmd_kiss_listen(a) -> int:
    from .radio.kiss import KissTNC, ax25_decode
    tnc = KissTNC(a.tnc)
    deadline = None if a.timeout is None else time.monotonic() + a.timeout
    try:
        while deadline is None or time.monotonic() < deadline:
            for port, raw in tnc.recv():
                try:
                    f = ax25_decode(raw)
                    rec = {"port": port, "src": f.src, "dst": f.dst, "path": f.path,
                           "info": f.info.decode("utf-8", "replace")}
                except ValueError:
                    rec = {"port": port, "raw": raw.hex()}
                print(json.dumps(rec), flush=True)
    finally:
        tnc.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wavecaster", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="list modems, codecs, profiles and attached devices").set_defaults(fn=cmd_info)

    p = sub.add_parser("tx", help="transmit")
    _common(p)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--text")
    g.add_argument("--file")
    g.add_argument("--stream", action="store_true", help="continuous stdin stream (e.g. codec2 audio)")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--interval", type=float, default=0.0)
    p.add_argument("--mtu", type=int, default=1024)
    p.add_argument("--chunk", type=int, default=64, help="stdin read size for --stream")
    p.set_defaults(fn=cmd_tx)

    p = sub.add_parser("rx", help="receive (prints JSON lines)")
    _common(p)
    p.add_argument("--timeout", type=float)
    p.add_argument("--count", type=int)
    p.add_argument("--raw", action="store_true", help="write payload bytes to stdout")
    p.add_argument("--stream", action="store_true", help="reassemble a --stream transmission to stdout")
    p.add_argument("--show-bad", action="store_true", help="also report CRC-failed frames")
    p.set_defaults(fn=cmd_rx)

    p = sub.add_parser("bench", help="frame error rate vs Es/N0 through the whole PHY")
    _common(p)
    p.add_argument("--esn0", type=float, nargs="+", default=[0, 2, 4, 6, 8, 10])
    p.add_argument("--frames", type=int, default=50)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--cfo", type=float, default=1e-4, help="cycles/sample")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("kiss-send", help="send an AX.25 UI frame via a KISS TNC (Direwolf etc.)")
    p.add_argument("--tnc", default="localhost:8001")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", default="CQ")
    p.add_argument("--path", nargs="*")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--text")
    p.set_defaults(fn=cmd_kiss_send)

    p = sub.add_parser("kiss-listen", help="print frames received by a KISS TNC")
    p.add_argument("--tnc", default="localhost:8001")
    p.add_argument("--timeout", type=float)
    p.set_defaults(fn=cmd_kiss_listen)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
