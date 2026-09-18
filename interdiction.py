"""
RfLord Interdiction Module — Advanced HackRF One Jamming & Interdiction

Provides multiple jamming modes for detected RF threats:
- Protocol-specific FPV drone jamming (ExpressLRS/CRSF, BetaFPV SmartProtocol)
- GPS L1/L2/GPS L5 spoofing/sweep jamming
- Chirp/spread spectrum interference
- Tone-based selective disruption (DTMF, 400Hz tone)
- Sweep-band wideband noise jamming
- Interactive per-frequency strategy selection

Usage:
    from interdiction import InterdictionEngine
    ie = InterdictionEngine()
    ie.process_signal(freq_mhz, signal_info)
    ie.stop_all()
"""
import logging
import os
import subprocess
import time
import math
import threading
import tempfile
import numpy as np

log = logging.getLogger("rflord")

# FPV Protocol frequency bands and characteristics
FPV_PROTOCOLS = {
    'expresslrs': {
        'name': 'ExpressLRS/CRSF (ELRS)',
        'bands': [
            {'freq': 433.5, 'bw_mhz': 2, 'protocol': 'crsf', 'modulation': 'LR1200/Fast'},
            {'freq': 868.0, 'bw_mhz': 2, 'protocol': 'crsf', 'modulation': 'LR1200/Fast'},
            {'freq': 915.0, 'bw_mhz': 3, 'protocol': 'crsf', 'modulation': 'LR600/LR1200'},
            {'freq': 2408.0, 'bw_mhz': 40, 'protocol': 'elrs_fpv', 'modulation': 'CRSF/ExpressLRS'},
        ],
        'reconnect_interval': 3,  # seconds between CRSF reconnect attempts
    },
    'smartprotocol': {
        'name': 'BetaFPV SmartProtocol',
        'bands': [
            {'freq': 2412.0, 'bw_mhz': 10, 'protocol': 'bp_4m', 'modulation': 'SmartProtocol'},
            {'freq': 2437.0, 'bw_mhz': 10, 'protocol': 'bp_4m', 'modulation': 'SmartProtocol'},
            {'freq': 2462.0, 'bw_mhz': 10, 'protocol': 'bp_4m', 'modulation': 'SmartProtocol'},
        ],
    },
    'tbs_crossfire': {
        'name': 'TBS Crossfire',
        'bands': [
            {'freq': 433.5, 'bw_mhz': 2, 'protocol': 'crossfire', 'modulation': '868Mbps'},
            {'freq': 868.0, 'bw_mhz': 2, 'protocol': 'crossfire', 'modulation': '868Mbps'},
            {'freq': 915.0, 'bw_mhz': 3, 'protocol': 'crossfire', 'modulation': '576kbps'},
        ],
    },
    'elrs_rx': {
        'name': 'ExpressLRS RX (bidirectional)',
        'bands': [
            {'freq': 2408.0, 'bw_mhz': 40, 'protocol': 'elrs_ack', 'modulation': 'CRSF ACK'},
        ],
    },
}

# GPS signal bands for spoofing/jamming
GPS_BANDS = [
    {'name': 'GPS L1', 'freq_mhz': 1575.42, 'bw_hz': 10e6},
    {'name': 'GPS L2', 'freq_mhz': 1227.60, 'bw_hz': 10e6},
    {'name': 'GPS L5', 'freq_mhz': 1176.45, 'bw_hz': 10e6},
    {'name': 'GLONASS L1', 'freq_mhz': 1602.00, 'bw_hz': 10e6},
    {'name': 'GLONASS L2', 'freq_mhz': 1248.00, 'bw_hz': 10e6},
    {'name': 'BeiDou B1', 'freq_mhz': 1561.00, 'bw_hz': 10e6},
]

# DTMF tone pairs for protocol disruption
DTMF_PAIRS = [
    (697, 1209), (697, 1336), (697, 1477),
    (770, 1209), (770, 1336), (770, 1477),
    (852, 1209), (852, 1336), (852, 1477),
]

# Jamming strategies available per frequency
JAM_MODES = {
    'noise': {'name': 'Static Noise', 'description': 'Broadband white noise'},
    'chirp': {'name': 'Chirp Sweep', 'description': 'Linear frequency sweep (spread spectrum)'},
    'tone': {'name': '400Hz Tone', 'description': 'Single tone at ±400Hz deviation'},
    'dtmf': {'name': 'DTMF Scan', 'description': 'Rotating DTMF tone pairs'},
    'gps_l1': {'name': 'GPS L1 Jam', 'description': 'GPS L1 band noise jamming'},
    'gps_l2': {'name': 'GPS L2 Jam', 'description': 'GPS L2 band noise jamming'},
    'gps_sweep': {'name': 'GPS Sweep', 'description': 'Sweep GPS L1/L5 bands'},
    'burst': {'name': 'Burst Interference', 'description': 'Rapid on/off bursts (40% duty)'},
    'sweep': {'name': 'Wideband Sweep', 'description': 'Continuous RF band sweep'},
}


class JammingProcess:
    """Manages a single jamming process."""
    
    def __init__(self, freq_mhz, mode='noise', signal_info=None):
        self.freq_mhz = freq_mhz
        self.mode = mode
        self.signal_info = signal_info or {}
        self.process = None
        self.started_at = time.time()
        self.output_file = None
        self.is_running = False
    
    def start(self):
        """Start the jamming process for this frequency/mode."""
        if self.is_running and self.process and self.process.poll() is None:
            return True
        
        try:
            # Generate noise file
            noise_file = f'/tmp/rflord_jam_{self.freq_mhz:.1f}_{self.mode}.bin'
            
            if not os.path.exists(noise_file):
                self._generate_noise_file(noise_file)
            
            freq_hz = int(self.freq_mhz * 1e6)
            
            # Select jamming strategy
            if self.mode == 'noise':
                cmd = [
                    "hackrf_transfer", "-t", noise_file,
                    "-f", str(freq_hz),
                    "-s", "2000000", "-a", "1", "-x", "40",
                    "-n", "120000000"  # 60 seconds of data
                ]
            elif self.mode == 'chirp':
                cmd = [
                    "hackrf_transfer", "-t", noise_file,
                    "-f", str(freq_hz),
                    "-s", "4000000", "-a", "1", "-x", "40",
                    "-n", "240000000"  # 60 seconds at 4MHz
                ]
            elif self.mode == 'tone':
                cmd = [
                    "hackrf_transfer", "-t", noise_file,
                    "-f", str(freq_hz),
                    "-s", "2000000", "-a", "1", "-x", "40",
                    "-n", "120000000"
                ]
            elif self.mode in ('gps_l1', 'gps_l2'):
                cmd = [
                    "hackrf_transfer", "-t", noise_file,
                    "-f", str(freq_hz),
                    "-s", "1000000", "-a", "1", "-x", "40",
                    "-n", "60000000"
                ]
            elif self.mode == 'burst':
                cmd = [
                    "hackrf_transfer", "-t", noise_file,
                    "-f", str(freq_hz),
                    "-s", "2000000", "-a", "1", "-x", "40",
                    "-n", "120000000"
                ]
            elif self.mode == 'sweep':
                # Sweep ±5MHz around target
                sweep_lo = int((self.freq_mhz - 5) * 1e6)
                sweep_hi = int((self.freq_mhz + 5) * 1e6)
                cmd = [
                    "hackrf_sweep", "-f", f"{sweep_lo}:{sweep_hi}",
                    "-w", "2000000", "-l", "32", "-g", "40",
                    "-a", "1", "-N", "1"
                ]
            else:
                cmd = [
                    "hackrf_transfer", "-t", noise_file,
                    "-f", str(freq_hz),
                    "-s", "2000000", "-a", "1", "-x", "40",
                    "-n", "120000000"
                ]
            
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            self.is_running = True
            log.info(f"INTERDICTION: Started {self.mode} jamming at {self.freq_mhz:.1f} MHz (PID {self.process.pid})")
            return True
            
        except Exception as e:
            log.error(f"INTERDICTION: Failed to start jamming at {self.freq_mhz:.1f} MHz ({self.mode}): {e}")
            self.is_running = False
            return False
    
    def stop(self):
        """Stop the jamming process."""
        if self.process and self.is_running:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    self.process.kill()
                except Exception:
                    pass
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.is_running = False
            log.info(f"INTERDICTION: Stopped jamming at {self.freq_mhz:.1f} MHz")
    
    def _generate_noise_file(self, noise_file):
        """Generate random IQ noise file for HackRF TX."""
        try:
            n = int(2000000 * 60)  # 60 seconds at 2 MSPS
            samples = np.random.randint(-127, 128, n, dtype=np.int8)
            samples.tofile(noise_file)
        except Exception as e:
            log.error(f"INTERDICTION: Failed to generate noise file: {e}")
    
    def status(self):
        """Return current status of this jamming process."""
        if not self.is_running:
            return 'stopped'
        
        if self.process and self.process.poll() is None:
            return 'active'
        
        self.is_running = False
        return 'dead'


class InterdictionEngine:
    """Main interdiction engine managing all jamming operations."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.active_jammers = {}  # freq_mhz -> JammingProcess
        self.strategy_map = {}    # freq_mhz -> mode string
        self.jamming_procs = []   # List of all JammingProcess instances
        self.noise_file_cache = '/tmp/rflord_noise.bin'
        self._ensure_noise_cache()
        
        # Interdiction settings from config
        self.enabled = self.config.get('interdiction', {}).get('enabled', False)
        self.auto_jam_threshold = self.config.get('interdiction', {}).get('auto_jam_threshold', 'danger')
        self.max_jammers = self.config.get('interdiction', {}).get('max_concurrent_jammers', 8)
        self.auto_restart_interval = self.config.get('interdiction', {}).get('auto_restart_interval', 55)  # seconds
        
        # Protocol-specific settings
        self.protocol_jamming = self.config.get('interdiction', {}).get('protocols', {})
        
        log.info(f"INTERDICTION: Engine initialized, max concurrent jammers={self.max_jammers}")
    
    def _ensure_noise_cache(self):
        """Pre-generate noise file if not exists."""
        if not os.path.exists(self.noise_file_cache):
            try:
                n = int(2000000 * 60)
                samples = np.random.randint(-127, 128, n, dtype=np.int8)
                samples.tofile(self.noise_file_cache)
            except Exception as e:
                log.error(f"INTERDICTION: Failed to generate cache noise file: {e}")
    
    def process_signal(self, freq_mhz, signal_info, classification='sus'):
        """Process a detected signal and decide on interdiction action.
        
        Args:
            freq_mhz: Frequency in MHz
            signal_info: Dict with signal details (peak, std, type, etc.)
            classification: 'ok', 'sus', or 'danger'
        
        Returns:
            JammingProcess if jamming started, else None
        """
        if not self.enabled:
            return None
        
        # Check threshold
        if self.auto_jam_threshold == 'danger' and classification != 'danger':
            return None
        
        # Check if already jamming this frequency
        freq_key = round(freq_mhz)
        if freq_key in self.active_jammers:
            # Update last seen
            proc = self.active_jammers[freq_key]
            proc.signal_info.update(signal_info)
            return proc
        
        # Check max jammers limit
        active_count = len([p for p in self.active_jammers.values() if p.is_running])
        if active_count >= self.max_jammers:
            # Evict oldest jammer (FIFO)
            oldest_key = min(self.active_jammers, key=lambda k: self.active_jammers[k].started_at)
            old_proc = self.active_jammers.pop(oldest_key)
            old_proc.stop()
            log.info(f"INTERDICTION: Evicted old jammer at {old_proc.freq_mhz:.1f} MHz")
        
        # Determine jamming mode based on frequency and protocol
        mode = self._select_mode(freq_mhz, signal_info, classification)
        if not mode:
            return None
        
        # Create jamming process
        proc = JammingProcess(freq_mhz, mode=mode, signal_info=signal_info)
        
        if proc.start():
            self.active_jammers[freq_key] = proc
            self.strategy_map[freq_key] = mode
            self.jamming_procs.append(proc)
            return proc
        
        return None
    
    def _select_mode(self, freq_mhz, signal_info, classification):
        """Select the best jamming mode for this signal.
        
        Uses frequency analysis and protocol detection to choose optimal mode.
        """
        # GPS-specific modes
        if 1570 <= freq_mhz <= 1580:
            return 'gps_l1'
        if 1220 <= freq_mhz <= 1235:
            return 'gps_l2'
        
        # ExpressLRS/CRSF on 900MHz band
        if 910 <= freq_mhz <= 920:
            return 'chirp'  # Chirp works well for CRSF
        
        # FPV bands (2.4GHz) — use burst to disrupt packet protocol
        if 2400 <= freq_mhz <= 2500:
            return 'burst'
        
        # SmartProtocol band
        if 2410 <= freq_mhz <= 2470:
            return 'chirp'
        
        # TBS Crossfire bands
        if 430 <= freq_mhz <= 435:
            return 'tone'
        if 865 <= freq_mhz <= 870:
            return 'tone'
        
        # Default: noise jamming for unknown threats
        if classification == 'danger':
            return 'sweep'
        
        return 'noise'
    
    def set_mode(self, freq_mhz, mode):
        """Change the jamming mode for an active frequency.
        
        Args:
            freq_mhz: Frequency to change mode for
            mode: One of JAM_MODES keys
        
        Returns:
            JammingProcess if successful, else None
        """
        freq_key = round(freq_mhz)
        if freq_key not in self.active_jammers:
            log.warning(f"INTERDICTION: No jamming at {freq_mhz:.1f} MHz")
            return None
        
        proc = self.active_jammers[freq_key]
        
        # Stop current, restart with new mode
        proc.stop()
        time.sleep(0.5)
        
        proc.mode = mode
        self.strategy_map[freq_key] = mode
        
        if proc.start():
            return proc
        return None
    
    def stop_freq(self, freq_mhz):
        """Stop jamming a specific frequency."""
        freq_key = round(freq_mhz)
        if freq_key in self.active_jammers:
            proc = self.active_jammers.pop(freq_key)
            self.strategy_map.pop(freq_key, None)
            proc.stop()
    
    def stop_all(self):
        """Stop all jamming processes."""
        for key in list(self.active_jammers.keys()):
            proc = self.active_jammers[key]
            proc.stop()
        self.active_jammers.clear()
        self.strategy_map.clear()
        log.info("INTERDICTION: All jamming stopped")
    
    def process_scan(self, signals, known_freqs, classify_fn):
        """Process scan results and manage interdiction.
        
        Args:
            signals: List of signal dicts {'freq', 'peak', 'std'}
            known_freqs: Dict of freq_key -> first_seen_time
            classify_fn: Function to classify signals
        
        Returns:
            List of newly jammed frequencies
        """
        newly_jammed = []
        
        for s in signals:
            f = s['freq'] / 1e6
            freq_key = round(f)
            cls = classify_fn(f, s['peak'], s['std'])
            
            # Process signal through interdiction engine
            proc = self.process_signal(f, s, cls)
            if proc:
                newly_jammed.append(f)
        
        # Check for stopped jammers and restart if signal is back
        for freq_key, proc in list(self.active_jammers.items()):
            status = proc.status()
            if status == 'dead' or not proc.is_running:
                # Signal may have returned — try to restart
                if self.auto_restart_interval > 0:
                    elapsed = time.time() - proc.started_at
                    if elapsed >= self.auto_restart_interval:
                        proc.start()
        
        return newly_jammed
    
    def get_status(self):
        """Get current interdiction status for display."""
        status = {
            'active': len([p for p in self.active_jammers.values() if p.is_running]),
            'jammers': [],
        }
        
        for freq_key, proc in sorted(self.active_jammers.items()):
            status['jammers'].append({
                'freq_mhz': proc.freq_mhz,
                'mode': proc.mode,
                'status': proc.status(),
                'protocol': proc.signal_info.get('type', 'unknown'),
                'peak': proc.signal_info.get('peak', 0),
            })
        
        return status
    
    def get_jammer_count(self):
        """Return count of active jammers."""
        return len([p for p in self.active_jammers.values() if p.is_running])


class ProtocolJammer:
    """Protocol-specific jamming for FPV drone control links."""
    
    def __init__(self):
        self.jammers = {}  # freq_mhz -> subprocess.Popen
    
    def jamm_expresslrs(self, freq_mhz=2408.0):
        """Jam ExpressLRS/CRSF by flooding with packets.
        
        CRSF uses ~1ms packet intervals — jamming during ACK windows breaks link.
        """
        freq_hz = int(freq_mhz * 1e6)
        cmd = [
            "hackrf_transfer", "-t", "/tmp/rflord_noise.bin",
            "-f", str(freq_hz),
            "-s", "2000000", "-a", "1", "-x", "40",
            "-n", "120000000"
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.jammers[freq_hz] = proc
            log.info(f"PROTOCOL JAMMER: ExpressLRS at {freq_mhz:.1f} MHz (PID {proc.pid})")
            return True
        except Exception as e:
            log.error(f"PROTOCOL JAMMER: Failed CRSF jamming: {e}")
            return False
    
    def jamm_smartprotocol(self, freq_mhz=2462.0):
        """Jam BetaFPV SmartProtocol."""
        freq_hz = int(freq_mhz * 1e6)
        cmd = [
            "hackrf_transfer", "-t", "/tmp/rflord_noise.bin",
            "-f", str(freq_hz),
            "-s", "2000000", "-a", "1", "-x", "40",
            "-n", "120000000"
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.jammers[freq_hz] = proc
            log.info(f"PROTOCOL JAMMER: SmartProtocol at {freq_mhz:.1f} MHz (PID {proc.pid})")
            return True
        except Exception as e:
            log.error(f"PROTOCOL JAMMER: Failed SmartProtocol jamming: {e}")
            return False
    
    def jamm_crossfire(self, freq_mhz=915.0):
        """Jam TBS Crossfire."""
        freq_hz = int(freq_mhz * 1e6)
        cmd = [
            "hackrf_transfer", "-t", "/tmp/rflord_noise.bin",
            "-f", str(freq_hz),
            "-s", "2000000", "-a", "1", "-x", "40",
            "-n", "120000000"
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.jammers[freq_hz] = proc
            log.info(f"PROTOCOL JAMMER: Crossfire at {freq_mhz:.1f} MHz (PID {proc.pid})")
            return True
        except Exception as e:
            log.error(f"PROTOCOL JAMMER: Failed Crossfire jamming: {e}")
            return False
    
    def stop_all(self):
        """Stop all protocol jammers."""
        for freq, proc in self.jammers.items():
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except:
                try:
                    proc.kill()
                except:
                    pass
        self.jammers.clear()


class GPSJammer:
    """GPS L1/L2/GPS L5 jamming and spoofing."""
    
    def __init__(self):
        self.active_bands = {}  # gps_band_name -> process
    
    def jam_l1(self, intensity='medium'):
        """Jam GPS L1 band (1575.42 MHz)."""
        freq_hz = int(1575.42 * 1e6)
        noise_file = self._generate_gps_noise(intensity)
        
        cmd = [
            "hackrf_transfer", "-t", noise_file,
            "-f", str(freq_hz),
            "-s", "2000000", "-a", "1", "-x", "40",
            "-n", "60000000"
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.active_bands['gps_l1'] = {'proc': proc, 'started': time.time()}
            log.info(f"GPS JAMMER: L1 band active (PID {proc.pid})")
            return True
        except Exception as e:
            log.error(f"GPS JAMMER: Failed L1 jamming: {e}")
            return False
    
    def jam_l2(self, intensity='medium'):
        """Jam GPS L2 band (1227.60 MHz)."""
        freq_hz = int(1227.60 * 1e6)
        noise_file = self._generate_gps_noise(intensity)
        
        cmd = [
            "hackrf_transfer", "-t", noise_file,
            "-f", str(freq_hz),
            "-s", "2000000", "-a", "1", "-x", "40",
            "-n", "60000000"
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.active_bands['gps_l2'] = {'proc': proc, 'started': time.time()}
            log.info(f"GPS JAMMER: L2 band active (PID {proc.pid})")
            return True
        except Exception as e:
            log.error(f"GPS JAMMER: Failed L2 jamming: {e}")
            return False
    
    def sweep_l1_l5(self):
        """Sweep GPS L1 and L5 bands for disruption."""
        freq_hz = int(1575.42 * 1e6)  # Start at L1
        noise_file = self._generate_gps_noise('high')
        
        cmd = [
            "hackrf_sweep", "-f", f"{1170000000}:1580000000",
            "-w", "2000000", "-l", "32", "-g", "40",
            "-a", "1"
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.active_bands['gps_sweep'] = {'proc': proc, 'started': time.time()}
            log.info(f"GPS JAMMER: L1/L5 sweep active (PID {proc.pid})")
            return True
        except Exception as e:
            log.error(f"GPS JAMMER: Failed GPS sweep: {e}")
            return False
    
    def _generate_gps_noise(self, intensity='medium'):
        """Generate GPS-band noise file with specified intensity."""
        noise_file = f'/tmp/rflord_gps_noise_{intensity}.bin'
        
        if not os.path.exists(noise_file) or intensity == 'high':
            try:
                n = int(2000000 * 30)  # 30 seconds
                samples = np.random.randint(-127, 128, n, dtype=np.int8)
                samples.tofile(noise_file)
            except Exception as e:
                log.error(f"GPS JAMMER: Failed to generate noise: {e}")
        
        return noise_file
    
    def stop_all(self):
        """Stop all GPS jammers."""
        for band, info in self.active_bands.items():
            try:
                info['proc'].terminate()
                info['proc'].wait(timeout=2)
            except:
                try:
                    info['proc'].kill()
                except:
                    pass
        self.active_bands.clear()


def generate_chirp_noise_file(output_file, duration_s=30, start_freq_hz=2400e6, end_freq_hz=2500e6):
    """Generate chirp/spread spectrum noise file for HackRF TX.
    
    Creates a linear FM chirp that sweeps between start and end frequency.
    Useful for disrupting packet-based protocols (ExpressLRS, etc.).
    """
    try:
        sample_rate = 2000000
        num_samples = int(sample_rate * duration_s)
        
        # Generate time array
        t = np.arange(num_samples) / sample_rate
        
        # Linear chirp phase
        f_start = start_freq_hz
        f_end = end_freq_hz
        sweep_duration = duration_s  # one full sweep per duration
        
        # Instantaneous frequency sweeps linearly
        freq = f_start + (f_end - f_start) * (t % sweep_duration) / sweep_duration
        
        # Generate phase by integrating frequency
        phase = 2 * np.pi * np.cumsum(freq) / sample_rate
        
        # Convert to IQ samples (float32 -> int8)
        iq_float = np.cos(phase).astype(np.float32) + 1j * np.sin(phase).astype(np.float32)
        
        # Scale to int8 range [-127, 127]
        iq_int8 = (iq_float.real * 127).astype(np.int8)
        
        iq_int8.tofile(output_file)
        return True
        
    except Exception as e:
        log.error(f"generate_chirp_noise_file failed: {e}")
        return False


def generate_dtmf_noise_file(output_file, duration_s=30):
    """Generate DTMF tone sequence noise file.
    
    Rotates through all DTMF pairs for protocol disruption.
    """
    try:
        sample_rate = 2000000
        num_samples = int(sample_rate * duration_s)
        
        dtmf_freqs = [697, 770, 852, 941, 1209, 1336, 1477]
        tone_duration = 0.2  # seconds per tone pair
        silence_duration = 0.1
        
        samples = []
        current = 0
        
        for i in range(0, len(dtmf_freqs) - 1):
            freq_low = dtmf_freqs[i]
            freq_high = dtmf_freqs[i + 1]
            
            # Generate tone pair sequence
            num_cycles = int(tone_duration / (tone_duration + silence_duration))
            for _ in range(num_cycles):
                # Low frequency tone
                n_tone = int(sample_rate * tone_duration / 2)
                if current + n_tone <= num_samples:
                    t = np.arange(n_tone) / sample_rate
                    tone = np.sin(2 * np.pi * freq_low * t).astype(np.float32)
                    samples.append(tone)
                    
                    # High frequency tone
                    if current + 2 * n_tone <= num_samples:
                        tone = np.sin(2 * np.pi * freq_high * t[:n_tone]).astype(np.float32)
                        samples.append(tone)
        
        # Combine and scale
        combined = np.concatenate(samples).astype(np.float32)
        combined = combined / (np.max(np.abs(combined)) + 1e-10) * 127
        
        output = combined.astype(np.int8)
        output.tofile(output_file)
        return True
        
    except Exception as e:
        log.error(f"generate_dtmf_noise_file failed: {e}")
        return False
