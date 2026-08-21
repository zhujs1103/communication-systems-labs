# Communication Systems Labs

Reproducible Python/Jupyter experiments covering digital and analog modulation.

## Modules

- `digital-communication-system`: end-to-end digital communication system notebook.
- `fsk-iq-modulation`: 2FSK and IQ/16QAM simulation with BER analysis.
- `analog-modulation`: DSB-SC, conventional AM, SSB and FM modulation/demodulation under high and low SNR.

The analog module can use a WAV baseband input but falls back to a synthetic signal, so no personal recording is required or included.

## Verification status

- All three notebooks parse as valid Jupyter JSON.
- All 48 code cells have execution counts and stored outputs contain no Jupyter error records.
- The analog simulation script passes Python syntax parsing.
- Existing notebook outputs are retained for review, but the full experiments were not re-executed during portfolio preparation.

Typical dependencies:

```bash
pip install numpy scipy matplotlib pandas tqdm jupyter
```
