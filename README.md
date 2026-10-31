# Exit the Moat

Code, benchmarks and results behind the YouTube channel
[Exit the Moat](https://www.youtube.com/@ExitTheMoat): can you climb out of the CUDA moat, what does it cost, and is
the other side actually better?

Every number in a video comes from the code and the `results/` data in this repo.

- YouTube: [@ExitTheMoat](https://www.youtube.com/@ExitTheMoat)
- Hugging Face: [exit-the-moat](https://huggingface.co/exit-the-moat) (converted weights)
- Patreon: [Exit the Moat](https://patreon.com/Exitthemoat) (support the channel)

## Episode 0 in numbers: Fish Audio Drama 3 (36B voice model) on one Tenstorrent p150a

One card, PCIe Gen3 x16, 8 test lines, 2026-10-31. RTF = compute seconds per second of audio (lower is better).
Details, method and run files: [`fish-drama3-p150/`](fish-drama3-p150).

| | p150a, this port | original model, rented RTX PRO 6000 |
|---|---|---|
| real-time factor, one voice | **0.381** (0.404 over the whole 47-line narration) | 0.173 |
| time to first audio | **0.41-0.47 s** | 0.16-0.20 s |
| many voices at once (audio s per wall s) | 9.51 (16 slots, narration chunks) | 17.7 (8 concurrent requests) |
| board power while speaking, one voice | mean 196 W | mean 263-270 W |
| price in Germany, incl. 19 % VAT | €1,613 (paid €1,355.67 + import VAT) | ~€21,000 (idealo.de, 31 Oct 2026; under €15,000 on 1 Sep 2026) |

Over the port: 14.7 RTF at the first end-to-end run, 0.853 at the end of "day 2", 0.381 after four more days of
optimization.

## Episodes

| # | Topic | Folder |
|---|---|---|
| 0 | I almost returned two Tenstorrent AI cards. It was the cable. (Fish Audio Drama 3 on one p150) | [`fish-drama3-p150/`](fish-drama3-p150) · weights: [Hugging Face](https://huggingface.co/exit-the-moat/drama-3-exit-the-moat-tenstorrent-p150) · video: [YouTube](https://youtu.be/mpleglhA_Dg) |

New episodes are added here when they are published.

## License

Apache-2.0, see [LICENSE](LICENSE). Model weights follow their own licences (Drama 3: Fish Audio Community License,
see [`fish-drama3-p150/`](fish-drama3-p150)).
