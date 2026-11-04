# Judananas.jl 🍍🧃 (pilot)

**JU**lia **D**ata **AN**alysis for **A**ction potentials, **N**eural **A**ctivity and **S**pikes.

Also *jus d'ananas*: pineapple juice. A small Julia core for spike trains and other
neurophysiological time series, squeezed from
[pynapple](https://github.com/pynapple-org/pynapple) (PYthon Neural Analysis Package).
Pilot stage.

The core is kept small on purpose: data objects plus a few basic operations.
Bigger analyses (tuning curves, CCGs, decoding...) will live in separate packages
built on top of it.

## Objects

| Judananas      | pynapple equivalent                | What it holds |
|-------------|------------------------------------|---------------|
| `IntervalSet` | `IntervalSet`                    | Sorted, non-overlapping epochs `[start, stop]` with optional metadata |
| `Ts`          | `Ts`                             | Event timestamps |
| `Tsd`         | `Tsd`, `TsdFrame`, `TsdTensor`   | Timestamped data; vector, matrix (named columns) or N-d array, time on dim 1 |
| `TsGroup`     | `TsGroup`                        | A population of `Ts` with per-unit metadata (`:rate` computed automatically) |

Each object carries a **time support**, the `IntervalSet` on which it is defined.
All times are `Float64` seconds.

## Loading NWB files

```julia
using Judananas, HDF5          # HDF5 switches on the NWB reader (a package extension)
data = load_file("session.nwb")
units  = data["units"]         # TsGroup
trials = data["trials"]        # IntervalSet
lfp    = data["LFP/ElectricalSeries"]
```

`load_file` works like `nap.load_file`. Each object is read on first access and then cached.
- The `Units` table becomes a `TsGroup`. Every column becomes metadata, even columns missing
  from `colnames` (which pynwb ignores). Each unit also gets its electrode's columns
  (`location`, coordinates...) and an `electrode` field holding the 0-based NWB row.
- `TimeIntervals` tables become `IntervalSet`s with metadata.
- Every numeric `TimeSeries` becomes a `Tsd` with time on the first dimension. Uncompressed data is
  memory-mapped, so a 10 GB LFP is only read where you index or `restrict` it.
- Data is returned as stored, without applying `conversion`.
- Per-unit waveforms (`waveform_mean(s)`, `waveform_sd(s)`) are never read, and table columns over
  50 MB are skipped with a message: `load_file(path; skip_columns = ["..."], max_column_bytes = Inf)`.
- `dandi_url(dandiset, asset_path; version = "draft")` gives the S3 URL of a DANDI file.
  `load_file(url)` streams it with HDF5's ROS3 driver (experimental: it needs `HDF5.has_ros3()`,
  e.g. HDF5_jll 2.2, and hung on macOS in the only test so far).

## Operations

- Interval algebra: `intersect`, `union`, `setdiff`, `merge_close_intervals`,
  `drop_short_intervals`, `drop_long_intervals`, `t in ep`
- `restrict(x, ep)` or `restrict(x, start, stop)`: the support becomes the intersection
- `count(x, bin_size)`: binned counts (a matrix `Tsd` for a `TsGroup`), or `count(x; ep)` per epoch
- `bin_average(tsd, bin_size)`
- `value_from(ts, tsd)`: the closest sample of `tsd` at each event, e.g. position at each spike
- `threshold(tsd, thr; method = :above)`: the support becomes the passing epochs
- `compute_perievent(ts_or_group, events, (before, after))`: events are a vector (order kept),
  a `Ts`, or an `IntervalSet` of trials (aligned on starts, trial metadata copied)
- `rate(ts)`, `time_diff(ts)` (ISIs as a `Tsd`), `compute_isi_distribution(ts_or_group; bins, log_scale)`
- `filter(u -> u.location == "PFC" && u.rate > 1, group)`
- by unit id: `group[id = 185]`, `group[id = [185, 228]]`, `group[:location, id = 185]`,
  `metadata(group, 185)`, `indexof(group, 185)`
- `value_counts(group, :location)`, `metadata_table(group)` (a Tables.jl table: `DataFrame(metadata_table(group))`)
- `to_tsd(group)`: all spikes in one `Tsd`, with unit ids as values

## Plotting

Plot recipes for Plots.jl are defined through RecipesBase, so Judananas does not depend on Plots itself:

```julia
using Judananas, Plots
plot(group)              # raster
plot!(epochs)            # shaded epochs
plot(tsd)                # traces, broken across epochs
plot(compute_perievent(group[1], onsets, (-0.5, 1)))   # trial raster, rows labelled "Trial"
plot(pe; color = colors_per_trial, yflip = true)       # coloured ticks, first trial on top
```

## Design decisions

- Epochs are closed intervals `[start, stop]`. Overlapping epochs are merged on construction,
  while touching epochs (back-to-back trials) are kept separate.
- Interval algebra (`intersect`, `union`, `setdiff`) drops metadata.
- In `count` and `bin_average`, bins start at each epoch's start and an incomplete last bin is dropped.
  Bins are `[left, right)`, except the last bin of an epoch, which includes the epoch's right edge.
- `TsGroup` indexing is positional (`g[3]` is the third unit), and `ids(g)` gives the unit ids;
  `g[id = 185]` selects by id.

## Status

Tested on Julia 1.13 (`julia --project -e 'using Pkg; Pkg.test()'`).
`examples/quickstart.jl` runs on synthetic data and saves a 4-panel figure.

Next step: automated parity tests against pynapple on a real session (PythonCall.jl).
