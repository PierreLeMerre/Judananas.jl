"""
    Judananas

**JU**lia **D**ata **AN**alysis for **A**ction potentials, **N**eural **A**ctivity and **S**pikes
(or *jus d'ananas*, pineapple juice).

A small core for neurophysiological time series in Julia, inspired by pynapple.

Five objects:

- `IntervalSet` — a set of non-overlapping epochs `[start, stop]` (trials, states...)
- `Ts`          — event timestamps (one spike train, stimulus onsets...)
- `Tsd`         — timestamped data; the data can be a vector, a matrix
                  (one column per channel/unit) or any N-d array (time on dim 1)
- `TsGroup`     — a group of `Ts` with per-unit metadata (a population)

Every time series carries a `time_support`: the `IntervalSet` on which it is defined.
Operations such as `restrict`, `count` or `bin_average` respect and propagate it.

All times are `Float64` seconds.
"""
module Judananas

using Statistics: mean
using RecipesBase
using Downloads

export IntervalSet, Ts, Tsd, TsGroup
export starts, stops, tot_length, merge_close_intervals,
       drop_short_intervals, drop_long_intervals
export times, time_support, restrict, columns, ids, metadata, rates, rate, indexof
export metadata_table, value_counts
export bin_average, value_from, threshold, compute_perievent, to_tsd,
       time_diff, compute_isi_distribution
export load_file, NWBFile, dandi_url

const Metadata = Dict{Symbol,Vector}

include("intervalset.jl")
include("timeseries.jl")
include("tsgroup.jl")
include("operations.jl")
include("plotting.jl")
include("nwb.jl")

end # module
