# ---------------------------------------------------------------------------
# Correlograms: auto-, cross- and event-correlograms, jitter
# ---------------------------------------------------------------------------

"""
    spike_time_differences(ref, target, max_lag) -> Vector{Float64}

All the differences `t_target - t_ref` within `±max_lag` (bounds included), sorted.
`ref` and `target` are `Ts` or sorted vectors of times. Each pair of spikes gives one
difference, so passing the same train twice includes the zeros of each spike with itself.

This is the definition of a cross-correlogram: its histogram is the CCG in counts.
"""
function spike_time_differences(ref, target, max_lag::Real)
    r, t = _times(ref), _times(target)
    d = Float64[]
    for x in r
        a = searchsortedfirst(t, x - max_lag)
        b = searchsortedlast(t, x + max_lag)
        for j in a:b
            dj = t[j] - x
            abs(dj) <= max_lag && push!(d, dj)
        end
    end
    return sort!(d)
end

_times(x::Ts) = x.t
_times(x::AbstractVector{<:Real}) = issorted(x) ? x : sort(x)

"""
    correlogram_edges(binsize, windowsize)

Bin edges of a correlogram, as in pynapple: an odd number of bins of width `binsize`,
centered on multiples of `binsize` (one bin is centered on 0), covering `±windowsize`.
With `binsize = 1 ms` and `windowsize = 20 ms`: 41 bins, edges from -20.5 to +20.5 ms.
"""
function correlogram_edges(binsize::Real, windowsize::Real)
    binsize > 0 && windowsize > 0 || throw(ArgumentError("binsize and windowsize must be positive"))
    n = floor(Int, windowsize / binsize + 1e-9)     # bins on each side of 0
    return ((-n:n+1) .- 0.5) .* binsize
end

function _edges_and_lags(binsize, windowsize, edges)
    if edges !== nothing
        e = collect(Float64, edges)
        length(e) >= 2 && issorted(e) && allunique(e) ||
            throw(ArgumentError("edges must be increasing, with at least 2 edges"))
        return e, (e[1:end-1] .+ e[2:end]) ./ 2
    end
    (binsize === nothing || windowsize === nothing) &&
        throw(ArgumentError("give binsize and windowsize, or edges"))
    e = collect(Float64, correlogram_edges(binsize, windowsize))
    n = (length(e) - 2) ÷ 2
    return e, collect((-n:n) .* Float64(binsize))   # exact multiples of binsize
end

# Add to `c` the number of pairs (i in ri, j in rj) with d = (target[j] - toff) - (ref[i] - roff)
# in each bin [edges[k], edges[k+1]). With `self`, the pair i == j is skipped.
function _add_counts!(c, ref, ri, target, rj, edges, roff, toff, self)
    lo, hi = edges[1], edges[end]
    for i in ri
        r = ref[i] - roff
        a = max(searchsortedfirst(target, r + toff + lo - 1e-9), first(rj))
        b = min(searchsortedlast(target, r + toff + hi + 1e-9), last(rj))
        for j in a:b
            self && i == j && continue
            d = roff == toff ? target[j] - ref[i] : (target[j] - toff) - r
            k = searchsortedlast(edges, d)
            1 <= k < length(edges) && (c[k] += 1)
        end
    end
    return c
end

_span(t, s, e) = searchsortedfirst(t, s):searchsortedlast(t, e)

# Counts of target relative to ref, the number of reference spikes and the target rate.
function _ccg_core(ref::Ts, target::Ts, edges; ep, same_epoch, epoch_shift, self)
    c = zeros(Int, length(edges) - 1)
    if ep === nothing && !same_epoch
        _add_counts!(c, ref.t, eachindex(ref.t), target.t, eachindex(target.t), edges, 0.0, 0.0, self)
        return c, length(ref), rate(target)
    end
    ep = ep === nothing ? target.time_support : ep
    if !same_epoch
        r, t = restrict(ref, ep), restrict(target, ep)
        _add_counts!(c, r.t, eachindex(r.t), t.t, eachindex(t.t), edges, 0.0, 0.0, self)
        return c, length(r), rate(t)
    end
    # only pairs of spikes from the same epoch (or epoch k of ref with k + epoch_shift of target)
    n = length(ep)
    nref = ntarget = 0
    for k in 1:n
        k2 = mod1(k + epoch_shift, n)
        ri = _span(ref.t, ep.start[k], ep.stop[k])
        rj = _span(target.t, ep.start[k2], ep.stop[k2])
        nref += length(ri)
        ntarget += length(_span(target.t, ep.start[k], ep.stop[k]))
        (isempty(ri) || isempty(rj)) && continue
        roff, toff = k == k2 ? (0.0, 0.0) : (ep.start[k], ep.start[k2])
        _add_counts!(c, ref.t, ri, target.t, rj, edges, roff, toff, self && k == k2)
    end
    T = tot_length(ep)
    return c, nref, T > 0 ? ntarget / T : NaN
end

# counts -> spikes/s of the target after a reference spike, or that rate / target rate
function _scale(c, nref, target_rate, edges; norm, counts)
    counts && return c
    v = c ./ (nref .* diff(edges))
    return norm ? v ./ target_rate : v
end

_check_shift(same_epoch, epoch_shift) = epoch_shift == 0 || same_epoch ||
    throw(ArgumentError("epoch_shift needs same_epoch = true"))

const _CCG_KW_DOC = """
Keywords (shared by all correlogram functions):

- `binsize`, `windowsize` (s): an odd number of bins of width `binsize`, centered on multiples
  of `binsize` (one bin centered on 0) and covering `±windowsize`, as in pynapple.
  Or `edges`: any increasing vector of bin edges (s), which overrides them.
- `norm = true`: divide by the mean rate of the target, so 1 means "as expected by chance".
  `norm = false`: rate of the target in spikes/s after a reference spike,
  `counts / (n_ref × bin width)`.
- `counts = true`: raw numbers of coincidences (integers), ignoring `norm`; what jitter
  correction, significance tests and transmission probabilities need.
- `ep`: only spikes (and events) inside `ep` are used, and the target rate is computed on `ep`.
  Pairs of spikes from different epochs of `ep` are counted, as in pynapple, unless
  `same_epoch = true`. Without `ep`, all spikes are used and the target rate is computed
  on its own time support.
- `same_epoch = true`: only pairs of spikes inside the same epoch of `ep` (e.g. the same trial;
  default `ep`: the time support of the target).
- `epoch_shift = s` (with `same_epoch`): pair epoch `k` of the reference with epoch `k + s`
  of the target (circularly: the last epochs pair with the first ones), with times taken
  relative to each epoch start. With trials, this is the **shift predictor**.

Returns a NamedTuple with `lags` (bin centers, s), `edges` and `values`.

Bins are `[left, right)`. Spike times are often multiples of the sampling period
(1/30000 s), so many differences fall exactly on a bin edge, and floating-point rounding
decides their bin. This explains small count differences between tools (pynapple compares
`t_target` with `t_ref + edge`, Judananas compares `t_target - t_ref` with the edge).
"""

"""
    compute_crosscorrelogram(ref::Ts, target::Ts; binsize, windowsize, norm = true, ...)
    compute_crosscorrelogram(g::TsGroup; ...)
    compute_crosscorrelogram((refs, targets)::Tuple{TsGroup,TsGroup}; ...)
    compute_crosscorrelogram(refs::TsGroup, targets::TsGroup; ...)

Cross-correlogram (CCG) of `target` relative to `ref`: the rate of `target` at each lag
after a spike of `ref`. A peak at +2 ms means `target` tends to fire 2 ms after `ref`.

- Two `Ts`: `values` is a vector.
- One `TsGroup`: every pair `(g[i], g[j])` with `i < j` **in the order of the group**
  (not sorted by id), `g[j]` relative to `g[i]`. So `g[id = [339, 98]]` gives the pair
  `(339, 98)`: 98 relative to 339. (pynapple sorts the units by id, so the same call
  would give 339 relative to 98.)
- Two `TsGroup`s `(refs, targets)`: every pair `(i, j)` with `i` in `refs` and `j` in
  `targets`, `j` relative to `i`, the reference in the outer loop.

For groups, `values` is a matrix lags × pairs and `pairs` lists the `(ref_id, target_id)`
of each column.

$_CCG_KW_DOC
"""
function compute_crosscorrelogram(ref::Ts, target::Ts; binsize = nothing, windowsize = nothing,
                                  edges = nothing, norm::Bool = true, counts::Bool = false,
                                  ep::Union{Nothing,IntervalSet} = nothing,
                                  same_epoch::Bool = false, epoch_shift::Integer = 0)
    _check_shift(same_epoch, epoch_shift)
    e, lags = _edges_and_lags(binsize, windowsize, edges)
    c, nref, tr = _ccg_core(ref, target, e; ep, same_epoch, epoch_shift, self = false)
    return (lags = lags, edges = e, values = _scale(c, nref, tr, e; norm, counts))
end

function compute_crosscorrelogram(g::TsGroup; kwargs...)
    n = length(g)
    return _ccg_pairs(g, g, [(i, j) for i in 1:n for j in i+1:n]; kwargs...)
end

compute_crosscorrelogram(gs::Tuple{TsGroup,TsGroup}; kwargs...) =
    compute_crosscorrelogram(gs[1], gs[2]; kwargs...)

compute_crosscorrelogram(refs::TsGroup, targets::TsGroup; kwargs...) =
    _ccg_pairs(refs, targets, [(i, j) for i in 1:length(refs) for j in 1:length(targets)]; kwargs...)

function _ccg_pairs(refs, targets, idx; binsize = nothing, windowsize = nothing, edges = nothing,
                    kwargs...)
    e, lags = _edges_and_lags(binsize, windowsize, edges)
    cols = [compute_crosscorrelogram(refs[i], targets[j]; edges = e, kwargs...).values
            for (i, j) in idx]
    v = reduce(hcat, cols; init = zeros(_eltype(; kwargs...), length(lags), 0))
    return (lags = lags, edges = e, values = v,
            pairs = [(refs.ids[i], targets.ids[j]) for (i, j) in idx])
end

_eltype(; counts::Bool = false, kwargs...) = counts ? Int : Float64

"""
    compute_autocorrelogram(x::Ts; binsize, windowsize, norm = true, ...)
    compute_autocorrelogram(g::TsGroup; ...)

Autocorrelogram (ACG): the rate of a unit at each lag after one of its own spikes. Each spike
paired with itself is not counted, but other spikes in the central bin are (pynapple sets
the whole central bin to 0). For a `TsGroup`, `values` is a matrix lags × units and `ids`
gives the unit of each column.

$_CCG_KW_DOC
"""
function compute_autocorrelogram(x::Ts; binsize = nothing, windowsize = nothing, edges = nothing,
                                 norm::Bool = true, counts::Bool = false,
                                 ep::Union{Nothing,IntervalSet} = nothing,
                                 same_epoch::Bool = false, epoch_shift::Integer = 0)
    _check_shift(same_epoch, epoch_shift)
    e, lags = _edges_and_lags(binsize, windowsize, edges)
    c, n, r = _ccg_core(x, x, e; ep, same_epoch, epoch_shift, self = true)
    return (lags = lags, edges = e, values = _scale(c, n, r, e; norm, counts))
end

function compute_autocorrelogram(g::TsGroup; binsize = nothing, windowsize = nothing,
                                 edges = nothing, kwargs...)
    e, lags = _edges_and_lags(binsize, windowsize, edges)
    cols = [compute_autocorrelogram(u; edges = e, kwargs...).values for u in g.units]
    v = reduce(hcat, cols; init = zeros(_eltype(; kwargs...), length(lags), 0))
    return (lags = lags, edges = e, values = v, ids = copy(g.ids))
end

"""
    compute_eventcorrelogram(x::Ts, events; binsize, windowsize, norm = true, ...)
    compute_eventcorrelogram(g::TsGroup, events; ...)

Event correlogram (a peri-event time histogram, PETH): the rate of each unit at each lag
after an event. `events` is a `Ts` or a vector of times. With `norm = true`, the rate is
divided by the mean rate of the unit. For a `TsGroup`, `values` is a matrix lags × units and
`ids` gives the unit of each column. With `ep`, only the events inside `ep` are used.

$_CCG_KW_DOC
"""
compute_eventcorrelogram(x::Ts, events; kwargs...) =
    compute_crosscorrelogram(_as_ts(events), x; kwargs...)

function compute_eventcorrelogram(g::TsGroup, events; binsize = nothing, windowsize = nothing,
                                  edges = nothing, kwargs...)
    e, lags = _edges_and_lags(binsize, windowsize, edges)
    ev = _as_ts(events)
    cols = [compute_crosscorrelogram(ev, u; edges = e, kwargs...).values for u in g.units]
    v = reduce(hcat, cols; init = zeros(_eltype(; kwargs...), length(lags), 0))
    return (lags = lags, edges = e, values = v, ids = copy(g.ids))
end

# events keep all their times: the support is set by `ep` (default: the units' support)
_as_ts(ev::Ts) = Ts(ev.t, IntervalSet(-Inf, Inf), Val(:unchecked))
_as_ts(ev::AbstractVector{<:Real}) = _as_ts(Ts(ev))

# --- jitter ------------------------------------------------------------------

"""
    jitter(x::Ts, width; rng = Random.default_rng())
    jitter(g::TsGroup, width; rng = Random.default_rng())

Move each spike by an independent uniform random time in `±width` (s), and sort.
The time support is kept; spikes jittered outside it are dropped. For a `TsGroup`,
ids and metadata are kept and the rates recomputed.

Jittering keeps everything slower than `width` (rate changes, responses to a stimulus)
and destroys the timing at finer scales: the CCG of jittered spikes is the CCG
expected without fast interactions.
"""
function jitter(x::Ts, width::Real; rng::AbstractRNG = Random.default_rng())
    t = x.t .+ width .* (2 .* rand(rng, length(x.t)) .- 1)
    return Ts(t; time_support = x.time_support)
end

function jitter(g::TsGroup, width::Real; rng::AbstractRNG = Random.default_rng())
    units = Ts[jitter(u, width; rng) for u in g.units]
    md = copy(g.metadata)
    md[:rate] = _rates(units, g.time_support)
    return TsGroup(units, copy(g.ids), md, g.time_support, Val(:unchecked))
end

"""
    jittered_crosscorrelogram(ref::Ts, target::Ts; jitter = 0.025, n_rep = 40,
                              rng = Random.default_rng(), binsize, windowsize, ...)

Mean cross-correlogram over `n_rep` repetitions, each with the spikes of `target` (only)
jittered by a uniform random time in `±jitter` (s). Takes the same keywords as
`compute_crosscorrelogram`; with `counts = true`, `values` is the mean number of
coincidences (not an integer). The jitter-corrected CCG is the real CCG minus this one.
"""
function jittered_crosscorrelogram(ref::Ts, target::Ts; jitter::Real = 0.025, n_rep::Integer = 40,
                                   rng::AbstractRNG = Random.default_rng(),
                                   binsize = nothing, windowsize = nothing, edges = nothing,
                                   kwargs...)
    n_rep > 0 || throw(ArgumentError("n_rep must be positive"))
    e, lags = _edges_and_lags(binsize, windowsize, edges)
    total = zeros(length(lags))
    for _ in 1:n_rep
        total .+= compute_crosscorrelogram(ref, Judananas.jitter(target, jitter; rng);
                                           edges = e, kwargs...).values
    end
    return (lags = lags, edges = e, values = total ./ n_rep)
end
